"""
Live Voice Server End-to-End Benchmark Runner.
Executes the full conversational round-trip through the running FastAPI + Pipecat + LangGraph WebSocket pipeline:
Caller Text -> Kokoro TTS (Caller Voice) -> WebSocket Stream (ws://localhost:8000/ws) -> Live Voice Server (Silero VAD + Whisper STT + LangGraph Agent + RAG + Server Kokoro TTS) -> Client WebSocket Receiver -> TTFT, TTFA, STT Accuracy, and Response Audio Metrics.
"""

import os
import sys
import time
import json
import csv
import base64
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.signal
import soundfile as sf
import websockets

# Add repository root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.config import settings
from kokoro_onnx import Kokoro
from tests.datasets.medical_questions import get_questions, MEDICAL_QUESTIONS
from services.audio_metrics import (
    calculate_amplitude_metrics,
    calculate_recognition_metrics,
    normalize_clinical_text
)


def compute_statistical_summary(values: List[float]) -> Dict[str, float]:
    """Computes Mean, Std, Median, P95, Min, and Max for a metric series."""
    valid_vals = [v for v in values if v is not None and not np.isnan(v)]
    if not valid_vals:
        return {"mean": None, "std": None, "median": None, "p95": None, "min": None, "max": None}
    arr = np.array(valid_vals)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4)
    }


async def generate_caller_audio(kokoro: Kokoro, text: str, voice: str = "af_sarah", speed: float = 1.0, target_sr: int = 24000) -> Tuple[bytes, np.ndarray]:
    """Synthesizes question into mono int16 PCM audio at target_sr."""
    samples, sr = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    if sr != target_sr:
        target_len = int(len(samples) * target_sr / sr)
        samples_out = scipy.signal.resample(samples, target_len).astype(np.float32)
    else:
        samples_out = samples.astype(np.float32)

    pcm_int16 = (np.clip(samples_out, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm_int16.tobytes(), samples_out


async def stream_question_to_voice_server(
    pcm_bytes: bytes,
    ws_url: str,
    chunk_ms: float = 100.0,
    trailing_silence_ms: float = 500.0,
    timeout_s: float = 25.0,
    sample_rate: int = 24000
) -> Dict[str, Any]:
    """
    Streams caller audio to the live voice server WebSocket and collects live metrics:
    - User STT transcript emitted by the server
    - TTFT (Time to First Token)
    - TTFA (Time to First Audio frame)
    - AI response text
    - AI response audio bytes
    - Total round-trip latency
    """
    chunk_size = int(sample_rate * (chunk_ms / 1000.0) * 2)  # 2 bytes per int16 sample
    trailing_size = int(sample_rate * (trailing_silence_ms / 1000.0) * 2)

    user_transcript: Optional[str] = None
    ai_tokens: List[str] = []
    ai_audio_chunks: List[bytes] = []
    ttft: Optional[float] = None
    ttfa: Optional[float] = None
    final_ai_received = False

    t_start_connection = time.perf_counter()

    async with websockets.connect(ws_url) as ws:
        # Allow WebSocket transport handshake to settle
        await asyncio.sleep(0.15)

        t_start_streaming = time.perf_counter()

        # Stream caller audio chunks
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            b64_payload = base64.b64encode(chunk).decode("utf-8")
            await ws.send(json.dumps({
                "event": "media",
                "streamSid": "benchmark-sid",
                "media": {"payload": b64_payload, "sampleRate": sample_rate}
            }))
            await asyncio.sleep(chunk_ms / 2000.0)  # Real-time streaming simulation

        # Stream trailing silence so server VAD reliably detects speech endpoint
        silence_bytes = b"\x00" * trailing_size
        for i in range(0, len(silence_bytes), chunk_size):
            chunk = silence_bytes[i : i + chunk_size]
            b64_payload = base64.b64encode(chunk).decode("utf-8")
            await ws.send(json.dumps({
                "event": "media",
                "streamSid": "benchmark-sid",
                "media": {"payload": b64_payload, "sampleRate": sample_rate}
            }))
            await asyncio.sleep(chunk_ms / 2500.0)

        t_end_stream = time.perf_counter()
        stream_duration_ms = (t_end_stream - t_start_streaming) * 1000.0

        # Receive responses from server
        loop_start = time.perf_counter()
        while (time.perf_counter() - loop_start) < timeout_s:
            try:
                remaining_time = max(1.0, timeout_s - (time.perf_counter() - loop_start))
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=min(15.0, remaining_time))
                data = json.loads(raw_msg)
                event = data.get("event")

                if event == "transcript":
                    role = data.get("role")
                    text = data.get("text", "")
                    if role == "user":
                        # This is what the live server's STT heard!
                        user_transcript = text
                    elif role == "ai":
                        if ttft is None:
                            ttft = round((time.perf_counter() - t_end_stream) * 1000.0, 2)
                        ai_tokens.append(text)
                        if data.get("isFinal"):
                            final_ai_received = True
                            # If audio has already arrived, we can exit cleanly
                            if ttfa is not None:
                                break

                elif event == "media":
                    if ttfa is None:
                        ttfa = round((time.perf_counter() - t_end_stream) * 1000.0, 2)
                    payload = data.get("media", {}).get("payload", "")
                    if payload:
                        ai_audio_chunks.append(base64.b64decode(payload))
                    if final_ai_received:
                        # Allow a brief moment to catch any remaining buffered frames
                        await asyncio.sleep(0.2)
                        break

            except asyncio.TimeoutError:
                if final_ai_received:
                    break
                if (time.perf_counter() - loop_start) < timeout_s:
                    continue
                break

    t_total_end = time.perf_counter()
    total_roundtrip_ms = round((t_total_end - t_end_stream) * 1000.0, 2)

    full_ai_text = " ".join(ai_tokens).strip()

    return {
        "user_stt_transcript": user_transcript or "",
        "ai_response_text": full_ai_text,
        "ttft_ms": ttft,
        "ttfa_ms": ttfa,
        "total_roundtrip_ms": total_roundtrip_ms,
        "stream_duration_ms": round(stream_duration_ms, 2),
        "ai_audio_chunks_count": len(ai_audio_chunks),
        "ai_audio_bytes": b"".join(ai_audio_chunks)
    }


async def run_single_live_test(
    item: Dict[str, Any],
    kokoro: Kokoro,
    ws_url: str,
    output_dir: Path,
    voice: str = "af_sarah",
    speed: float = 1.0,
    sample_rate: int = 24000
) -> Dict[str, Any]:
    """Runs a single clinical inquiry through the live WebSocket voice-server."""
    test_id = item["id"]
    cat = item.get("category", "general")
    ref_text = item["text"]
    med_terms = item.get("medical_terms", [])
    test_dir = output_dir / test_id
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Synthesize caller audio
        t_synth0 = time.perf_counter()
        pcm_bytes, samples_np = await generate_caller_audio(kokoro, ref_text, voice=voice, speed=speed, target_sr=sample_rate)
        synth_ms = round((time.perf_counter() - t_synth0) * 1000.0, 2)

        caller_wav_path = test_dir / "caller_input.wav"
        sf.write(str(caller_wav_path), samples_np, sample_rate)

        # 2. Stream to live Voice Server
        server_res = await stream_question_to_voice_server(pcm_bytes, ws_url, sample_rate=sample_rate)

        # 3. Save response audio if received
        ai_audio_bytes = server_res["ai_audio_bytes"]
        if ai_audio_bytes:
            ai_audio_np = np.frombuffer(ai_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            resp_wav_path = test_dir / "server_response_audio.wav"
            sf.write(str(resp_wav_path), ai_audio_np, 24000)
            ai_amp_metrics = calculate_amplitude_metrics(ai_audio_np, 24000)
        else:
            ai_amp_metrics = None

        # 4. Compute STT Accuracy on what the live server transcribed
        stt_heard = server_res["user_stt_transcript"]
        rec_metrics = calculate_recognition_metrics(ref_text, stt_heard, med_terms)

        # 5. Outlier & Performance Flags
        flags = []
        wer = rec_metrics["word_error_rate"]
        if wer > 0.25:
            flags.append(f"HIGH_LIVE_WER (WER={wer:.3f})")
        if rec_metrics["medical_terminology"].get("missed_terms"):
            missed = rec_metrics["medical_terminology"]["missed_terms"]
            flags.append(f"MISSED_MEDICAL_TERMS ({', '.join(missed)})")
        if server_res["ttfa_ms"] is not None and server_res["ttfa_ms"] > 3500.0:
            flags.append(f"HIGH_TTFA ({server_res['ttfa_ms']:.0f}ms > 3500ms)")
        if not server_res["ai_response_text"]:
            flags.append("EMPTY_SERVER_RESPONSE")

        full_result = {
            "test_id": test_id,
            "category": cat,
            "status": "PASSED",
            "reference_text": ref_text,
            "server_stt_transcript": stt_heard,
            "server_ai_response": server_res["ai_response_text"],
            "timing": {
                "caller_synth_ms": synth_ms,
                "ttft_ms": server_res["ttft_ms"],
                "ttfa_ms": server_res["ttfa_ms"],
                "total_roundtrip_ms": server_res["total_roundtrip_ms"]
            },
            "live_stt_recognition": rec_metrics,
            "response_audio_amplitude": ai_amp_metrics,
            "outlier_flags": flags,
            "is_outlier": len(flags) > 0
        }

        with open(test_dir / "live_metrics.json", "w", encoding="utf-8") as f:
            json.dump(full_result, f, indent=2)

        return full_result

    except Exception as e:
        err_res = {
            "test_id": test_id,
            "category": cat,
            "status": "FAILED",
            "error": str(e),
            "reference_text": ref_text,
            "server_stt_transcript": "",
            "server_ai_response": "",
            "timing": {},
            "outlier_flags": [f"LIVE_PIPELINE_EXCEPTION: {e}"],
            "is_outlier": True
        }
        with open(test_dir / "live_metrics.json", "w", encoding="utf-8") as f:
            json.dump(err_res, f, indent=2)
        return err_res


def generate_live_aggregate_reports(results: List[Dict[str, Any]], output_dir: Path, metadata: Dict[str, Any]):
    """Generates aggregate JSON, Markdown, and CSV reports for the live voice-server pipeline."""
    passed = [r for r in results if r.get("status") == "PASSED"]
    failed = [r for r in results if r.get("status") == "FAILED"]
    outliers = [r for r in results if r.get("is_outlier", False)]

    ttft_vals = [r["timing"]["ttft_ms"] for r in passed if r["timing"].get("ttft_ms") is not None]
    ttfa_vals = [r["timing"]["ttfa_ms"] for r in passed if r["timing"].get("ttfa_ms") is not None]
    roundtrip_vals = [r["timing"]["total_roundtrip_ms"] for r in passed if r["timing"].get("total_roundtrip_ms") is not None]
    wer_vals = [r["live_stt_recognition"]["word_error_rate"] for r in passed]
    cer_vals = [r["live_stt_recognition"]["character_error_rate"] for r in passed]
    med_acc_vals = [
        r["live_stt_recognition"]["medical_terminology"].get("medical_term_accuracy", 1.0)
        for r in passed if "medical_terminology" in r["live_stt_recognition"]
    ]

    summary = {
        "live_wer": compute_statistical_summary(wer_vals),
        "live_cer": compute_statistical_summary(cer_vals),
        "live_med_acc": compute_statistical_summary(med_acc_vals),
        "ttft_ms": compute_statistical_summary(ttft_vals),
        "ttfa_ms": compute_statistical_summary(ttfa_vals),
        "total_roundtrip_ms": compute_statistical_summary(roundtrip_vals)
    }

    # Category breakdown
    cat_breakdown = {}
    for cat in sorted(set(r["category"] for r in results)):
        cat_p = [r for r in passed if r["category"] == cat]
        if not cat_p:
            continue
        c_ttft = [r["timing"]["ttft_ms"] for r in cat_p if r["timing"].get("ttft_ms") is not None]
        c_ttfa = [r["timing"]["ttfa_ms"] for r in cat_p if r["timing"].get("ttfa_ms") is not None]
        cat_breakdown[cat] = {
            "total": len([r for r in results if r["category"] == cat]),
            "passed": len(cat_p),
            "mean_wer": round(float(np.mean([r["live_stt_recognition"]["word_error_rate"] for r in cat_p])), 4),
            "mean_med_acc": round(float(np.mean([r["live_stt_recognition"]["medical_terminology"].get("medical_term_accuracy", 1.0) for r in cat_p])), 4),
            "mean_ttft_ms": round(float(np.mean(c_ttft)), 1) if c_ttft else None,
            "mean_ttfa_ms": round(float(np.mean(c_ttfa)), 1) if c_ttfa else None,
            "mean_roundtrip_ms": round(float(np.mean([r["timing"]["total_roundtrip_ms"] for r in cat_p])), 1)
        }

    agg_data = {
        "metadata": metadata,
        "counts": {
            "total_questions": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "outliers": len(outliers)
        },
        "statistical_summary": summary,
        "category_breakdown": cat_breakdown,
        "outliers": [
            {
                "test_id": r["test_id"],
                "category": r["category"],
                "status": r["status"],
                "flags": r.get("outlier_flags", []),
                "wer": r.get("live_stt_recognition", {}).get("word_error_rate"),
                "ttfa_ms": r.get("timing", {}).get("ttfa_ms"),
                "ref": r.get("reference_text"),
                "server_stt": r.get("server_stt_transcript"),
                "server_ai": r.get("server_ai_response")
            }
            for r in outliers
        ]
    }

    # Save JSON
    with open(output_dir / "live_aggregate_report.json", "w", encoding="utf-8") as f:
        json.dump(agg_data, f, indent=2)

    # Save CSV
    csv_rows = []
    for r in results:
        is_p = (r.get("status") == "PASSED")
        csv_rows.append({
            "test_id": r["test_id"],
            "category": r["category"],
            "status": r["status"],
            "is_outlier": r.get("is_outlier", False),
            "flags": "; ".join(r.get("outlier_flags", [])),
            "wer": r["live_stt_recognition"]["word_error_rate"] if is_p else None,
            "cer": r["live_stt_recognition"]["character_error_rate"] if is_p else None,
            "med_term_acc": r["live_stt_recognition"]["medical_terminology"].get("medical_term_accuracy") if is_p else None,
            "ttft_ms": r["timing"].get("ttft_ms") if is_p else None,
            "ttfa_ms": r["timing"].get("ttfa_ms") if is_p else None,
            "roundtrip_ms": r["timing"].get("total_roundtrip_ms") if is_p else None,
            "server_stt": r.get("server_stt_transcript", ""),
            "server_ai_response": r.get("server_ai_response", "")[:100]
        })
    if csv_rows:
        with open(output_dir / "live_aggregate_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    # Save Markdown
    md = f"""# Live Voice Server End-to-End Pipeline Benchmark Report

**Run ID**: `{metadata['run_id']}`  
**Target Server**: `{metadata['ws_url']}`  
**Caller Voice Generator**: Kokoro-82M ONNX (`{metadata['caller_voice']}`)  
**Server Pipeline**: Pipecat + Silero VAD + Faster-Whisper + LangGraph Agent / RAG + Kokoro TTS  

---

## 1. Executive Performance Summary

| Total Questions | Passed | Failed | Outliers | Mean Live WER | Medical Term Accuracy | Median TTFT | Median TTFA | Median Roundtrip Latency |
|---|---|---|---|---|---|---|---|---|
| **{len(results)}** | **{len(passed)}** | **{len(failed)}** | **{len(outliers)}** | **{summary['live_wer']['mean']:.4f}** | **{summary['live_med_acc']['mean'] * 100:.1f}%** | **{summary['ttft_ms']['median']} ms** | **{summary['ttfa_ms']['median']} ms** | **{summary['total_roundtrip_ms']['median']} ms** |

---

## 2. Latency & Accuracy Distributions

| Metric | Mean | Std Dev | Median | P95 | Min | Max |
|---|---|---|---|---|---|---|
| **Live STT WER** | {summary['live_wer']['mean']} | {summary['live_wer']['std']} | {summary['live_wer']['median']} | {summary['live_wer']['p95']} | {summary['live_wer']['min']} | {summary['live_wer']['max']} |
| **Live STT CER** | {summary['live_cer']['mean']} | {summary['live_cer']['std']} | {summary['live_cer']['median']} | {summary['live_cer']['p95']} | {summary['live_cer']['min']} | {summary['live_cer']['max']} |
| **Medical Term Accuracy** | {summary['live_med_acc']['mean']} | {summary['live_med_acc']['std']} | {summary['live_med_acc']['median']} | {summary['live_med_acc']['p95']} | {summary['live_med_acc']['min']} | {summary['live_med_acc']['max']} |
| **TTFT (Time to First Token)** | {summary['ttft_ms']['mean']} ms | {summary['ttft_ms']['std']} ms | {summary['ttft_ms']['median']} ms | {summary['ttft_ms']['p95']} ms | {summary['ttft_ms']['min']} ms | {summary['ttft_ms']['max']} ms |
| **TTFA (Time to First Audio)** | {summary['ttfa_ms']['mean']} ms | {summary['ttfa_ms']['std']} ms | {summary['ttfa_ms']['median']} ms | {summary['ttfa_ms']['p95']} ms | {summary['ttfa_ms']['min']} ms | {summary['ttfa_ms']['max']} ms |
| **Total Roundtrip Latency** | {summary['total_roundtrip_ms']['mean']} ms | {summary['total_roundtrip_ms']['std']} ms | {summary['total_roundtrip_ms']['median']} ms | {summary['total_roundtrip_ms']['p95']} ms | {summary['total_roundtrip_ms']['min']} ms | {summary['total_roundtrip_ms']['max']} ms |

---

## 3. Clinical Category Performance

| Category | Passed / Total | Mean WER | Med Acc | Mean TTFT | Mean TTFA | Mean Roundtrip |
|---|---|---|---|---|---|---|
"""
    for cat, d in cat_breakdown.items():
        ttft_str = f"{d['mean_ttft_ms']:.0f} ms" if d['mean_ttft_ms'] else "N/A"
        ttfa_str = f"{d['mean_ttfa_ms']:.0f} ms" if d['mean_ttfa_ms'] else "N/A"
        md += f"| `{cat}` | {d['passed']} / {d['total']} | **{d['mean_wer']:.4f}** | **{d['mean_med_acc'] * 100:.1f}%** | {ttft_str} | {ttfa_str} | {d['mean_roundtrip_ms']:.0f} ms |\n"

    md += f"""
---

## 4. Live Pipeline Outliers & Regressions

**Total Flagged Cases**: {len(outliers)}

"""
    if agg_data["outliers"]:
        md += "| Test ID | Category | Status | Outlier Flags | Live WER | TTFA | Server Transcript vs Server AI Answer |\n|---|---|---|---|---|---|---|\n"
        for o in agg_data["outliers"]:
            flags_str = "<br>".join(o.get("flags", []))
            wer_val = f"{o['wer']:.3f}" if o.get("wer") is not None else "N/A"
            ttfa_val = f"{o['ttfa_ms']:.0f} ms" if o.get("ttfa_ms") is not None else "N/A"
            md += f"| `{o['test_id']}` | `{o['category']}` | **{o['status']}** | {flags_str} | {wer_val} | {ttfa_val} | **STT**: {o.get('server_stt', '')[:60]}...<br>**AI**: {o.get('server_ai', '')[:60]}... |\n"

    with open(output_dir / "live_aggregate_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[REPORTS] Saved live pipeline reports to {output_dir}")


async def main_async():
    parser = argparse.ArgumentParser(description="Live Voice Server Benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run 5-sample smoke test against live server")
    parser.add_argument("--count", type=int, default=None, help="Limit question count")
    parser.add_argument("--category", type=str, default=None, help="Filter category")
    parser.add_argument("--caller-voice", type=str, default="af_sarah", help="Kokoro voice for caller (default: af_sarah)")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Caller audio streaming sample rate in Hz (default: 24000)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    run_id = f"live_run_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = BASE_DIR / "artifacts" / "live_voice_server_test" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    ws_url = f"ws://localhost:8000/ws?token={settings.TWILIO_AUTH_TOKEN}&skip_greeting=true&sample_rate={args.sample_rate}"

    if args.smoke:
        smoke_ids = ["med_001", "med_011", "med_021", "med_029", "med_045"]
        questions = [q for q in MEDICAL_QUESTIONS if q["id"] in smoke_ids]
        print(f"[LIVE BENCHMARK] Running live smoke test with {len(questions)} questions at {args.sample_rate}Hz.")
    else:
        questions = get_questions(count=args.count, category=args.category)
        print(f"[LIVE BENCHMARK] Running live benchmark on {len(questions)} questions at {args.sample_rate}Hz against {ws_url}.")

    print("[INIT] Loading caller voice generator (Kokoro ONNX)...")
    model_path = BASE_DIR / "local_model" / "kokoro-v1.0.onnx"
    voices_path = BASE_DIR / "local_model" / "voices-v1.0.bin"
    kokoro = Kokoro(str(model_path), str(voices_path))

    metadata = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "ws_url": ws_url,
        "caller_voice": args.caller_voice,
        "sample_rate": args.sample_rate,
        "total_questions": len(questions)
    }

    results = []
    print("\n" + "=" * 70)
    print(f"  STARTING LIVE VOICE SERVER BENCHMARK (Run ID: {run_id} | {args.sample_rate}Hz)")
    print("=" * 70 + "\n")

    for idx, q in enumerate(questions, 1):
        test_id = q["id"]
        cat = q["category"]
        print(f"[{idx}/{len(questions)}] Live streaming {test_id} ({cat})...")

        res = await run_single_live_test(
            q,
            kokoro,
            ws_url,
            output_dir,
            voice=args.caller_voice,
            sample_rate=args.sample_rate
        )
        results.append(res)

        if res["status"] == "PASSED":
            wer = res["live_stt_recognition"]["word_error_rate"]
            ttft = res["timing"]["ttft_ms"]
            ttfa = res["timing"]["ttfa_ms"]
            stt = res["server_stt_transcript"]
            ai_ans = res["server_ai_response"][:50]
            ttft_str = f"{ttft:.0f}ms" if ttft else "N/A"
            ttfa_str = f"{ttfa:.0f}ms" if ttfa else "N/A"
            outlier_tag = f" [OUTLIER: {len(res['outlier_flags'])} flags]" if res.get("is_outlier") else ""
            print(f"    -> PASSED | WER: {wer:.3f} | TTFT: {ttft_str} | TTFA: {ttfa_str}{outlier_tag}")
            safe_stt = stt.encode("ascii", "replace").decode("ascii")
            print(f"       STT: \"{safe_stt}\"")
            safe_ai = ai_ans.encode("ascii", "replace").decode("ascii")
            print(f"       AI : \"{safe_ai}...\"")
        else:
            print(f"    -> FAILED | Error: {res.get('error')}")

        # Short cool-down between calls so LangGraph session resets cleanly
        await asyncio.sleep(0.5)

    print("\n" + "=" * 70)
    print("  LIVE BENCHMARK EXECUTION COMPLETE. GENERATING REPORTS...")
    print("=" * 70 + "\n")

    generate_live_aggregate_reports(results, output_dir, metadata)

    passed_count = len([r for r in results if r.get("status") == "PASSED"])
    print("Live Executive Summary:")
    print(f"  Total Questions: {len(results)}")
    print(f"  Passed: {passed_count} | Failed: {len(results) - passed_count}")
    print(f"  Artifact Directory: {output_dir}")
    print(f"  Report Markdown: {output_dir / 'live_aggregate_report.md'}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
