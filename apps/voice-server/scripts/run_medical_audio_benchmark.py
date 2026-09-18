"""
End-to-End Automated Benchmark & Quality Assessment Suite for Clinical Voice AI.
Pipeline:
Medical Question -> Kokoro TTS (24kHz) -> Resample (16kHz) -> Streaming VAD Pipeline (Silero) -> STT (Faster-Whisper) -> Signal & Quality Metrics -> Per-Test Artifacts & Aggregate Reports.
"""

import os
import sys
import time
import json
import csv
import platform
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import scipy.signal
import soundfile as sf
import psutil

# Add repository root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tests.datasets.medical_questions import get_questions, MEDICAL_QUESTIONS
from services.audio_metrics import (
    calculate_amplitude_metrics,
    calculate_spectral_metrics,
    calculate_comparative_quality_metrics,
    calculate_chunking_metrics,
    calculate_recognition_metrics,
    classify_outliers_and_failures,
    generate_diagnostic_plot,
    normalize_clinical_text
)


def get_whisper_snapshot_dir() -> Path:
    """Finds the local downloaded Faster-Whisper snapshot."""
    hf_hub = BASE_DIR / "local_model" / "huggingface_cache" / "hub" / "models--Systran--faster-whisper-small.en" / "snapshots"
    if hf_hub.exists():
        snapshots = [d for d in hf_hub.iterdir() if d.is_dir()]
        if snapshots:
            return snapshots[0]
    return Path("small.en")


def init_models(voice: str = "af_sarah"):
    """Initializes Kokoro TTS, Faster-Whisper STT, and Silero VAD."""
    print("[INIT] Initializing Kokoro TTS (ONNX)...")
    from kokoro_onnx import Kokoro
    model_path = BASE_DIR / "local_model" / "kokoro-v1.0.onnx"
    voices_path = BASE_DIR / "local_model" / "voices-v1.0.bin"
    if not model_path.exists() or not voices_path.exists():
        raise FileNotFoundError(f"Kokoro model files missing in {BASE_DIR / 'local_model'}")
    kokoro = Kokoro(str(model_path), str(voices_path))
    print("[INIT] Kokoro TTS loaded successfully.")

    print("[INIT] Initializing Faster-Whisper STT...")
    from faster_whisper import WhisperModel
    snapshot_path = get_whisper_snapshot_dir()
    whisper = WhisperModel(str(snapshot_path), device="cpu", compute_type="int8")
    print(f"[INIT] Faster-Whisper loaded successfully from {snapshot_path.name if snapshot_path.is_dir() else snapshot_path}.")

    print("[INIT] Initializing Silero VAD Analyzer...")
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    vad = SileroVADAnalyzer(params=VADParams(confidence=0.5, start_secs=0.1, stop_secs=0.2, min_volume=0.0))
    vad.set_sample_rate(16000)
    print("[INIT] Silero VAD initialized.")

    return kokoro, whisper, vad


def simulate_streaming_pipeline(
    samples_16k: np.ndarray,
    vad_analyzer: Any,
    chunk_ms: float = 100.0,
    trailing_silence_ms: float = 500.0
) -> Dict[str, Any]:
    """
    Simulates real-time chunked streaming through Silero VAD with trailing silence.
    Measures processing latencies, jitter, speech active frames, and premature cutoffs.
    """
    chunk_size = int(16000 * (chunk_ms / 1000.0))
    trailing_samples = int(16000 * (trailing_silence_ms / 1000.0))

    # Add trailing silence
    silence_padding = np.zeros(trailing_samples, dtype=np.float32)
    full_audio = np.concatenate([samples_16k, silence_padding])

    pcm_int16 = (np.clip(full_audio, -1.0, 1.0) * 32767).astype(np.int16)

    latencies_ms = []
    vad_states = []
    speech_frame_count = 0
    total_chunks = len(full_audio) // chunk_size

    # Simulate chunk ingestion
    for i in range(total_chunks):
        t0 = time.perf_counter()
        chunk = pcm_int16[i * chunk_size : (i + 1) * chunk_size]

        # Analyze in 512-sample frames
        for f_idx in range(0, len(chunk), 512):
            frame = chunk[f_idx : f_idx + 512]
            if len(frame) == 512:
                # Synchronously run analyzer logic
                state = vad_analyzer._run_analyzer(frame.tobytes())
                vad_states.append(str(state))
                if "SPEAKING" in str(state) or "STARTING" in str(state):
                    speech_frame_count += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    vad_speech_duration_s = (speech_frame_count * 512) / 16000.0

    # Premature cutoff detection: check if speech was detected right before trailing silence
    original_len = len(samples_16k)
    last_speech_idx = max(0, original_len - 1600)  # last 100ms
    last_slice = np.abs(samples_16k[last_speech_idx:])
    premature_cutoff = bool(np.mean(last_slice) > 0.10 and vad_speech_duration_s < 0.3)

    return {
        "processed_audio_16k": full_audio,
        "chunk_latencies_ms": latencies_ms,
        "mean_chunk_latency_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
        "max_chunk_latency_ms": round(float(np.max(latencies_ms)), 2) if latencies_ms else 0.0,
        "inter_chunk_jitter_ms": round(float(np.std(latencies_ms)), 2) if len(latencies_ms) > 1 else 0.0,
        "vad_speech_duration_s": round(vad_speech_duration_s, 3),
        "premature_cutoff_detected": premature_cutoff
    }


def run_single_test(
    item: Dict[str, Any],
    kokoro: Any,
    whisper: Any,
    vad: Any,
    output_dir: Path,
    voice: str = "af_sarah",
    speed: float = 1.0,
    chunk_ms: float = 100.0,
    generate_plot: bool = False
) -> Dict[str, Any]:
    """Runs a complete test case through TTS, Streaming Pipeline, STT, and Metrics Evaluation."""
    test_id = item["id"]
    category = item.get("category", "general")
    ref_text = item["text"]
    medical_terms = item.get("medical_terms", [])
    test_dir = output_dir / test_id
    test_dir.mkdir(parents=True, exist_ok=True)

    timing: Dict[str, float] = {}

    try:
        # 1. TTS Synthesis (24kHz)
        t_tts0 = time.perf_counter()
        samples_24k, sr_24k = kokoro.create(ref_text, voice=voice, speed=speed, lang="en-us")
        timing["tts_generation_ms"] = round((time.perf_counter() - t_tts0) * 1000.0, 2)
        samples_24k = samples_24k.astype(np.float32)

        tts_wav_path = test_dir / "tts.wav"
        sf.write(str(tts_wav_path), samples_24k, sr_24k)

        # 2. Resampling (24kHz -> 16kHz)
        t_res0 = time.perf_counter()
        num_16k = int(len(samples_24k) * 16000 / sr_24k)
        samples_16k = scipy.signal.resample(samples_24k, num_16k).astype(np.float32)
        timing["resample_ms"] = round((time.perf_counter() - t_res0) * 1000.0, 2)

        pipeline_input_wav_path = test_dir / "pipeline_input.wav"
        sf.write(str(pipeline_input_wav_path), samples_16k, 16000)

        # 3. Streaming Pipeline Simulation (100ms chunks + Silero VAD)
        t_pipe0 = time.perf_counter()
        pipe_res = simulate_streaming_pipeline(samples_16k, vad, chunk_ms=chunk_ms)
        timing["pipeline_streaming_ms"] = round((time.perf_counter() - t_pipe0) * 1000.0, 2)

        stt_input_audio = pipe_res["processed_audio_16k"]
        stt_input_wav_path = test_dir / "stt_input.wav"
        sf.write(str(stt_input_wav_path), stt_input_audio, 16000)

        # 4. STT Transcription (Faster-Whisper)
        t_stt0 = time.perf_counter()
        segments, info = whisper.transcribe(stt_input_audio, beam_size=1)
        hyp_text = " ".join([s.text for s in segments]).strip()
        timing["stt_transcription_ms"] = round((time.perf_counter() - t_stt0) * 1000.0, 2)

        stt_txt_path = test_dir / "stt.txt"
        with open(stt_txt_path, "w", encoding="utf-8") as f:
            f.write(hyp_text)

        # Real-time Factors
        audio_duration_s = len(samples_16k) / 16000.0
        tts_rtf = (timing["tts_generation_ms"] / 1000.0) / audio_duration_s if audio_duration_s > 0 else 0.0
        stt_rtf = (timing["stt_transcription_ms"] / 1000.0) / audio_duration_s if audio_duration_s > 0 else 0.0
        timing["total_e2e_ms"] = round(sum(timing.values()), 2)
        timing["tts_rtf"] = round(tts_rtf, 3)
        timing["stt_rtf"] = round(stt_rtf, 3)

        # 5. Objective Audio & Signal Metrics
        amp_metrics = calculate_amplitude_metrics(stt_input_audio, 16000)
        spec_metrics = calculate_spectral_metrics(stt_input_audio, 16000)
        quality_metrics = calculate_comparative_quality_metrics(samples_16k, stt_input_audio, 16000)
        chunk_metrics = calculate_chunking_metrics(stt_input_audio, int(16000 * chunk_ms / 1000.0), pipe_res["chunk_latencies_ms"])
        rec_metrics = calculate_recognition_metrics(ref_text, hyp_text, medical_terms)

        # 6. Phonetic Twin Analysis (if applicable)
        phonetic_analysis = None
        if "confusable_pair" in item:
            pair = item["confusable_pair"]
            norm_hyp = normalize_clinical_text(hyp_text).split()
            target_present = pair[0] in norm_hyp or pair[1] in norm_hyp
            confused = False
            # Check if incorrect counterpart was recognized instead of target
            ref_norm_words = normalize_clinical_text(ref_text).split()
            intended = pair[0] if pair[0] in ref_norm_words else pair[1]
            counterpart = pair[1] if intended == pair[0] else pair[0]
            if counterpart in norm_hyp and intended not in norm_hyp:
                confused = True
            phonetic_analysis = {
                "pair": pair,
                "intended_word": intended,
                "counterpart_word": counterpart,
                "transcribed_correctly": (intended in norm_hyp and not confused),
                "confused_with_counterpart": confused
            }

        # 7. Outlier & Anomaly Flagging
        full_metrics = {
            "test_id": test_id,
            "category": category,
            "status": "PASSED",
            "reference_text": ref_text,
            "hypothesis_text": hyp_text,
            "timing": timing,
            "amplitude": amp_metrics,
            "spectral": spec_metrics,
            "comparative_quality": quality_metrics,
            "streaming_chunking": chunk_metrics,
            "vad": {
                "vad_speech_duration_s": pipe_res["vad_speech_duration_s"],
                "premature_cutoff_detected": pipe_res["premature_cutoff_detected"],
                "mean_chunk_latency_ms": pipe_res["mean_chunk_latency_ms"]
            },
            "recognition": rec_metrics,
            "phonetic_twin_evaluation": phonetic_analysis
        }

        outlier_flags = classify_outliers_and_failures(full_metrics)
        if phonetic_analysis and phonetic_analysis.get("confused_with_counterpart"):
            outlier_flags.append(f"CONFUSED_PHONETIC_TWIN ({phonetic_analysis['intended_word']} -> {phonetic_analysis['counterpart_word']})")
        full_metrics["outlier_flags"] = outlier_flags
        full_metrics["is_outlier"] = len(outlier_flags) > 0

        # 8. Diagnostic Plot
        plot_path = None
        if generate_plot or full_metrics["is_outlier"]:
            plot_file = test_dir / "diagnostic_plot.png"
            plot_path = generate_diagnostic_plot(
                samples_16k, stt_input_audio, 16000, str(plot_file), test_id, title=f"[{category}] {test_id}"
            )
        full_metrics["diagnostic_plot_path"] = str(plot_path) if plot_path else None

        # 9. Save metrics.json
        with open(test_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(full_metrics, f, indent=2)

        return full_metrics

    except Exception as e:
        error_metrics = {
            "test_id": test_id,
            "category": category,
            "status": "FAILED",
            "error": str(e),
            "reference_text": ref_text,
            "hypothesis_text": "",
            "timing": timing,
            "outlier_flags": [f"TEST_EXECUTION_EXCEPTION: {e}"],
            "is_outlier": True
        }
        with open(test_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(error_metrics, f, indent=2)
        return error_metrics


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


def generate_aggregate_reports(
    results: List[Dict[str, Any]],
    output_dir: Path,
    metadata: Dict[str, Any]
):
    """Generates aggregate JSON, Markdown, and CSV reports with statistical breakdown."""
    passed_tests = [r for r in results if r.get("status") == "PASSED"]
    failed_tests = [r for r in results if r.get("status") == "FAILED"]
    outlier_tests = [r for r in results if r.get("is_outlier", False)]

    # Collect metric vectors
    metric_series = {
        "wer": [r["recognition"]["word_error_rate"] for r in passed_tests],
        "cer": [r["recognition"]["character_error_rate"] for r in passed_tests],
        "med_term_acc": [r["recognition"]["medical_terminology"].get("medical_term_accuracy", 1.0) for r in passed_tests if "medical_terminology" in r["recognition"]],
        "snr_db": [r["comparative_quality"]["snr_db"] for r in passed_tests],
        "seg_snr_db": [r["comparative_quality"]["segmental_snr_db"] for r in passed_tests],
        "si_sdr_db": [r["comparative_quality"]["si_sdr_db"] for r in passed_tests],
        "rmse": [r["comparative_quality"]["rmse"] for r in passed_tests],
        "stoi": [r["comparative_quality"]["stoi"] for r in passed_tests if r["comparative_quality"]["stoi"] is not None],
        "pesq_wb": [r["comparative_quality"]["pesq_wb"] for r in passed_tests if r["comparative_quality"]["pesq_wb"] is not None],
        "peak_dbfs": [r["amplitude"]["peak_amplitude_dbfs"] for r in passed_tests],
        "rms_dbfs": [r["amplitude"]["rms_amplitude_dbfs"] for r in passed_tests],
        "crest_factor_db": [r["amplitude"]["crest_factor_db"] for r in passed_tests],
        "clipping_ratio": [r["amplitude"]["clipping_ratio"] for r in passed_tests],
        "silence_ratio": [r["amplitude"]["silence_ratio"] for r in passed_tests],
        "speech_ratio": [r["amplitude"]["speech_ratio"] for r in passed_tests],
        "spectral_centroid_hz": [r["spectral"]["spectral_centroid_hz"] for r in passed_tests],
        "spectral_bandwidth_hz": [r["spectral"]["spectral_bandwidth_hz"] for r in passed_tests],
        "spectral_rolloff_hz": [r["spectral"]["spectral_rolloff_85_hz"] for r in passed_tests],
        "tts_latency_ms": [r["timing"]["tts_generation_ms"] for r in passed_tests],
        "stt_latency_ms": [r["timing"]["stt_transcription_ms"] for r in passed_tests],
        "total_latency_ms": [r["timing"]["total_e2e_ms"] for r in passed_tests],
        "stt_rtf": [r["timing"]["stt_rtf"] for r in passed_tests],
        "tts_rtf": [r["timing"]["tts_rtf"] for r in passed_tests]
    }

    # Summary stats per metric
    summary_stats = {k: compute_statistical_summary(v) for k, v in metric_series.items()}

    # Categorical breakdown
    categories = set(r["category"] for r in results)
    category_breakdown = {}
    for cat in sorted(categories):
        cat_passed = [r for r in passed_tests if r["category"] == cat]
        if not cat_passed:
            continue
        category_breakdown[cat] = {
            "total_count": len([r for r in results if r["category"] == cat]),
            "passed_count": len(cat_passed),
            "mean_wer": round(float(np.mean([r["recognition"]["word_error_rate"] for r in cat_passed])), 4),
            "mean_med_acc": round(float(np.mean([r["recognition"]["medical_terminology"].get("medical_term_accuracy", 1.0) for r in cat_passed])), 4),
            "mean_stoi": round(float(np.mean([r["comparative_quality"]["stoi"] for r in cat_passed if r["comparative_quality"]["stoi"] is not None])), 4),
            "mean_pesq": round(float(np.mean([r["comparative_quality"]["pesq_wb"] for r in cat_passed if r["comparative_quality"]["pesq_wb"] is not None])), 3),
            "mean_total_ms": round(float(np.mean([r["timing"]["total_e2e_ms"] for r in cat_passed])), 1)
        }

    # Phonetic twins breakdown
    phonetic_tests = [r for r in passed_tests if r.get("phonetic_twin_evaluation")]
    phonetic_total = len(phonetic_tests)
    phonetic_correct = len([r for r in phonetic_tests if r["phonetic_twin_evaluation"].get("transcribed_correctly")])
    phonetic_confused = [r for r in phonetic_tests if r["phonetic_twin_evaluation"].get("confused_with_counterpart")]
    phonetic_summary = {
        "total_pairs_tested": phonetic_total,
        "correctly_distinguished": phonetic_correct,
        "confusion_count": len(phonetic_confused),
        "accuracy": round(phonetic_correct / phonetic_total, 4) if phonetic_total > 0 else 1.0,
        "confused_cases": [
            {
                "test_id": r["test_id"],
                "intended": r["phonetic_twin_evaluation"]["intended_word"],
                "counterpart": r["phonetic_twin_evaluation"]["counterpart_word"],
                "stt_text": r["hypothesis_text"]
            }
            for r in phonetic_confused
        ]
    }

    aggregate_data = {
        "metadata": metadata,
        "counts": {
            "total_questions": len(results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "outlier_tests": len(outlier_tests)
        },
        "statistical_summary": summary_stats,
        "category_breakdown": category_breakdown,
        "phonetic_twins_analysis": phonetic_summary,
        "outliers": [
            {
                "test_id": r["test_id"],
                "category": r["category"],
                "status": r["status"],
                "flags": r.get("outlier_flags", []),
                "wer": r.get("recognition", {}).get("word_error_rate"),
                "med_acc": r.get("recognition", {}).get("medical_terminology", {}).get("medical_term_accuracy"),
                "ref": r.get("reference_text"),
                "hyp": r.get("hypothesis_text")
            }
            for r in outlier_tests
        ]
    }

    # 1. Save aggregate_report.json
    with open(output_dir / "aggregate_report.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_data, f, indent=2)

    # 2. Save aggregate_summary.csv
    csv_rows = []
    for r in results:
        is_p = (r.get("status") == "PASSED")
        csv_rows.append({
            "test_id": r["test_id"],
            "category": r["category"],
            "status": r["status"],
            "is_outlier": r.get("is_outlier", False),
            "flags": "; ".join(r.get("outlier_flags", [])),
            "wer": r["recognition"]["word_error_rate"] if is_p else None,
            "cer": r["recognition"]["character_error_rate"] if is_p else None,
            "med_term_accuracy": r["recognition"]["medical_terminology"].get("medical_term_accuracy") if is_p else None,
            "snr_db": r["comparative_quality"]["snr_db"] if is_p else None,
            "si_sdr_db": r["comparative_quality"]["si_sdr_db"] if is_p else None,
            "stoi": r["comparative_quality"]["stoi"] if is_p else None,
            "pesq_wb": r["comparative_quality"]["pesq_wb"] if is_p else None,
            "peak_dbfs": r["amplitude"]["peak_amplitude_dbfs"] if is_p else None,
            "rms_dbfs": r["amplitude"]["rms_amplitude_dbfs"] if is_p else None,
            "clipping_ratio": r["amplitude"]["clipping_ratio"] if is_p else None,
            "silence_ratio": r["amplitude"]["silence_ratio"] if is_p else None,
            "spectral_centroid_hz": r["spectral"]["spectral_centroid_hz"] if is_p else None,
            "tts_latency_ms": r["timing"]["tts_generation_ms"] if is_p else None,
            "stt_latency_ms": r["timing"]["stt_transcription_ms"] if is_p else None,
            "total_latency_ms": r["timing"]["total_e2e_ms"] if is_p else None,
            "stt_rtf": r["timing"]["stt_rtf"] if is_p else None
        })

    if csv_rows:
        with open(output_dir / "aggregate_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # 3. Save aggregate_report.md
    md_content = f"""# Medical Voice AI Benchmark & Quality Assessment Report

**Run ID**: `{metadata['run_id']}`  
**Date**: `{metadata['timestamp']}`  
**Environment**: Python `{metadata['python_version']}` on `{metadata['platform']}`  
**TTS Model**: Kokoro-82M ONNX (voice: `{metadata['tts_voice']}`, speed: `{metadata['tts_speed']}`)  
**STT Model**: Faster-Whisper small.en (CPU int8)  
**VAD**: Silero VAD (16kHz, confidence: 0.5, stop_secs: 0.2s)  

---

## 1. Executive Summary

| Total Questions | Passed Tests | Failed Tests | Flagged Outliers | Overall WER (Mean) | Medical Term Accuracy | STOI (Mean) | PESQ (Mean) | Total E2E Latency (Median) |
|---|---|---|---|---|---|---|---|---|
| **{len(results)}** | **{len(passed_tests)}** | **{len(failed_tests)}** | **{len(outlier_tests)}** | **{summary_stats['wer']['mean']:.4f}** | **{summary_stats['med_term_acc']['mean'] * 100:.1f}%** | **{summary_stats['stoi']['mean']:.3f}** | **{summary_stats['pesq_wb']['mean']:.2f}** | **{summary_stats['total_latency_ms']['median']:.1f} ms** |

> [!NOTE]
> All signal and acoustic metrics were computed on objectively aligned signals using cross-correlation lag compensation. Wideband PESQ is computed at 16kHz reference-to-processed sampling.

---

## 2. Statistical Metric Distribution

| Metric Group | Metric | Mean | Std Dev | Median | P95 | Min | Max |
|---|---|---|---|---|---|---|---|
| **Recognition & Domain** | Word Error Rate (WER) | {summary_stats['wer']['mean']} | {summary_stats['wer']['std']} | {summary_stats['wer']['median']} | {summary_stats['wer']['p95']} | {summary_stats['wer']['min']} | {summary_stats['wer']['max']} |
| | Character Error Rate (CER) | {summary_stats['cer']['mean']} | {summary_stats['cer']['std']} | {summary_stats['cer']['median']} | {summary_stats['cer']['p95']} | {summary_stats['cer']['min']} | {summary_stats['cer']['max']} |
| | Medical Term Accuracy | {summary_stats['med_term_acc']['mean']} | {summary_stats['med_term_acc']['std']} | {summary_stats['med_term_acc']['median']} | {summary_stats['med_term_acc']['p95']} | {summary_stats['med_term_acc']['min']} | {summary_stats['med_term_acc']['max']} |
| **Comparative Signal Quality** | SNR (dB) | {summary_stats['snr_db']['mean']} | {summary_stats['snr_db']['std']} | {summary_stats['snr_db']['median']} | {summary_stats['snr_db']['p95']} | {summary_stats['snr_db']['min']} | {summary_stats['snr_db']['max']} |
| | Segmental SNR (dB) | {summary_stats['seg_snr_db']['mean']} | {summary_stats['seg_snr_db']['std']} | {summary_stats['seg_snr_db']['median']} | {summary_stats['seg_snr_db']['p95']} | {summary_stats['seg_snr_db']['min']} | {summary_stats['seg_snr_db']['max']} |
| | SI-SDR (dB) | {summary_stats['si_sdr_db']['mean']} | {summary_stats['si_sdr_db']['std']} | {summary_stats['si_sdr_db']['median']} | {summary_stats['si_sdr_db']['p95']} | {summary_stats['si_sdr_db']['min']} | {summary_stats['si_sdr_db']['max']} |
| | RMSE | {summary_stats['rmse']['mean']} | {summary_stats['rmse']['std']} | {summary_stats['rmse']['median']} | {summary_stats['rmse']['p95']} | {summary_stats['rmse']['min']} | {summary_stats['rmse']['max']} |
| | STOI | {summary_stats['stoi']['mean']} | {summary_stats['stoi']['std']} | {summary_stats['stoi']['median']} | {summary_stats['stoi']['p95']} | {summary_stats['stoi']['min']} | {summary_stats['stoi']['max']} |
| | PESQ (WB) | {summary_stats['pesq_wb']['mean']} | {summary_stats['pesq_wb']['std']} | {summary_stats['pesq_wb']['median']} | {summary_stats['pesq_wb']['p95']} | {summary_stats['pesq_wb']['min']} | {summary_stats['pesq_wb']['max']} |
| **Acoustic & Waveform** | Peak Amplitude (dBFS) | {summary_stats['peak_dbfs']['mean']} | {summary_stats['peak_dbfs']['std']} | {summary_stats['peak_dbfs']['median']} | {summary_stats['peak_dbfs']['p95']} | {summary_stats['peak_dbfs']['min']} | {summary_stats['peak_dbfs']['max']} |
| | RMS Amplitude (dBFS) | {summary_stats['rms_dbfs']['mean']} | {summary_stats['rms_dbfs']['std']} | {summary_stats['rms_dbfs']['median']} | {summary_stats['rms_dbfs']['p95']} | {summary_stats['rms_dbfs']['min']} | {summary_stats['rms_dbfs']['max']} |
| | Crest Factor (dB) | {summary_stats['crest_factor_db']['mean']} | {summary_stats['crest_factor_db']['std']} | {summary_stats['crest_factor_db']['median']} | {summary_stats['crest_factor_db']['p95']} | {summary_stats['crest_factor_db']['min']} | {summary_stats['crest_factor_db']['max']} |
| | Clipping Ratio | {summary_stats['clipping_ratio']['mean']} | {summary_stats['clipping_ratio']['std']} | {summary_stats['clipping_ratio']['median']} | {summary_stats['clipping_ratio']['p95']} | {summary_stats['clipping_ratio']['min']} | {summary_stats['clipping_ratio']['max']} |
| | Silence Ratio | {summary_stats['silence_ratio']['mean']} | {summary_stats['silence_ratio']['std']} | {summary_stats['silence_ratio']['median']} | {summary_stats['silence_ratio']['p95']} | {summary_stats['silence_ratio']['min']} | {summary_stats['silence_ratio']['max']} |
| **Spectral Analysis** | Spectral Centroid (Hz) | {summary_stats['spectral_centroid_hz']['mean']} | {summary_stats['spectral_centroid_hz']['std']} | {summary_stats['spectral_centroid_hz']['median']} | {summary_stats['spectral_centroid_hz']['p95']} | {summary_stats['spectral_centroid_hz']['min']} | {summary_stats['spectral_centroid_hz']['max']} |
| | Spectral Bandwidth (Hz) | {summary_stats['spectral_bandwidth_hz']['mean']} | {summary_stats['spectral_bandwidth_hz']['std']} | {summary_stats['spectral_bandwidth_hz']['median']} | {summary_stats['spectral_bandwidth_hz']['p95']} | {summary_stats['spectral_bandwidth_hz']['min']} | {summary_stats['spectral_bandwidth_hz']['max']} |
| | Spectral Rolloff (85% Hz) | {summary_stats['spectral_rolloff_hz']['mean']} | {summary_stats['spectral_rolloff_hz']['std']} | {summary_stats['spectral_rolloff_hz']['median']} | {summary_stats['spectral_rolloff_hz']['p95']} | {summary_stats['spectral_rolloff_hz']['min']} | {summary_stats['spectral_rolloff_hz']['max']} |
| **Latency & Timing** | TTS Latency (ms) | {summary_stats['tts_latency_ms']['mean']} | {summary_stats['tts_latency_ms']['std']} | {summary_stats['tts_latency_ms']['median']} | {summary_stats['tts_latency_ms']['p95']} | {summary_stats['tts_latency_ms']['min']} | {summary_stats['tts_latency_ms']['max']} |
| | STT Latency (ms) | {summary_stats['stt_latency_ms']['mean']} | {summary_stats['stt_latency_ms']['std']} | {summary_stats['stt_latency_ms']['median']} | {summary_stats['stt_latency_ms']['p95']} | {summary_stats['stt_latency_ms']['min']} | {summary_stats['stt_latency_ms']['max']} |
| | Total E2E Latency (ms) | {summary_stats['total_latency_ms']['mean']} | {summary_stats['total_latency_ms']['std']} | {summary_stats['total_latency_ms']['median']} | {summary_stats['total_latency_ms']['p95']} | {summary_stats['total_latency_ms']['min']} | {summary_stats['total_latency_ms']['max']} |
| | STT Real-Time Factor (RTF) | {summary_stats['stt_rtf']['mean']} | {summary_stats['stt_rtf']['std']} | {summary_stats['stt_rtf']['median']} | {summary_stats['stt_rtf']['p95']} | {summary_stats['stt_rtf']['min']} | {summary_stats['stt_rtf']['max']} |

---

## 3. Clinical Category Performance

| Clinical Category | Tested | Mean WER | Medical Term Acc | Mean STOI | Mean PESQ | Mean Total Latency |
|---|---|---|---|---|---|---|
"""
    for cat, data in category_breakdown.items():
        md_content += f"| `{cat}` | {data['passed_count']} / {data['total_count']} | **{data['mean_wer']:.4f}** | **{data['mean_med_acc'] * 100:.1f}%** | {data['mean_stoi']:.3f} | {data['mean_pesq']:.2f} | {data['mean_total_ms']:.1f} ms |\n"

    md_content += f"""
---

## 4. Phonetic Twins Contrast Analysis

- **Total Confusable Pairs Tested**: {phonetic_summary['total_pairs_tested']}
- **Correctly Disambiguated**: {phonetic_summary['correctly_distinguished']} ({phonetic_summary['accuracy'] * 100:.1f}%)
- **Confusions**: {phonetic_summary['confusion_count']}

"""
    if phonetic_summary["confused_cases"]:
        md_content += "| Test ID | Intended Term | Counterpart Substituted | Actual STT Output |\n|---|---|---|---|\n"
        for c in phonetic_summary["confused_cases"]:
            md_content += f"| `{c['test_id']}` | `{c['intended']}` | `{c['counterpart']}` | \"{c['stt_text']}\" |\n"
    else:
        md_content += "> [!TIP]\n> No phonetic twin confusions detected across tested pairs (e.g. hypertension vs hypotension correctly distinguished).\n"

    md_content += f"""
---

## 5. Anomaly & Outlier Registry

**Total Flagged Outliers**: {len(outlier_tests)}

"""
    if outlier_tests:
        md_content += "| Test ID | Category | Status | Outlier Flags | WER | Med Acc | Reference vs Output |\n|---|---|---|---|---|---|---|\n"
        for o in outlier_tests:
            flags_str = "<br>".join(o.get("flags", []))
            wer_val = f"{o['wer']:.3f}" if o.get("wer") is not None else "N/A"
            acc_val = f"{o['med_acc'] * 100:.1f}%" if o.get("med_acc") is not None else "N/A"
            md_content += f"| `{o['test_id']}` | `{o['category']}` | **{o['status']}** | {flags_str} | {wer_val} | {acc_val} | **Ref**: {o.get('ref', '')[:60]}...<br>**Hyp**: {o.get('hyp', '')[:60]}... |\n"
    else:
        md_content += "> [!TIP]\n> Zero anomalies or metric outliers exceeded pre-set quality thresholds.\n"

    md_content += f"""
---

## 6. Reproducibility & Environment

```json
{json.dumps(metadata, indent=2)}
```
"""

    with open(output_dir / "aggregate_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[REPORTS] Saved aggregate reports to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Medical Voice AI Benchmark Suite")
    parser.add_argument("--smoke", action="store_true", help="Run 5-sample smoke test across diverse categories")
    parser.add_argument("--count", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--category", type=str, default=None, help="Filter by clinical category")
    parser.add_argument("--voice", type=str, default="af_sarah", help="Kokoro voice (default: af_sarah)")
    parser.add_argument("--speed", type=float, default=1.0, help="Kokoro TTS speed (default: 1.0)")
    parser.add_argument("--chunk-ms", type=float, default=100.0, help="Streaming chunk size ms (default: 100)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom artifact directory")
    args = parser.parse_args()

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = BASE_DIR / "artifacts" / "medical_audio_test" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Question selection
    if args.smoke:
        smoke_ids = ["med_001", "med_011", "med_021", "med_029", "med_037"]
        questions = [q for q in MEDICAL_QUESTIONS if q["id"] in smoke_ids]
        print(f"[BENCHMARK] Running smoke test with {len(questions)} samples across 5 distinct categories.")
    else:
        questions = get_questions(count=args.count, category=args.category)
        print(f"[BENCHMARK] Running benchmark on {len(questions)} medical questions.")

    kokoro, whisper, vad = init_models(voice=args.voice)

    # Metadata capture
    metadata = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "tts_model": "Kokoro-82M-v1.0 (ONNX)",
        "tts_voice": args.voice,
        "tts_speed": args.speed,
        "stt_model": "faster-whisper-small.en (CPU int8)",
        "vad_model": "Silero VAD (16kHz)",
        "chunk_size_ms": args.chunk_ms,
        "total_questions_configured": len(questions)
    }

    # Save reproducibility file
    with open(output_dir / "reproducibility.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    results = []
    print("\n" + "=" * 70)
    print(f"  STARTING MEDICAL VOICE BENCHMARK (Run ID: {run_id})")
    print("=" * 70 + "\n")

    for idx, q in enumerate(questions, 1):
        test_id = q["id"]
        cat = q["category"]
        print(f"[{idx}/{len(questions)}] Running {test_id} ({cat})...")

        # Generate plot for first 3 samples or any sample in smoke test
        gen_plot = (idx <= 3) or args.smoke

        res = run_single_test(
            q,
            kokoro,
            whisper,
            vad,
            output_dir,
            voice=args.voice,
            speed=args.speed,
            chunk_ms=args.chunk_ms,
            generate_plot=gen_plot
        )
        results.append(res)

        if res["status"] == "PASSED":
            wer = res["recognition"]["word_error_rate"]
            med_acc = res["recognition"]["medical_terminology"].get("medical_term_accuracy", 1.0)
            stoi = res["comparative_quality"]["stoi"]
            tot_ms = res["timing"]["total_e2e_ms"]
            outlier_tag = f" [OUTLIER: {len(res['outlier_flags'])} flags]" if res["is_outlier"] else ""
            print(f"    -> PASSED | WER: {wer:.3f} | MedAcc: {med_acc * 100:.0f}% | STOI: {stoi:.3f} | Latency: {tot_ms:.0f}ms{outlier_tag}")
        else:
            print(f"    -> FAILED | Error: {res.get('error')}")

    print("\n" + "=" * 70)
    print("  BENCHMARK EXECUTION COMPLETE. GENERATING AGGREGATE REPORTS...")
    print("=" * 70 + "\n")

    generate_aggregate_reports(results, output_dir, metadata)

    # Print executive summary to terminal
    passed = len([r for r in results if r.get("status") == "PASSED"])
    failed = len([r for r in results if r.get("status") == "FAILED"])
    outliers = len([r for r in results if r.get("is_outlier")])

    print("Executive Summary:")
    print(f"  Total Questions: {len(results)}")
    print(f"  Passed: {passed} | Failed: {failed} | Outliers: {outliers}")
    print(f"  Artifact Directory: {output_dir}")
    print(f"  Report Markdown: {output_dir / 'aggregate_report.md'}")
    print(f"  Summary CSV:     {output_dir / 'aggregate_summary.csv'}")
    print(f"  Full JSON:       {output_dir / 'aggregate_report.json'}")


if __name__ == "__main__":
    main()
