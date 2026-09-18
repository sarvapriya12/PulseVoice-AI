import asyncio
import json
import base64
import time
import os
import sys
import io

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
from datetime import datetime
import websockets

# Ensure we can import from the parent directory
sys.path.append(str(Path(__file__).parent.parent))

from core.config import settings
from kokoro_onnx import Kokoro

base_dir = Path(__file__).parent.parent
model_file = base_dir / "local_model" / "kokoro-v1.0.onnx"
voices_file = base_dir / "local_model" / "voices-v1.0.bin"

try:
    kokoro = Kokoro(str(model_file), str(voices_file))
except Exception as e:
    print(f"Failed to load Kokoro TTS model for testing: {e}")
    sys.exit(1)

# Categorized test suite to test:
# 1. Base Cache Priming
# 2. Semantic Cache Hits (rephrased queries that mean the same thing)
# 3. Dynamic RAG Retrieval
# 4. Guardrail / Refusal Interception
TEST_SUITE = [
    # ── Category 1: Base Question (Prime Cache) ──
    {
        "category": "Cache Priming",
        "question": "What are your business hours?",
        "expected_cache": False
    },
    # ── Category 2: Semantic Equivalence (Should Hit Semantic Cache) ──
    {
        "category": "Semantic Cache Hit",
        "question": "When does your clinic open and close today?",
        "expected_cache": True
    },
    {
        "category": "Semantic Cache Hit",
        "question": "At what time do you close?",
        "expected_cache": True
    },
    # ── Category 3: Conversational Question & Rephrasing ──
    {
        "category": "Greeting Priming",
        "question": "Hello, are you a real person or an AI assistant?",
        "expected_cache": False
    },
    {
        "category": "Semantic Cache Hit",
        "question": "Hi there, are you a robot or a human?",
        "expected_cache": True
    },
    # ── Category 4: Dynamic RAG FAQ Retrieval ──
    {
        "category": "RAG Knowledge Base",
        "question": "Do you accept Blue Cross Blue Shield insurance?",
        "expected_cache": False
    },
    # ── Category 5: Guardrail & Out-of-Domain Interception ──
    {
        "category": "Guardrail Check",
        "question": "Can you explain how black holes work in physics?",
        "expected_cache": False
    },
    # ── Category 6: Appointment Intent (Tool Calling) ──
    {
        "category": "DB Tool Mutation",
        "question": "I would like to check available slots for tomorrow morning.",
        "expected_cache": False
    }
]

from scipy.signal import resample

async def generate_audio(text: str) -> bytes:
    """Generates 16000Hz raw PCM audio from text."""
    samples, sample_rate = kokoro.create(text, voice="af_heart", speed=1.0, lang="en-us")
    if sample_rate != 16000:
        target_len = int(len(samples) * 16000 / sample_rate)
        samples = resample(samples, target_len).astype(np.float32)
    pcm_data = (samples * 32767).astype(np.int16).tobytes()
    return pcm_data

async def run_test(test_item: dict) -> dict:
    question = test_item["question"]
    category = test_item["category"]
    print(f"\n=======================================================")
    print(f"[{category}] Testing: '{question}'")
    print("Generating audio via Kokoro TTS...")
    audio_bytes = await generate_audio(question)
    
    url = f"ws://localhost:8000/ws?token={settings.TWILIO_AUTH_TOKEN}&skip_greeting=true"
    
    try:
        async with websockets.connect(url) as ws:
            # Short pause to let transport handshake complete cleanly
            await asyncio.sleep(0.2)

            print("Streaming audio chunks to Voice Server...")
            chunk_size = 3200 # 100ms chunks at 16000Hz (1600 samples * 2 bytes)
            
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                payload = base64.b64encode(chunk).decode("utf-8")
                await ws.send(json.dumps({
                    "event": "media",
                    "streamSid": "benchmark-sid",
                    "media": {"payload": payload}
                }))
                await asyncio.sleep(0.05)
                
            # Stream 500ms of silence (zeros) so VAD detects speech end without waiting for audio idle handler
            silence_bytes = b"\x00" * 3200 * 5
            for i in range(0, len(silence_bytes), chunk_size):
                chunk = silence_bytes[i:i+chunk_size]
                payload = base64.b64encode(chunk).decode("utf-8")
                await ws.send(json.dumps({
                    "event": "media",
                    "streamSid": "benchmark-sid",
                    "media": {"payload": payload}
                }))
                await asyncio.sleep(0.04)

            end_send_time = time.time()
            print("Audio stream finished. Listening for responses...")
            
            first_text_time = None
            first_media_time = None
            ttft = -1.0
            ttfa = -1.0
            collected_response = []
            final_received = False
            
            loop_start = time.time()
            while time.time() - loop_start < 25.0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
                    data = json.loads(msg)
                    
                    event = data.get("event")
                    if event == "transcript" and data.get("role") == "ai":
                        text_chunk = data.get("text", "")
                        if not first_text_time:
                            first_text_time = time.time()
                            ttft = round(first_text_time - end_send_time, 3)
                            print(f"  ⚡ [TTFT]: {ttft:.3f}s")
                        if data.get("isFinal"):
                            collected_response.append(text_chunk)
                            final_received = True
                            if first_media_time:
                                break
                    
                    if event == "media":
                        if not first_media_time:
                            first_media_time = time.time()
                            ttfa = round(first_media_time - end_send_time, 3)
                            print(f"  🔊 [TTFA - Audio]: {ttfa:.3f}s")
                        if final_received:
                            break
                            
                except asyncio.TimeoutError:
                    if final_received:
                        break
                    if first_text_time or first_media_time:
                        continue
                    print("  ⚠️ Timeout reached while waiting for response.")
                    break
            
            final_text = " ".join(collected_response).strip()
            print(f"  💬 Bot Response: \"{final_text[:120]}...\"" if len(final_text) > 120 else f"  💬 Bot Response: \"{final_text}\"")
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "category": category,
                "question": question,
                "expected_cache": test_item["expected_cache"],
                "ttft_seconds": ttft,
                "ttfa_seconds": ttfa,
                "response": final_text
            }
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

async def main():
    print("=" * 60)
    print("🚀 STARTING VOICE SERVER COMPREHENSIVE BENCHMARK")
    print("Testing: Semantic Cache Hits, RAG Lookups, Guardrails & Latency")
    print("=" * 60)
    
    metrics = []
    log_file = Path(__file__).parent / "latency_metrics.log"
    
    for item in TEST_SUITE:
        res = await run_test(item)
        if res:
            metrics.append(res)
        await asyncio.sleep(1.5) # Inter-test rest
        
    if not metrics:
        print("\nNo metrics recorded. Ensure Voice Server is running at localhost:8000.")
        return
        
    # Append structured log
    with open(log_file, "a", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
            
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    for m in metrics:
        cache_indicator = "⚡ CACHE TARGET" if m["expected_cache"] else "🔄 COLD PATH"
        print(f"[{m['category']:<20}] {cache_indicator} | TTFT: {m['ttft_seconds']:>6.3f}s | TTFA: {m['ttfa_seconds']:>6.3f}s | Q: {m['question'][:30]}...")

    cache_ttfa = [m["ttfa_seconds"] for m in metrics if m["expected_cache"] and m["ttfa_seconds"] > 0]
    cold_ttfa = [m["ttfa_seconds"] for m in metrics if not m["expected_cache"] and m["ttfa_seconds"] > 0]
    
    if cold_ttfa:
        print(f"\nAverage Cold Path Audio Latency (TTFA): {sum(cold_ttfa)/len(cold_ttfa):.3f}s")
    if cache_ttfa:
        print(f"Average Semantic Cache Audio Latency (TTFA): {sum(cache_ttfa)/len(cache_ttfa):.3f}s")
    print(f"\nDetailed metrics saved to: {log_file}")

if __name__ == "__main__":
    asyncio.run(main())
