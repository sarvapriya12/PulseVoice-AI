"""The Pipecat Real-Time Audio Pipeline (LangGraph Adapter Pattern).

CRITICAL: All services (STT, TTS) MUST be instantiated inside run_bot(),
not at module level. Each WebSocket connection needs its own dedicated
service instances. Sharing a single instance across connections causes
a pipeline deadlock — the second caller hijacks the first's audio stream.
"""

import uuid
import asyncio
import random
import json
import base64
from contextlib import AsyncExitStack

import aiohttp
import numpy as np

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from core.config import settings


import re

# Abbreviations as a compiled set for O(1) lookup
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st",
    "e.g", "i.e", "vs", "etc", "approx", "dept", "est",
    "fig", "no", "vol", "lt", "capt", "sgt", "mt",
}

# Matches any terminal punctuation, used only for a quick early-exit check.
_HAS_TERMINAL = re.compile(r'[.!?]')

EMERGENCY_WORDS = {"emergency", "blood", "bleeding", "hurt", "injury", "pain", "dying", "heart attack", "breathe", "chest", "ambulance"}
FILLERS = ["Let me check my system.", "One moment, please.", "I'm pulling up that information."]
_CLEAN_UI = re.compile(r'[*_#|]+|<[^>]+>|-{2,}')
FALLBACK_TEXT = "I'm sorry, my system encountered a temporary error. Could you please repeat that?"

def _is_sentence_boundary(text: str, pos: int) -> bool:
    """
    Returns True if the character at `pos` is a genuine sentence boundary.
    pos must point to a '.', '!', or '?' character.
    """
    char = text[pos]

    # '!' and '?' are almost always boundaries (skip abbreviation check)
    if char in ('!', '?'):
        return True

    # For '.', check whether it's an abbreviation
    # Grab the word immediately before the dot
    start = pos - 1
    while start >= 0 and text[start].isalpha():
        start -= 1
    word_before = text[start + 1 : pos].lower()

    if word_before in ABBREVIATIONS:
        return False

    # Single capital letter followed by dot = initial (e.g. "J. Smith") → skip
    if len(word_before) == 1 and text[start + 1].isupper():
        return False

    # Digit followed by dot = list marker ("1. Item") → skip
    if start >= 0 and text[start].isdigit():
        return False

    return True


class AggressiveChunker:
    """
    Drop-in replacement for Pipecat's SentenceAggregator.
    Instantiate once per LLM turn, call feed() on each token,
    and drain() at the end.
    """

    # Markdown / formatting artifacts to strip before buffering
    _CLEAN = re.compile(r'[*_#|<>`\\-]+')

    def __init__(self):
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        """
        Accepts a raw LLM token. Returns a (possibly empty) list of
        complete sentences ready to push to TTS.
        """
        # 1. Strip formatting noise
        clean = self._CLEAN.sub('', token)
        if not clean:
            return []

        self._buf += clean

        # 2. Quick exit: no terminal punctuation in buffer at all
        if not _HAS_TERMINAL.search(self._buf):
            return []

        # 3. Scan for genuine boundaries
        sentences = []
        scan_start = 0

        for i, ch in enumerate(self._buf):
            if ch in '.!?' and _is_sentence_boundary(self._buf, i):
                # Slice out the complete sentence INCLUDING the punctuation
                sentence = self._buf[scan_start : i + 1].strip()
                if sentence:
                    sentences.append(sentence)
                scan_start = i + 1

        # Keep the overflow (next sentence in progress) in the buffer
        self._buf = self._buf[scan_start:].lstrip()
        return sentences

    def drain(self) -> list[str]:
        """
        Call when the LLM signals end-of-turn. Flushes whatever remains.
        """
        remainder = self._buf.strip()
        self._buf = ""
        return [remainder] if remainder else []


# ── Semantic Response Cache (Orchestration Layer Caching) ───────────────────
# Uses embedding cosine similarity to match rephrased queries ("how fast you sprint" vs "how fast you run").
class SemanticResponseCache:
    def __init__(self, similarity_threshold: float = 0.90, max_size: int = 250):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self._entries: list[dict] = []  # [{query: str, embedding: np.ndarray, response: str}]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def get(self, query: str, query_embedding: list[float] = None) -> tuple[str | None, float]:
        if not self._entries or not query_embedding:
            return None, 0.0
        q_vec = np.array(query_embedding, dtype=np.float32)
        best_score = -1.0
        best_resp = None
        for entry in self._entries:
            sim = self._cosine_similarity(q_vec, entry["embedding"])
            if sim > best_score:
                best_score = sim
                best_resp = entry["response"]
        if best_score >= self.similarity_threshold:
            return best_resp, best_score
        return None, best_score

    def put(self, query: str, query_embedding: list[float], response: str):
        if not query_embedding or not response:
            return
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)
        self._entries.append({
            "query": query,
            "embedding": np.array(query_embedding, dtype=np.float32),
            "response": response
        })

_SEMANTIC_RESPONSE_CACHE = SemanticResponseCache(similarity_threshold=0.88)



# ── LangGraph Adapter ────────────────────────────────────────────────────────

class LangGraphAdapter(FrameProcessor):
    """Bridges Pipecat and LangGraph — receives transcribed text, returns AI response."""

    def __init__(self, langgraph_app, call_sid: str, transport: FastAPIWebsocketTransport, tavus_url: str = None, skip_greeting: bool = False, websocket=None):
        super().__init__()
        self.langgraph_app = langgraph_app
        self.call_sid = call_sid
        self.transport = transport
        self.tavus_url = tavus_url
        self.skip_greeting = skip_greeting
        self.websocket = websocket
        self._buffer = []
        self._debounce_task = None
        self.was_interrupted = False
        self._chunker = None
        # Speculative prefetch: start RAG on first interim transcript so results
        # may already be warm by the time the final transcript arrives.
        self._speculative_rag_task: asyncio.Task | None = None
        self._speculative_rag_result = None

    async def _send_transcript(self, msg: dict):
        if getattr(self, "websocket", None):
            try:
                await self.websocket.send_json(msg)
                return
            except Exception:
                pass
        await self.push_frame(OutputTransportMessageFrame(msg))

    async def _process_final_transcript(self, text: str):
        try:
            print(f"[STT] Patient: {text}")
            await self._send_transcript({"event": "transcript", "role": "user", "text": text, "isFinal": True})

            text_lower = text.lower()
            is_emergency = any(w in text_lower for w in EMERGENCY_WORDS)
            pending_filler = None if is_emergency else random.choice(FILLERS)

            await self.push_frame(OutputTransportMessageFrame({"event": "status", "text": "Thinking..."}))

            user_msg = text
            if self.was_interrupted:
                user_msg = f"[SYSTEM NOTE: You were cut off mid-sentence. Address their new input directly.]\n{text}"
                self.was_interrupted = False

            # Caching Layer 1: True Semantic Response Cache (Cosine Distance)
            # Short-circuit semantically equivalent inquiries ("how fast you sprint" vs "how fast you run")
            query_embedding = None
            from services.langgraph_agent.nodes import rag_retriever
            if rag_retriever and hasattr(rag_retriever, "embeddings"):
                try:
                    query_embedding = rag_retriever.embeddings.embed_query(text)
                except Exception as e:
                    print(f"[CACHE] Warning: could not embed query: {e}")

            if not is_emergency and query_embedding:
                cached_reply, score = _SEMANTIC_RESPONSE_CACHE.get(text, query_embedding)
                if cached_reply:
                    print(f"[SEMANTIC CACHE HIT] (score: {score:.3f}) Reusing reply for: '{text[:35]}'")
                    await self._send_transcript({"event": "transcript", "role": "ai", "text": cached_reply, "isFinal": True})
                    await self.push_frame(LLMFullResponseStartFrame())
                    
                    chunker = AggressiveChunker()
                    # Feed words/tokens to trigger incremental yielding
                    for word in cached_reply.split(' '):
                        for sentence in chunker.feed(word + ' '):
                            await self.push_frame(TextFrame(text=sentence))
                            # Add a tiny sleep to let TTS start processing the first chunk
                            await asyncio.sleep(0.01)
                    
                    # Flush the rest
                    for sentence in chunker.drain():
                        await self.push_frame(TextFrame(text=sentence))
                        
                    await self.push_frame(LLMFullResponseEndFrame())
                    return

            # OPTIMIZATION 1: Fan-out RAG Prefetch and Filler
            from services.langgraph_agent.nodes import rag_retriever

            async def _push_filler_task():
                await self.push_frame(LLMFullResponseStartFrame())
                if pending_filler:
                    await self.push_frame(TextFrame(text=pending_filler))

            # Use the speculative RAG result if it completed while STT was running,
            # otherwise kick off a fresh search now (in parallel with filler).
            if self._speculative_rag_task is not None and self._speculative_rag_task.done():
                try:
                    speculative_docs = self._speculative_rag_task.result()
                except Exception:
                    speculative_docs = None
                rag_task = None
                print("[RAG SPECULATIVE] Reusing prefetched docs from interim transcript.")
            else:
                speculative_docs = None
                rag_task = asyncio.create_task(
                    asyncio.to_thread(rag_retriever.search_faq_docs, text)
                ) if rag_retriever and not is_emergency else None

            # Cancel any still-running speculative task now that we have the final text
            if self._speculative_rag_task and not self._speculative_rag_task.done():
                self._speculative_rag_task.cancel()
            self._speculative_rag_task = None

            filler_task = asyncio.create_task(_push_filler_task())

            # Await them simultaneously — total wait = slowest one!
            rag_docs, _ = await asyncio.gather(
                rag_task if rag_task else asyncio.sleep(0),
                filler_task,
                return_exceptions=True
            )

            # If we used the speculative result, rag_docs is the dummy sleep return
            if speculative_docs is not None:
                rag_docs = speculative_docs

            if rag_docs and not isinstance(rag_docs, Exception):
                context = " ".join([d.metadata.get("window_context", d.page_content) for d in rag_docs[:3]])
                user_msg += f"\n\n[SYSTEM PREFETCHED FAQ CONTEXT (use if relevant, ignore if not)]: {context}"
                print("[RAG PREFETCH] Injected sentence-window FAQ context natively to bypass tool calls.")
            elif isinstance(rag_docs, Exception):
                print(f"[RAG PREFETCH] Error: {rag_docs}")

            import re
            from datetime import datetime
            from core.database import AsyncSessionLocal
            from tools.db_tools import AppointmentRepository

            DATE_PATTERN = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')

            async def _speculative_db_prefetch(text: str) -> dict:
                cache = {}
                dates = DATE_PATTERN.findall(text)
                
                async with AsyncSessionLocal() as db:
                    repo = AppointmentRepository(db)
                    try:
                        provider = await repo.get_default_provider()
                        cache["provider"] = provider
                    except Exception:
                        pass
                        
                    for date_str in dates[:2]:
                        try:
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                            appts = await repo.get_appointments_on_date(date_obj)
                            cache[f"slots_{date_str}"] = appts
                        except Exception:
                            pass
                return cache

            prefetch_task = asyncio.create_task(_speculative_db_prefetch(text))
            
            self._chunker = AggressiveChunker()
            tool_names = []
            ui_tokens = []
            
            self._db_cache = await prefetch_task

            async for event in self.langgraph_app.astream_events(
                {"messages": [("user", user_msg)]},
                {"configurable": {"thread_id": self.call_sid, "db_cache": self._db_cache}},
                version="v2"
            ):
                kind = event["event"]
                
                if kind == "on_tool_end" and event["data"].get("output") == "__RESET_CONTEXT_SIGNAL__":
                    print("[AGENT] LLM invoked clear_memory tool — resetting context.")
                    await self._send_transcript({"event": "transcript", "role": "ai", "text": "*Context Reset*", "isFinal": True})
                    self.call_sid = str(uuid.uuid4())
                    await self.push_frame(LLMFullResponseEndFrame())
                    return
                    
                if kind == "on_tool_start":
                    name = event["name"]
                    if name not in tool_names:
                        tool_names.append(name)
                        
                if kind == "on_chain_end" and event.get("name") in ["refusal", "refusal_node"]:
                    output = event["data"].get("output", {})
                    msgs = output.get("messages", [])
                    if msgs and hasattr(msgs[-1], "content"):
                        refusal_text = msgs[-1].content
                        ui_tokens.append(refusal_text)
                        if self._chunker:
                            for sentence in self._chunker.feed(refusal_text):
                                safe_sentence = sentence.encode('ascii', 'replace').decode('ascii')
                                print(f"[CHUNKER] -> TTS: {safe_sentence!r}")
                                await self.push_frame(TextFrame(text=sentence))

                if kind == "on_chat_model_stream" and event.get("name") == "agent_model":
                    chunk = event["data"]["chunk"]
                    raw_chunk_content = getattr(chunk, "content", None)
                    if raw_chunk_content and not getattr(chunk, "tool_calls", None):
                        if isinstance(raw_chunk_content, str):
                            token = raw_chunk_content
                        elif isinstance(raw_chunk_content, list):
                            text_parts = []
                            for part in raw_chunk_content:
                                if isinstance(part, dict) and "text" in part:
                                    text_parts.append(str(part["text"]))
                                elif isinstance(part, str):
                                    text_parts.append(part)
                            token = "".join(text_parts)
                        else:
                            token = str(raw_chunk_content)

                        if not token:
                            continue

                        ui_tokens.append(token)
                        
                        # Stream the text chunk directly to the UI immediately, bypassing TTS queue
                        if self.websocket:
                            try:
                                await self.websocket.send_json({"event": "transcript", "role": "ai", "text": token, "isFinal": False})
                            except Exception:
                                pass
                        else:
                            await self._send_transcript({"event": "transcript", "role": "ai", "text": token, "isFinal": False})

                        if self._chunker:
                            for sentence in self._chunker.feed(token):
                                safe_sentence = sentence.encode('ascii', 'replace').decode('ascii')
                                print(f"[CHUNKER] -> TTS: {safe_sentence!r}")
                                await self.push_frame(TextFrame(text=sentence))

            if self._chunker:
                for sentence in self._chunker.drain():
                    safe_sentence = sentence.encode('ascii', 'replace').decode('ascii')
                    print(f"[CHUNKER] final flush: {safe_sentence!r}")
                    await self.push_frame(TextFrame(text=sentence))
                self._chunker = None

            # UI update
            accumulated = _CLEAN_UI.sub('', ''.join(ui_tokens))
            if tool_names:
                accumulated = f"{accumulated} [tools: {', '.join(tool_names)}]"
                
            safe_accumulated = accumulated.encode('ascii', 'replace').decode('ascii')
            print(f"[AGENT] Spoken: {safe_accumulated}")
            
            # Cache pure conversational replies (without tool invocations or emergency tags)
            clean_reply = _CLEAN_UI.sub('', ''.join(ui_tokens)).strip()
            if not tool_names and not is_emergency and clean_reply and len(clean_reply) > 4 and query_embedding:
                _SEMANTIC_RESPONSE_CACHE.put(text, query_embedding, clean_reply)

            await self._send_transcript({"event": "transcript", "role": "ai", "text": accumulated, "isFinal": True})
            await self.push_frame(LLMFullResponseEndFrame())

        except Exception as e:
            import traceback
            self._chunker = None
            print(f"[ERROR] processing transcript: {e}")
            traceback.print_exc()
            await self._send_transcript({"event": "transcript", "role": "ai", "text": FALLBACK_TEXT, "isFinal": True})
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TextFrame(text=FALLBACK_TEXT))
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame, direction):
        # CRITICAL: Always call super() first so Pipecat's internal state machine
        # can process system frames (StartFrame sets __started, CancelFrame sets
        # _cancelling, etc.). Without this, push_frame() silently drops everything.
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            from pipecat.frames.frames import (
                InterimTranscriptionFrame, 
                StartFrame, 
                InputAudioRawFrame, 
                AudioRawFrame,
                InterruptionFrame,
                TranscriptionFrame
            )
            
            # DROP microphone audio from flowing into the TTS
            if isinstance(frame, (InputAudioRawFrame, AudioRawFrame)):
                return
            
            if isinstance(frame, StartFrame):
                # IMPORTANT: Push StartFrame downstream FIRST to initialize TTS and Output!
                await self.push_frame(frame, direction)
                
                if self.skip_greeting:
                    print(f"[PIPELINE] skip_greeting=True. Clean pipeline ready for caller input.")
                    return

                greeting = "Hello! I'm Sarah from Dr. Smith's clinic. How can I help you today?"
                async def _send_greeting():
                    # Wait a tiny bit to ensure the downstream pipeline has processed the StartFrame
                    await asyncio.sleep(0.1)
                    if getattr(self, "tavus_url", None):
                        await self.push_frame(
                            OutputTransportMessageFrame(
                                {"event": "avatar_ready", "url": self.tavus_url}
                            )
                        )
                    # Push transcript downstream
                    await self._send_transcript({"event": "transcript", "role": "ai", "text": greeting, "isFinal": True})
                    from pipecat.frames.frames import LLMFullResponseStartFrame
                    # Push as TWO separate sentences — each gets its own TTS call
                    # so Kokoro renders them independently (no concatenation stutter).
                    await self.push_frame(LLMFullResponseStartFrame())
                    await self.push_frame(TextFrame(text="Hello! I'm Sarah from Dr. Smith's clinic."))
                    # Small yield to let Kokoro start processing the first sentence
                    # before we enqueue the second. Prevents the aggregator from
                    # merging them into one chunk with no space.
                    await asyncio.sleep(0)
                    await self.push_frame(TextFrame(text=" How can I help you today?"))
                    await self.push_frame(LLMFullResponseEndFrame())
                
                asyncio.create_task(_send_greeting())
                return # We already pushed StartFrame

            if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
                # Critical: When user starts speaking, instantly cancel any TTS generation!
                await self.push_frame(InterruptionFrame())
                await self.push_frame(OutputTransportMessageFrame({"event": "interruption"}))
                if self._debounce_task and not self._debounce_task.done():
                    self._debounce_task.cancel()
                self._chunker = None
                self.was_interrupted = True
                self._buffer.clear()
                # Also cancel any in-flight speculative RAG from the previous turn
                if self._speculative_rag_task and not self._speculative_rag_task.done():
                    self._speculative_rag_task.cancel()
                self._speculative_rag_task = None
                self._speculative_rag_result = None
                
            if isinstance(frame, InterimTranscriptionFrame):
                # Push interim transcripts downstream through the pipeline
                await self._send_transcript({"event": "transcript", "role": "user", "text": frame.text, "isFinal": False})

                # Speculative RAG prefetch: fire a background search on the FIRST
                # interim chunk. By the time the final transcript arrives ~0.5–1 s
                # later, results are already in self._speculative_rag_result.
                # We only launch one task per turn (skip if one is already running).
                if (
                    frame.text
                    and self._speculative_rag_task is None
                    and len(frame.text.split()) >= 3  # skip 1–2 word fragments
                ):
                    from services.langgraph_agent.nodes import rag_retriever
                    if rag_retriever:
                        async def _speculative_rag(query: str):
                            try:
                                import asyncio as _ai
                                return await _ai.to_thread(
                                    rag_retriever.search_faq_docs, query
                                )
                            except Exception:
                                return None
                        self._speculative_rag_task = asyncio.create_task(
                            _speculative_rag(frame.text)
                        )
                return

            if isinstance(frame, TranscriptionFrame):
                print(f"[STT-CHUNK] Final: {frame.text}")
                
                # Accumulate the text chunks
                self._buffer.append(frame.text)

                # Cancel any existing debounce task
                if self._debounce_task and not self._debounce_task.done():
                    self._debounce_task.cancel()

                # Process the accumulated text after a shorter pause to prevent split sentences
                # 0.05 s debounce (was 0.1 s) — coalesce rapid STT chunks without
                # adding noticeable delay. Parakeet rarely splits a single utterance
                # into more than 1–2 chunks, so 50 ms is plenty.
                async def _debounce():
                    try:
                        print("[DEBOUNCE] Task started, sleeping for 0.05s...")
                        await asyncio.sleep(0.05)
                        print(f"[DEBOUNCE] Woke up! Buffer size: {len(self._buffer)}")
                        full_text = " ".join(self._buffer).strip()
                        self._buffer.clear()
                        if full_text:
                            print(f"[DEBOUNCE] Triggering LangGraph with text: {full_text}")
                            try:
                                await self._process_final_transcript(full_text)
                            except asyncio.CancelledError:
                                # If interrupted by new speech, put the text back so it gets prepended to the new chunk!
                                self._buffer.insert(0, full_text)
                                raise
                        else:
                            print("[DEBOUNCE] Buffer was empty, nothing to process.")
                    except asyncio.CancelledError:
                        print("[DEBOUNCE] Task cancelled because user kept speaking!")
                    except Exception as e:
                        print(f"[DEBOUNCE] FATAL ERROR: {e}")

                self._debounce_task = asyncio.create_task(_debounce())
                # Keep a strong reference in a class-level set if we were really paranoid,
                # but self._debounce_task is a strong reference on the instance.
                return

        # Forward the frame to the next processor in the pipeline
        await self.push_frame(frame, direction)





def _is_elevenlabs_voice_error(error: Exception) -> bool:
    message = str(error).lower()
    return "voice_id" in message and "does not exist" in message


# ── Shared Core Models (Single-Tenant RAM Optimization) ──────────────

from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.kokoro.tts import KokoroTTSService

class SharedWhisperSTTService(WhisperSTTService):
    def _load(self):
        from main import get_whisper_model
        model = get_whisper_model()
        if model is None:
            print("[BOT] Fallback: Loading duplicate Whisper model (main.py was not ready).")
            super()._load()
        else:
            self._model = model

    from typing_extensions import override
    from typing import AsyncGenerator
    from pipecat.frames.frames import Frame

    @override
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        import numpy as np
        from pipecat.frames.frames import TranscriptionFrame, ErrorFrame
        from pipecat.utils.time import time_now_iso8601
        from pipecat.transcriptions.language import Language
        import asyncio

        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        def _transcribe_sync(model, audio_np):
            segments, _ = model.transcribe(
                audio_np,
                beam_size=1,
                language="en",
                vad_filter=False,
                word_timestamps=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()

        text = await asyncio.to_thread(_transcribe_sync, self._model, audio_float)
        
        if text:
            yield TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=Language.EN,
            )

class SharedKokoroTTSService(KokoroTTSService):
    _AUDIO_CACHE: dict[str, list] = {}
    _MAX_AUDIO_CACHE_SIZE: int = 150

    def __init__(self, **kwargs):
        from main import get_kokoro_model
        from pipecat.services.tts_service import TTSService
        from pipecat.audio.utils import create_stream_resampler
        from pipecat.transcriptions.language import Language

        shared_kokoro = get_kokoro_model()
        if shared_kokoro is not None:
            default_settings = self.Settings(
                model=None,
                voice=None,
                language=Language.EN,
            )
            settings_arg = kwargs.pop("settings", None)
            if settings_arg is not None:
                default_settings.apply_update(settings_arg)

            TTSService.__init__(
                self,
                push_start_frame=True,
                push_stop_frames=True,
                settings=default_settings,
                **kwargs,
            )
            self._kokoro = shared_kokoro
            self._resampler = create_stream_resampler()
        else:
            super().__init__(**kwargs)

    from typing_extensions import override
    from typing import AsyncGenerator
    from pipecat.frames.frames import Frame

    @override
    async def run_tts(self, text: str, context_id: str = None, *args, **kwargs) -> AsyncGenerator[Frame, None]:
        clean_key = text.strip().lower()
        if clean_key in self._AUDIO_CACHE:
            print(f"[CACHE HIT - TTS AUDIO] Reusing cached speech for: '{clean_key[:30]}'")
            for frame in self._AUDIO_CACHE[clean_key]:
                yield frame
            return

        buffered_frames = []
        async for frame in super().run_tts(text, context_id, *args, **kwargs):
            buffered_frames.append(frame)
            yield frame

        if buffered_frames and len(clean_key) < 120:
            if len(self._AUDIO_CACHE) >= self._MAX_AUDIO_CACHE_SIZE:
                self._AUDIO_CACHE.pop(next(iter(self._AUDIO_CACHE)))
            self._AUDIO_CACHE[clean_key] = buffered_frames


import httpx

async def create_tavus_avatar_session(call_sid: str):
    """
    Creates a Tavus conversational avatar session.
    Requires TAVUS_API_KEY in the environment.
    """
    persona_id = getattr(settings, "TAVUS_PERSONA_ID", "pd43ffef")
    api_key = getattr(settings, "TAVUS_API_KEY", "")
    
    if not api_key:
        print("[WARNING] TAVUS_API_KEY not set. Avatar will not be created.")
        return None
        
    payload = {
        "persona_id": persona_id,
        "custom_greeting": "Hello! I'm Sarah from Dr. Smith's clinic. How can I help you today?",
        "conversational_context": f"You are an AI assistant for the clinic. Session ID: {call_sid}"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://tavusapi.com/v2/conversations",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key
                }
            )
            response.raise_for_status()
            data = response.json()
            print(f"[TAVUS] Avatar session created: {data.get('conversation_id')}")
            return data
    except Exception as e:
        print(f"[TAVUS ERROR] Failed to create avatar session: {e}")
        return None

# ── Entry point ───────────────────────────────────────────────────────────────

async def run_bot(websocket, call_sid: str, skip_greeting: bool = False, input_sample_rate: int = 24000):
    """
    Runs the real-time audio pipeline over a WebSocket.

    IMPORTANT: STT and TTS services are created fresh here for EVERY connection.
    Do NOT move them to module level — they are stateful and cannot be shared.
    We also use a pre-built LangGraph app (built once at startup) to avoid rebuilding per connection.
    Local VAD is used for perfect endpointing for all STT providers.
    """

    stt_provider = (settings.STT_PROVIDER or "deepgram").strip().lower()
    tts_provider = (settings.TTS_PROVIDER or "deepgram").strip().lower()
    async with AsyncExitStack() as stack:
        elevenlabs_session = None
        if stt_provider == "elevenlabs" or tts_provider == "elevenlabs":
            elevenlabs_session = await stack.enter_async_context(aiohttp.ClientSession())

        if stt_provider == "openai":
            from pipecat.services.openai.stt import OpenAISTTService
            stt_service = OpenAISTTService(api_key=settings.OSS_API_KEY)
        elif stt_provider == "elevenlabs":
            from pipecat.services.elevenlabs.stt import ElevenLabsSTTService, ElevenLabsSTTSettings
            elevenlabs_stt_settings = ElevenLabsSTTSettings(model="scribe_v2")
            stt_service = ElevenLabsSTTService(
                api_key=settings.ELEVENLABS_API_KEY,
                sample_rate=8000,
                aiohttp_session=elevenlabs_session,
                settings=elevenlabs_stt_settings,            )
        elif stt_provider == "whisper":
            stt_service = SharedWhisperSTTService(
                settings=WhisperSTTService.Settings(model="small.en"),
                device="cuda", # Blazing fast GPU inference
                compute_type="float16"
            )
        elif stt_provider == "sherpa_onnx":
            from services.pipecat_pipeline.sherpa_onnx_stt import SherpaOnnxSTTService
            # Pass provider/threads from config so PARAKEET_PROVIDER and
            # PARAKEET_NUM_THREADS env vars are honoured without touching bot.py.
            stt_service = SherpaOnnxSTTService(
                model_dir=settings.SHERPA_ONNX_MODEL_PATH,
                provider=settings.PARAKEET_PROVIDER.strip() or None,
                num_threads=settings.PARAKEET_NUM_THREADS or None,
            )

        else:
            from pipecat.services.deepgram.stt import DeepgramSTTService
            stt_service = DeepgramSTTService(
                api_key=settings.DEEPGRAM_API_KEY,
                settings=DeepgramSTTService.Settings(endpointing=300),
            )

        if tts_provider == "openai":
            from pipecat.services.openai.tts import OpenAITTSService

            tts_service = OpenAITTSService(
                api_key=settings.OSS_API_KEY,
                voice=settings.TTS_VOICE,
            )
        elif tts_provider == "elevenlabs":
            from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsTTSSettings

            elevenlabs_tts_settings = ElevenLabsTTSSettings(model=settings.ELEVENLABS_MODEL)
            if settings.ELEVENLABS_VOICE_ID.strip():
                elevenlabs_tts_settings.voice = settings.ELEVENLABS_VOICE_ID.strip()

            tts_service = ElevenLabsTTSService(
                api_key=settings.ELEVENLABS_API_KEY,
                settings=elevenlabs_tts_settings,
            )
        elif tts_provider == "kokoro":
            from pipecat.services.tts_service import TextAggregationMode
            tts_service = SharedKokoroTTSService(
                text_aggregation_mode=TextAggregationMode.TOKEN,
                settings=KokoroTTSService.Settings(voice="af_heart")
            )
        elif tts_provider == "piper":
            from pipecat.services.piper.tts import PiperTTSService
            import pathlib
            tts_service = PiperTTSService(
                download_dir=str(pathlib.Path(__file__).parent.parent.parent / "data" / "models" / "piper"),
                settings=PiperTTSService.Settings(
                    voice=settings.TTS_VOICE or "en_US-lessac-medium"
                )
            )
        else:
            from pipecat.services.deepgram.tts import DeepgramTTSService

            tts_service = DeepgramTTSService(
                api_key=settings.DEEPGRAM_API_KEY,
                settings=DeepgramTTSService.Settings(voice=settings.TTS_VOICE),
            )

        print(f"[BOT] New connection: call_sid={call_sid}")

        tavus_session = await create_tavus_avatar_session(call_sid)
        tavus_url = tavus_session.get("conversation_url") if tavus_session else None

        # ── Use the pre-built LangGraph app (built once at startup in main.py lifespan) ──
        # Avoid rebuilding the graph per connection — it's expensive and redundant.
        from main import get_langgraph_app

        langgraph_app = get_langgraph_app()
        if langgraph_app is None:
            # Fallback: build on-demand if called before lifespan (e.g. in tests)
            from services.langgraph_agent.graph import build_graph

            langgraph_app = await build_graph()

        from pipecat.serializers.twilio import TwilioFrameSerializer, FrameSerializer

        class DebugTwilioSerializer(FrameSerializer):
            def __init__(self, stream_sid, default_sample_rate: int = 24000):
                self.stream_sid = stream_sid
                self.default_sample_rate = default_sample_rate
                self.inner = TwilioFrameSerializer(
                    stream_sid=stream_sid,
                    params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
                )
                from pipecat.audio.utils import create_stream_resampler
                self._resampler = create_stream_resampler()
            
            async def setup(self, frame):
                await self.inner.setup(frame)
                
            async def serialize(self, frame):
                from pipecat.frames.frames import AudioRawFrame
                if isinstance(frame, OutputTransportMessageFrame):
                    return json.dumps(frame.message)
                if isinstance(frame, AudioRawFrame):
                    payload = base64.b64encode(frame.audio).decode("utf-8")
                    return json.dumps({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": payload}
                    })
                return await self.inner.serialize(frame)
                
            async def deserialize(self, data):
                try:
                    if isinstance(data, (str, bytes)):
                        obj = json.loads(data)
                        if obj.get("event") == "interruption":
                            from pipecat.frames.frames import UserStartedSpeakingFrame
                            return UserStartedSpeakingFrame()
                        if obj.get("event") == "chat" and "text" in obj:
                            from pipecat.frames.frames import TranscriptionFrame
                            return TranscriptionFrame(text=obj["text"], user_id="user", timestamp="")
                        if obj.get("event") == "media" and "payload" in obj.get("media", {}):
                            from pipecat.frames.frames import InputAudioRawFrame
                            audio_bytes = base64.b64decode(obj["media"]["payload"])
                            media_obj = obj.get("media", {})
                            sr = int(media_obj.get("sampleRate") or media_obj.get("sample_rate") or self.default_sample_rate)
                            if sr == 24000:
                                audio_bytes = await self._resampler.resample(audio_bytes, 24000, 16000)
                            return InputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)
                except Exception:
                    pass
                res = await self.inner.deserialize(data)
                return res

        # ── Transport ────────────────────────────────────────────────────────────
        # VAD is an *analyzer*, not a pipeline processor.
        # It must live inside FastAPIWebsocketParams, not Pipeline([]).
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                audio_out_sample_rate=24000,
                audio_in_sample_rate=16000,
                serializer=DebugTwilioSerializer(stream_sid=call_sid, default_sample_rate=input_sample_rate),
            ),
        )

        # stop_secs=0.5: was 0.8 — cuts 300 ms dead time between user stopping
        # and the STT endpoint firing. Silero VAD is reliable enough at 0.5 s;
        # going below 0.4 s risks false endpoints on brief mid-sentence pauses.
        vad_processor = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5, confidence=0.3, min_volume=0.01))
        )

        adapter = LangGraphAdapter(langgraph_app, call_sid, transport, tavus_url=tavus_url, skip_greeting=skip_greeting, websocket=websocket)

        processors = [
            transport.input(),
            vad_processor,       # 1. Detect when user truly stops speaking
            stt_service,         # 2. Transcribe audio → text
            adapter,             # 3. Route text through LangGraph, get response
            tts_service,         # 4. Synthesise response → audio
            transport.output(),  # 5. Stream audio back to caller
        ]

        pipeline = Pipeline(processors)

        try:
            worker = PipelineWorker(pipeline, idle_timeout_secs=30)
            runner = WorkerRunner()
            await runner.run(worker)
        finally:
            try:
                # Explicitly wipe the conversation memory from the checkpointer
                if hasattr(langgraph_app, 'checkpointer') and langgraph_app.checkpointer:
                    await langgraph_app.checkpointer.adelete_thread(call_sid)
                    print(f"[MEMORY] Cleared conversation memory for call_sid={call_sid}")
            except Exception as e:
                print(f"[MEMORY] Failed to clear conversation memory: {e}")