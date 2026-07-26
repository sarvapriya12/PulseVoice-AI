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


# ── LangGraph Adapter ────────────────────────────────────────────────────────

class LangGraphAdapter(FrameProcessor):
    """Bridges Pipecat and LangGraph — receives transcribed text, returns AI response."""

    def __init__(self, langgraph_app, call_sid: str, transport: FastAPIWebsocketTransport):
        super().__init__()
        self.langgraph_app = langgraph_app
        self.call_sid = call_sid
        self.transport = transport
        self._buffer = []
        self._debounce_task = None
        self.was_interrupted = False
        self._chunker = None

    async def _process_final_transcript(self, text: str):
        try:
            print(f"[STT] Patient: {text}")
            await self.push_frame(
                OutputTransportMessageFrame(
                    {"event": "transcript", "role": "user", "text": text, "isFinal": True}
                )
            )

            text_lower = text.lower()
            is_emergency = any(w in text_lower for w in EMERGENCY_WORDS)
            pending_filler = None if is_emergency else random.choice(FILLERS)

            await self.push_frame(OutputTransportMessageFrame({"event": "status", "text": "Thinking..."}))

            user_msg = text
            if self.was_interrupted:
                user_msg = f"[SYSTEM NOTE: You were cut off mid-sentence. Address their new input directly.]\n{text}"
                self.was_interrupted = False

            # OPTIMIZATION 1: Fan-out RAG Prefetch and Filler
            from services.langgraph_agent.nodes import rag_retriever
            import asyncio
            
            async def _push_filler_task():
                await self.push_frame(LLMFullResponseStartFrame())
                if pending_filler:
                    await self.push_frame(TextFrame(text=pending_filler))

            rag_task = asyncio.create_task(
                asyncio.to_thread(rag_retriever.search_faq_docs, text)
            ) if rag_retriever and not is_emergency else None

            # If you add a guardrail later, you can add: guardrail_task = asyncio.create_task(_run_guardrail(text))
            filler_task = asyncio.create_task(_push_filler_task())

            # Await them simultaneously — total wait = slowest one!
            rag_docs, _ = await asyncio.gather(
                rag_task if rag_task else asyncio.sleep(0),
                filler_task,
                return_exceptions=True
            )

            if rag_docs and not isinstance(rag_docs, Exception):
                context = " ".join([d.page_content for d in rag_docs[:3]])
                user_msg += f"\n\n[SYSTEM PREFETCHED FAQ CONTEXT (use if relevant, ignore if not)]: {context}"
                print("[RAG PREFETCH] Injected FAQ context natively to bypass tool calls.")
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
                    await self.push_frame(
                        OutputTransportMessageFrame(
                            {"event": "transcript", "role": "ai", "text": "*Context Reset*", "isFinal": True}
                        )
                    )
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
                                print(f"[CHUNKER] → TTS: {sentence!r}")
                                await self.push_frame(TextFrame(text=sentence))

                if kind == "on_chat_model_stream" and event.get("name") == "agent_model":
                    chunk = event["data"]["chunk"]
                    if getattr(chunk, "content", None) and not chunk.tool_calls:
                        token = chunk.content
                        ui_tokens.append(token)
                        
                        # Stream the text chunk directly to the UI immediately
                        await self.push_frame(
                            OutputTransportMessageFrame(
                                {"event": "transcript", "role": "ai", "text": token, "isFinal": False}
                            )
                        )

                        if self._chunker:
                            for sentence in self._chunker.feed(token):
                                print(f"[CHUNKER] → TTS: {sentence!r}")
                                await self.push_frame(TextFrame(text=sentence))

            if self._chunker:
                for sentence in self._chunker.drain():
                    print(f"[CHUNKER] final flush: {sentence!r}")
                    await self.push_frame(TextFrame(text=sentence))
                self._chunker = None

            # UI update
            accumulated = _CLEAN_UI.sub('', ''.join(ui_tokens))
            if tool_names:
                accumulated = f"{accumulated} [tools: {', '.join(tool_names)}]"
                
            print(f"[AGENT] Spoken: {accumulated}")
            
            await self.push_frame(
                OutputTransportMessageFrame(
                    {"event": "transcript", "role": "ai", "text": accumulated, "isFinal": True}
                )
            )
            await self.push_frame(LLMFullResponseEndFrame())

        except Exception as e:
            self._chunker = None
            print(f"[ERROR] processing transcript: {e}")
            await self.push_frame(
                OutputTransportMessageFrame(
                    {"event": "transcript", "role": "ai", "text": FALLBACK_TEXT, "isFinal": True}
                )
            )
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
                
                greeting = "Hello! I'm Sarah from Dr. Smith's clinic. How can I help you today?"
                async def _send_greeting():
                    # Wait a tiny bit to ensure the downstream pipeline has processed the StartFrame
                    await asyncio.sleep(0.1)
                    # Push transcript downstream (it will pass through TTS unmodified and reach transport)
                    await self.push_frame(
                        OutputTransportMessageFrame(
                            {"event": "transcript", "role": "ai", "text": greeting, "isFinal": True}
                        )
                    )
                    from pipecat.frames.frames import LLMFullResponseStartFrame
                    # Push text to TTS
                    await self.push_frame(LLMFullResponseStartFrame())
                    await self.push_frame(TextFrame(text="Hello! I'm Sarah from Dr. Smith's clinic."))
                    await self.push_frame(TextFrame(text="How can I help you today?"))
                    await self.push_frame(LLMFullResponseEndFrame())
                
                asyncio.create_task(_send_greeting())
                return # We already pushed StartFrame

            if isinstance(frame, UserStartedSpeakingFrame):
                # Critical: When user starts speaking, instantly cancel any TTS generation!
                await self.push_frame(InterruptionFrame())
                if self._debounce_task and not self._debounce_task.done():
                    self._debounce_task.cancel()
                self._chunker = None
                self.was_interrupted = True
                
            if isinstance(frame, InterimTranscriptionFrame):
                # Push interim transcripts downstream through the pipeline
                await self.push_frame(
                    OutputTransportMessageFrame(
                        {"event": "transcript", "role": "user", "text": frame.text, "isFinal": False}
                    )
                )
                return

            if isinstance(frame, TranscriptionFrame):
                print(f"[STT-CHUNK] Final: {frame.text}")
                
                # Accumulate the text chunks
                self._buffer.append(frame.text)

                # Cancel any existing debounce task
                if self._debounce_task and not self._debounce_task.done():
                    self._debounce_task.cancel()

                # Process the accumulated text after a 1.5s pause to prevent split sentences
                async def _debounce():
                    try:
                        print("[DEBOUNCE] Task started, sleeping for 0.3s...")
                        await asyncio.sleep(0.3)
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
    def _load(self):
        from main import get_kokoro_model
        model = get_kokoro_model()
        if model is None:
            print("[BOT] Fallback: Loading duplicate Kokoro model (main.py was not ready).")
            super()._load()
        else:
            self._model = model


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_bot(websocket, call_sid: str):
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
            tts_service = SharedKokoroTTSService(
                aggregate_sentences=False,
                settings=KokoroTTSService.Settings(voice="af_heart")
            )
        else:
            from pipecat.services.deepgram.tts import DeepgramTTSService

            tts_service = DeepgramTTSService(
                api_key=settings.DEEPGRAM_API_KEY,
                settings=DeepgramTTSService.Settings(voice=settings.TTS_VOICE),
            )

        print(f"[BOT] New connection: call_sid={call_sid}")

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
            def __init__(self, stream_sid):
                self.stream_sid = stream_sid
                self.inner = TwilioFrameSerializer(
                    stream_sid=stream_sid,
                    params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
                )
            
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
                        if obj.get("event") == "chat" and "text" in obj:
                            from pipecat.frames.frames import TranscriptionFrame
                            return TranscriptionFrame(text=obj["text"], user_id="user", timestamp="")
                        if obj.get("event") == "media" and "payload" in obj.get("media", {}):
                            from pipecat.frames.frames import InputAudioRawFrame
                            audio_bytes = base64.b64decode(obj["media"]["payload"])
                            return InputAudioRawFrame(audio=audio_bytes, sample_rate=24000, num_channels=1)
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
                audio_in_sample_rate=24000,
                serializer=DebugTwilioSerializer(stream_sid=call_sid),
            ),
        )

        vad_processor = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8, confidence=0.3, min_volume=0.01))
        )

        adapter = LangGraphAdapter(langgraph_app, call_sid, transport)

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