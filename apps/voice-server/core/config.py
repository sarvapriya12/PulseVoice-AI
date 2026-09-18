from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = 'sqlite+aiosqlite:///./test.db'

    # ── API Keys ──────────────────────────────────────────────────────────────
    OSS_API_KEY: str = ''
    GROQ_API_KEY: str = ''
    DEEPGRAM_API_KEY: str = ''
    ELEVENLABS_API_KEY: str = ''
    GEMINI_API_KEY: str = ''

    # ── Twilio ────────────────────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID: str = ''
    TWILIO_AUTH_TOKEN: str = ''
    TWILIO_PHONE_NUMBER: str = '+1234567890'
    BASE_URL: str = 'localhost:8000'

    # ── Phone routing ─────────────────────────────────────────────────────────
    WAITLIST_PHONE_NUMBER: str = '+1098765432'
    EMERGENCY_PHONE: str = '+1123456789'
    FRONT_DESK_PHONE: str = '+1987654321'

    # ── LLM ───────────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = 'openai/gpt-oss-20b'
    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
    DOCS_DIR: str = 'data/documents'

    # ── STT ───────────────────────────────────────────────────────────────────
    # options: whisper, deepgram, elevenlabs, openai, sherpa_onnx
    STT_PROVIDER: str = 'whisper'

    # Path to the Nemotron TDT model directory containing:
    #   encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt
    SHERPA_ONNX_MODEL_PATH: str = 'data/models/nemotron_new'

    # ONNX execution provider for Nemotron inference.
    # Set to 'cuda' to use GPU (requires onnxruntime-gpu), 'cpu' to force CPU.
    # Leave as '' to auto-detect: CUDA is used if CUDAExecutionProvider is available.
    PARAKEET_PROVIDER: str = ''

    # Number of intra-op threads for the ONNX thread pool.
    # 4 is optimal for most int8 workloads on modern CPUs.
    # Increase to 6–8 on high-core servers; decrease to 2 on low-power devices.
    PARAKEET_NUM_THREADS: int = 4

    # Endpoint detection thresholds (seconds) — tune for your call latency target.
    # rule1: trailing silence after the last non-blank token (primary voice stop)
    # rule2: trailing silence safety net (catches slow speakers)
    # rule3: maximum utterance length before forced flush
    PARAKEET_RULE1_SILENCE: float = 1.2
    PARAKEET_RULE2_SILENCE: float = 2.0
    PARAKEET_RULE3_LENGTH:  float = 25.0

    # ── TTS ───────────────────────────────────────────────────────────────────
    TTS_PROVIDER: str = 'kokoro'
    TTS_VOICE: str = 'af_heart'
    ELEVENLABS_VOICE_ID: str = ''
    ELEVENLABS_MODEL: str = 'eleven_multilingual_v2'

    # ── Optional integrations ─────────────────────────────────────────────────
    TAVUS_API_KEY: str = ''
    TAVUS_PERSONA_ID: str = 'pd43ffef'


settings = Settings()
