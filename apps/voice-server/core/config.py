from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    
    DATABASE_URL: str = 'sqlite+aiosqlite:///./test.db'
    
    OSS_API_KEY: str = ''
    GROQ_API_KEY: str = ''
    DEEPGRAM_API_KEY: str = ''
    ELEVENLABS_API_KEY: str = ''
    
    TWILIO_ACCOUNT_SID: str = ''
    TWILIO_AUTH_TOKEN: str = ''
    TWILIO_PHONE_NUMBER: str = '+1234567890'
    BASE_URL: str = 'localhost:8000'
    
    WAITLIST_PHONE_NUMBER: str = '+1098765432'
    EMERGENCY_PHONE: str = '+1123456789'
    FRONT_DESK_PHONE: str = '+1987654321'
    
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = 'openai/gpt-oss-20b'
    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
    DOCS_DIR: str = 'data/documents'
    
    STT_PROVIDER: str = 'whisper'
    TTS_PROVIDER: str = 'kokoro'
    TTS_VOICE: str = 'af_heart'
    ELEVENLABS_VOICE_ID: str = ''
    ELEVENLABS_MODEL: str = 'eleven_multilingual_v2'

settings = Settings()
