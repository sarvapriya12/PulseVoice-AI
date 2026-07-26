import pytest
from core.config import Settings

def test_settings_defaults(monkeypatch):
    # Make sure we don't accidentally load environment variables from actual .env during this test
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OSS_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.delenv("TTS_PROVIDER", raising=False)
    
    settings = Settings(_env_file=None)
    
    assert settings.DATABASE_URL == "sqlite+aiosqlite:///./test.db"
    # By default, API keys are empty strings
    assert settings.OSS_API_KEY == ""
    assert settings.GROQ_API_KEY == ""
    assert settings.DEEPGRAM_API_KEY == ""
    assert settings.ELEVENLABS_API_KEY == ""
    assert settings.TWILIO_PHONE_NUMBER == "+1234567890"
    assert settings.STT_PROVIDER == "whisper"
    assert settings.TTS_PROVIDER == "kokoro"

def test_settings_env_override(monkeypatch):
    # Override environment variables
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/db")
    monkeypatch.setenv("OSS_API_KEY", "super-secret-key")
    monkeypatch.setenv("LLM_MODEL", "custom-gpt-model")
    monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
    
    settings = Settings()
    
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@localhost/db"
    assert settings.OSS_API_KEY == "super-secret-key"
    assert settings.LLM_MODEL == "custom-gpt-model"
    assert settings.TTS_PROVIDER == "elevenlabs"

def test_empty_api_keys_do_not_crash():
    try:
        # Pydantic Settings should allow empty strings as defined by defaults
        settings = Settings(OSS_API_KEY="", TWILIO_ACCOUNT_SID="")
        assert settings.OSS_API_KEY == ""
        assert settings.TWILIO_ACCOUNT_SID == ""
    except Exception as e:
        pytest.fail(f"Settings instantiation failed with empty API keys: {e}")
