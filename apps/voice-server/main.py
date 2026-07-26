import set_env  # noqa: F401
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse

_langgraph_app = None
_startup_status = {
    "db": False,
    "langgraph": False,
    "whisper": False,
    "kokoro": False,
    "reranker": False,
}
_shared_whisper_model = None
_shared_kokoro_model = None
_shared_reranker = None

def _maybe_add_cuda_dll():
    import os
    import platform
    if platform.system() == "Windows":
        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
        if os.path.exists(cuda_path):
            os.add_dll_directory(cuda_path)
            os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _langgraph_app, _shared_whisper_model, _shared_kokoro_model, _shared_reranker, _startup_status

    # 1. Database — fatal if this fails
    try:
        from core.database import engine
        from core.models import Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _startup_status["db"] = True
        print("[DB] Tables ready.")
    except Exception as e:
        print(f"[DB] FATAL: {e}")
        raise

    # 2. LangGraph — fatal if this fails
    try:
        from services.langgraph_agent.graph import build_graph
        _langgraph_app = await build_graph()
        _startup_status["langgraph"] = True
        print("[LangGraph] Ready.")
    except Exception as e:
        print(f"[LangGraph] FATAL: {e}")
        raise

    # 3. Whisper — non-fatal
    try:
        _maybe_add_cuda_dll()
        from faster_whisper import WhisperModel
        _shared_whisper_model = WhisperModel("small.en", device="cuda", compute_type="float16")
        _startup_status["whisper"] = True
        print("[Whisper] Loaded on CUDA.")
    except Exception as e:
        print(f"[Whisper] Degraded — will fall back per-connection: {e}")

    # 4. Kokoro — non-fatal
    try:
        from kokoro_onnx import Kokoro
        from pathlib import Path
        base_dir = Path(__file__).parent
        model_file = base_dir / "local_model" / "kokoro-v1.0.onnx"
        voices_file = base_dir / "local_model" / "voices-v1.0.bin"
        if not model_file.exists() or not voices_file.exists():
            raise FileNotFoundError(f"Kokoro model files missing in {base_dir / 'local_model'}")
        _shared_kokoro_model = Kokoro(str(model_file), str(voices_file))
        _startup_status["kokoro"] = True
        print("[Kokoro] Loaded.")
    except Exception as e:
        print(f"[Kokoro] Degraded: {e}")

    # 5. Reranker — non-fatal
    try:
        from sentence_transformers import CrossEncoder
        _shared_reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        _startup_status["reranker"] = True
        print("[Reranker] Loaded.")
    except Exception as e:
        print(f"[Reranker] Degraded: {e}")

    # 6. Background cron
    import asyncio
    from services.waitlist_notifier import waitlist_cron_job
    cron_task = asyncio.create_task(waitlist_cron_job())

    print(f"[STARTUP] Status: {_startup_status}")
    yield

    # Teardown
    cron_task.cancel()
    try:
        await cron_task
    except asyncio.CancelledError:
        pass

    if hasattr(_langgraph_app, "_pool") and _langgraph_app._pool:
        await _langgraph_app._pool.close()
        print("[DB] Pool closed.")

app = FastAPI(
    title='Clinic AI Voice Agent',
    description='Single-Tenant AI Voice Server powered by LangGraph and Pipecat',
    version='1.0.0',
    lifespan=lifespan
)

from api.routes import router as twilio_router
from api.admin_routes import router as admin_router
app.include_router(twilio_router)
app.include_router(admin_router)

@app.get('/health')
def health_check():
    critical_ready = _startup_status["db"] and _startup_status["langgraph"]
    return {
        "status": "healthy" if critical_ready else "degraded",
        "components": _startup_status,
    }

@app.get('/')
def index():
    return FileResponse('test.html')

@app.get('/test.html')
def test_html():
    return FileResponse('test.html')

def get_langgraph_app():
    """Retrieve the compiled LangGraph workflow instance."""
    return _langgraph_app

def get_whisper_model():
    """Retrieve the globally shared Whisper STT model instance."""
    return _shared_whisper_model

def get_kokoro_model():
    """Retrieve the globally shared Kokoro TTS model instance."""
    return _shared_kokoro_model

def get_reranker_model():
    """Retrieve the globally shared Cross-Encoder Reranker model instance."""
    return _shared_reranker

if __name__ == "__main__":
    import uvicorn
    import os
    is_dev = os.getenv("ENV", "production") == "development"
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=is_dev,
        workers=1
    )
