import os
import sys
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

base_dir = Path(__file__).parent
hf_cache = base_dir / "local_model" / "huggingface_cache"
hf_cache.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(hf_cache))
os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
