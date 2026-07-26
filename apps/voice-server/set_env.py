import os
from pathlib import Path

base_dir = Path(__file__).parent
hf_cache = base_dir / "local_model" / "huggingface_cache"
hf_cache.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(hf_cache))
os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
