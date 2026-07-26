import os
import urllib.request
import logging
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logging.info(f"File already exists: {dest_path}. Skipping download.")
        return
    logging.info(f"Downloading {url} to {dest_path}...")
    try:
        # Create a request object with a User-Agent header
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        logging.info(f"Successfully downloaded {dest_path}")
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_dir = os.path.join(base_dir, "local_model")
    os.makedirs(local_model_dir, exist_ok=True)
    
    logging.info(f"Target directory for local models: {local_model_dir}")
    hf_cache_dir = os.path.join(local_model_dir, "huggingface_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = hf_cache_dir
    os.environ["HF_HOME"] = hf_cache_dir

    # 1. Download Kokoro TTS ONNX Model and Voices from HuggingFace
    logging.info("Downloading Kokoro TTS ONNX from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        
        # Download the model
        model_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", filename="model.onnx", cache_dir=hf_cache_dir)
        shutil.copy(model_path, os.path.join(local_model_dir, "kokoro-v1.0.onnx"))
        
        # Download the voices file
        voices_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", filename="voices.json", cache_dir=hf_cache_dir)
        shutil.copy(voices_path, os.path.join(local_model_dir, "voices.json"))
        
        logging.info("Successfully downloaded Kokoro models.")
    except Exception as e:
        logging.error(f"Failed to download Kokoro: {e}")

    # 2. Download Faster-Whisper Model
    # Faster Whisper natively caches to HuggingFace hub cache. 
    # By setting HF_HUB_CACHE, we force it to download into our local_model folder cleanly!
    logging.info("Downloading Faster-Whisper (small.en) model...")
    try:
        # This will download the model weights explicitly to the local_model/huggingface_cache folder
        WhisperModel(
            model_size_or_path="small.en",
            device="cpu", # Just using cpu to initialize the download
            compute_type="int8"
        )
        logging.info(f"Successfully downloaded Faster-Whisper into {hf_cache_dir}")
    except Exception as e:
        logging.error(f"Failed to download Faster-Whisper: {e}")

    logging.info("All local models have been downloaded and configured!")

if __name__ == "__main__":
    main()
