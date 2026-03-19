"""
models/__init__.py – Central helper for model downloading/caching.
Ensures models are only downloaded once into a specific directory.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def ensure_weights_dir(weights_dir: str):
    """Create the model directory if it doesn't exist."""
    path = Path(weights_dir)
    if not path.exists():
        logger.info(f"Creating model directory: {path.absolute()}")
        path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())

def get_model_path(model_id: str, weights_dir: str) -> str:
    """Return the expected path for a model within weights_dir."""
    # Replace slashes in model ID to make a valid folder name
    safe_id = model_id.replace("/", "--")
    return os.path.join(weights_dir, safe_id)

def check_model_exists(model_id: str, weights_dir: str) -> bool:
    """Simple check if the model folder exists and isn't empty."""
    target = get_model_path(model_id, weights_dir)
    if os.path.isdir(target) and any(os.listdir(target)):
        return True
    return False

def download_huggingface_model(model_id: str, weights_dir: str, is_tokenizer: bool = False):
    """
    Downloads a HuggingFace model/tokenizer explicitly to the weights_dir.
    Checks if it exists first.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, VisionEncoderDecoderModel, TrOCRProcessor
    
    weights_dir = ensure_weights_dir(weights_dir)
    target_path = get_model_path(model_id, weights_dir)
    
    if check_model_exists(model_id, weights_dir):
        logger.info(f"Model {model_id} already exists in {target_path}. Skipping download.")
        return target_path
        
    logger.info(f"Downloading {model_id} to {target_path}...")
    
    # We use snapshot_download for a full, clean download to the specific folder
    from huggingface_hub import snapshot_download
    
    snapshot_download(
        repo_id=model_id,
        local_dir=target_path,
        local_dir_use_symlinks=False,
        revision="main"
    )
    
    logger.info(f"Download complete: {model_id}")
    return target_path
