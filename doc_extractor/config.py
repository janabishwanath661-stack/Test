"""
config.py – Central configuration for the extraction pipeline.

All model choices are justified for the "mini resources" constraint:
  • TrOCR-base-handwritten : ~340 MB, CPU-only, specialized for cursive/print mix
  • Qwen2.5-1.5B-Instruct  : ~1.5 GB (fp16) → ~800 MB at int8; runs on 4 GB RAM
  • EasyOCR                : No transformer weights, classical + CRAFT detector
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
import logging

logger = logging.getLogger(__name__)

# --- GPU Check Warning ---
if not torch.cuda.is_available():
    logger.warning(
        "\n" + "="*80 + "\n"
        "WARNING: PyTorch is NOT detecting any NVIDIA GPUs (torch.cuda.is_available() is False).\n"
        "If you are running this on a device that HAS an NVIDIA GPU, your Python environment\n"
        "likely installed the CPU-only version of PyTorch.\n"
        "To enable GPU acceleration, please reinstall PyTorch with CUDA support:\n"
        "  pip uninstall torch torchvision torchaudio accelerate bitsandbytes\n"
        "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
        "  pip install accelerate bitsandbytes\n"
        "="*80 + "\n"
    )

@dataclass
class OCRConfig:
    # ── Handwriting specialist (HuggingFace) ──────────────────────────────────
    trocr_model_id: str = "microsoft/trocr-base-handwritten"
    # Use "auto" to let transformers decide (will use all GPUs if available).
    # To force a specific GPU on a multi-GPU machine, change this to "cuda:0", "cuda:1", etc.
    trocr_device: str = "auto"

    # ── Printed-text OCR ──────────────────────────────────────────────────────
    easyocr_languages: list = field(default_factory=lambda: ["en"])
    # EasyOCR needs an explicit bool. We check cuda but allow override.
    easyocr_gpu: bool = field(default_factory=lambda: torch.cuda.is_available())

    # ── Pre-processing ────────────────────────────────────────────────────────
    image_max_dim: int = 2048           # Resize long edge before OCR (memory cap)
    pdf_dpi: int = 200                  # Higher = better OCR, more RAM; 200 is safe
    contrast_enhance: float = 1.4       # Contrast multiplier before OCR
    sharpen: bool = True

    # ── Handwriting detection heuristic ──────────────────────────────────────
    # Regions taller than this pixel-height are sent to TrOCR (line-level crops)
    handwriting_min_box_h: int = 20


@dataclass
class LLMConfig:
    # ── Small extraction LLM ─────────────────────────────────────────────────
    # Qwen2.5-1.5B is the sweet spot: instruction-tuned, JSON-capable, <2 GB RAM
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Quantization: "int8" ≈ halves RAM, "int4" ≈ quarters RAM (needs bitsandbytes)
    load_in_8bit: bool = True
    load_in_4bit: bool = False          # Overrides 8-bit if True

    # "auto" will use all GPUs if available, or CPU if not.
    # To force a specific GPU on a multi-GPU machine, change this to "cuda:0", "cuda:1", etc.
    device_map: str = "auto"
    max_new_tokens: int = 1024
    temperature: float = 0.0            # Greedy = deterministic JSON
    repetition_penalty: float = 1.1

    # How many times to retry if JSON parse fails
    max_retries: int = 3


@dataclass
class PipelineConfig:
    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # ── Local Weight Management ───────────────────────────────────────────────
    # All downloaded models (TrOCR, Qwen, EasyOCR) are stored here.
    # This ensures no internet is needed once downloaded.
    weights_dir: str = "models/pretrained_weights"

    # ── Lazy loading: models loaded on first use, not at import ───────────────
    lazy_load: bool = True

    # ── Supported file extensions ─────────────────────────────────────────────
    image_exts: tuple = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")
    pdf_exts: tuple = (".pdf",)
    excel_exts: tuple = (".xlsx", ".xls", ".xlsm", ".csv")

    # ── Output ────────────────────────────────────────────────────────────────
    fallback_value: Optional[str] = None   # Value for fields not found in doc


# Singleton – import this everywhere
PIPELINE_CFG = PipelineConfig()
