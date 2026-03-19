"""
models/ocr_engine.py – Dual-mode OCR engine.

Architecture
────────────
┌─────────────────────────────────────────────────────────────┐
│  Input Image (PIL)                                          │
│        │                                                    │
│  ┌─────▼──────┐   bounding boxes                           │
│  │  EasyOCR   │ ──────────────────────────────────┐        │
│  │  (CRAFT    │   (printed text detector)          │        │
│  │  backbone) │                                    │        │
│  └────────────┘                                    │        │
│                                                    ▼        │
│                                         For each text box:  │
│                                         classify printed    │
│                                         vs handwritten      │
│                                              │       │      │
│                                        print │   hw  │      │
│                                              │       ▼      │
│                                              │  ┌─────────┐ │
│                                              │  │  TrOCR  │ │
│                                              │  │  base-  │ │
│                                              │  │ handwrt │ │
│                                              │  └────┬────┘ │
│                                              │       │      │
│                                              └──┬────┘      │
│                                                 ▼           │
│                                         Merged text blocks  │
│                              (handwritten text PRIORITISED) │
└─────────────────────────────────────────────────────────────┘

Models used (both HuggingFace):
  • microsoft/trocr-base-handwritten  — 340 MB, encoder-decoder, CPU-friendly
  • EasyOCR (jaided-ai)               — CRAFT detector + CRNN recogniser, ~50 MB
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── Lazy imports so the module doesn't crash if a dep is missing ─────────────
_trocr_processor = None
_trocr_model = None
_easyocr_reader = None


def _load_trocr(device: str = "cpu"):
    global _trocr_processor, _trocr_model
    if _trocr_model is None:
        logger.info("Loading TrOCR handwriting model…")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _trocr_processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        ).to(device)
        _trocr_model.eval()
        logger.info("TrOCR ready.")
    return _trocr_processor, _trocr_model


def _load_easyocr(languages: list[str], gpu: bool):
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Loading EasyOCR reader…")
        import easyocr
        _easyocr_reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
        logger.info("EasyOCR ready.")
    return _easyocr_reader


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img: Image.Image, max_dim: int, contrast: float, sharpen: bool) -> Image.Image:
    """Resize, enhance contrast, optionally sharpen for better OCR accuracy."""
    # 1. Convert to RGB (drop alpha channel, handle greyscale)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 2. Cap resolution to avoid OOM
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # 3. Contrast enhancement
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # 4. Optional sharpening
    if sharpen:
        img = img.filter(ImageFilter.SHARPEN)

    return img


def _crop_box(img: Image.Image, box) -> Image.Image:
    """Crop a detected bounding-box region from the image."""
    # EasyOCR returns [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    # Add 2-px padding
    x1, y1 = max(0, x1 - 2), max(0, y1 - 2)
    x2, y2 = min(img.width, x2 + 2), min(img.height, y2 + 2)
    return img.crop((x1, y1, x2, y2))


# ─────────────────────────────────────────────────────────────────────────────
# Handwriting heuristic
# ─────────────────────────────────────────────────────────────────────────────

def _is_likely_handwritten(crop: Image.Image, easyocr_confidence: float) -> bool:
    """
    Lightweight heuristic to decide if a text-box crop is handwritten:
      1. EasyOCR confidence < 0.72 suggests messy/irregular strokes
      2. High pixel-level variance relative to bounding-box area
         (printed text is uniform; handwriting has irregular ink distribution)
    """
    if easyocr_confidence < 0.72:
        return True

    arr = np.array(crop.convert("L"), dtype=np.float32)
    # Coefficient of variation of pixel intensities
    std_dev = float(np.std(arr))
    mean_val = float(np.mean(arr))
    cv = std_dev / (mean_val + 1e-6)
    return cv > 0.45   # empirically tuned threshold


# ─────────────────────────────────────────────────────────────────────────────
# TrOCR inference on a single crop
# ─────────────────────────────────────────────────────────────────────────────

def _trocr_read(crop: Image.Image, processor, model, device: str) -> str:
    """Run TrOCR on a single PIL image crop."""
    import torch
    pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class OCREngine:
    """
    Hybrid OCR engine: EasyOCR for printed text, TrOCR for handwritten regions.
    Handwritten text is flagged and returned separately for priority handling.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # Models are loaded lazily on first call to .extract()

    def extract(self, img: Image.Image) -> dict:
        """
        Run full OCR on a PIL image.

        Returns:
            {
              "printed_text"     : str,   # concatenated printed OCR
              "handwritten_text" : str,   # concatenated handwritten OCR
              "full_text"        : str,   # merged; handwritten prepended
              "raw_blocks"       : list   # per-box detail for debugging
            }
        """
        cfg = self.cfg.ocr

        # ── 1. Pre-process ──────────────────────────────────────────────────
        img = preprocess_image(img, cfg.image_max_dim, cfg.contrast_enhance, cfg.sharpen)

        # ── 2. EasyOCR pass – gets bounding boxes for ALL text ──────────────
        reader = _load_easyocr(cfg.easyocr_languages, cfg.easyocr_gpu)
        raw_results = reader.readtext(np.array(img))
        # raw_results: list of (bbox, text, confidence)

        printed_parts: list[str] = []
        handwritten_parts: list[str] = []
        raw_blocks: list[dict] = []

        processor, model = None, None  # loaded only if handwriting detected

        for (bbox, easy_text, confidence) in raw_results:
            crop = _crop_box(img, bbox)
            is_hw = _is_likely_handwritten(crop, confidence)

            if is_hw and crop.height >= cfg.handwriting_min_box_h:
                # ── 3. Re-read with TrOCR for handwritten crops ─────────────
                if processor is None:
                    processor, model = _load_trocr(cfg.trocr_device)
                hw_text = _trocr_read(crop, processor, model, cfg.trocr_device)
                final_text = hw_text if hw_text else easy_text
                handwritten_parts.append(final_text)
                raw_blocks.append({
                    "text": final_text, "source": "trocr",
                    "confidence": confidence, "bbox": [list(p) for p in bbox]
                })
            else:
                printed_parts.append(easy_text)
                raw_blocks.append({
                    "text": easy_text, "source": "easyocr",
                    "confidence": confidence, "bbox": [list(p) for p in bbox]
                })

        printed_text = " ".join(printed_parts)
        handwritten_text = " ".join(handwritten_parts)

        # Handwritten text prepended so the LLM sees it first (priority signal)
        separator = "\n[HANDWRITTEN]\n" if handwritten_text else ""
        full_text = (
            f"{separator}{handwritten_text}\n[PRINTED]\n{printed_text}"
            if handwritten_text
            else printed_text
        )

        return {
            "printed_text": printed_text,
            "handwritten_text": handwritten_text,
            "full_text": full_text.strip(),
            "raw_blocks": raw_blocks,
        }
