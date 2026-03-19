"""
extractors/image_extractor.py – Single-image OCR extraction wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def extract_text_from_image(file_path: str | Path, ocr_engine) -> str:
    """
    Open an image file and run the hybrid OCR engine on it.

    Returns:
        Full OCR text string with [HANDWRITTEN] / [PRINTED] section markers.
    """
    p = Path(file_path)
    try:
        img = Image.open(p)
        img.load()   # force decode so errors surface here, not later
    except Exception as e:
        raise RuntimeError(f"Cannot open image '{p}': {e}") from e

    logger.info(f"Running OCR on image: {p.name} ({img.size}, {img.mode})")
    result = ocr_engine.extract(img)
    return result["full_text"]
