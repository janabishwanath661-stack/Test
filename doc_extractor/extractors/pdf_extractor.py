"""
extractors/pdf_extractor.py – Converts each PDF page to a PIL image.

Library: PyMuPDF (fitz)
  • Pure Python bindings to MuPDF — no system Poppler needed
  • Handles encrypted/damaged PDFs gracefully
  • Very fast rasterization even on CPU

Multi-page strategy:
  • Each page is rasterized individually and passed through the OCR engine
  • Text is concatenated page-by-page with a PAGE BREAK marker so the LLM
    can distinguish page boundaries (useful for multi-page invoices)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from PIL import Image

logger = logging.getLogger(__name__)


def pdf_pages_to_images(file_path: str | Path, dpi: int = 200) -> Generator[Image.Image, None, None]:
    """
    Yield one PIL Image per page of the PDF.

    Args:
        file_path: Path to the PDF file.
        dpi: Rendering resolution. 150 = fast, 200 = good quality, 300 = high quality.
             Higher DPI improves OCR on small text but uses more RAM.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. Install it with:\n"
            "  pip install pymupdf"
        )

    p = Path(file_path)
    logger.info(f"Opening PDF: {p.name}")

    try:
        doc = fitz.open(str(p))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{p}': {e}") from e

    zoom = dpi / 72.0   # MuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            logger.debug(f"  Page {page_num + 1}/{len(doc)}: {img.size}")
            yield img
        except Exception as e:
            logger.warning(f"  Skipping page {page_num + 1}: {e}")

    doc.close()


def extract_text_from_pdf(file_path: str | Path, ocr_engine, cfg) -> str:
    """
    Full pipeline: PDF → per-page OCR → concatenated text.

    Args:
        file_path : path to the PDF
        ocr_engine: an initialised OCREngine instance
        cfg       : PipelineConfig

    Returns:
        Multi-page OCR text with PAGE BREAK markers.
    """
    pages_text: list[str] = []

    for page_num, page_img in enumerate(
        pdf_pages_to_images(file_path, dpi=cfg.ocr.pdf_dpi), start=1
    ):
        logger.info(f"  OCR on page {page_num}…")
        result = ocr_engine.extract(page_img)
        page_text = result["full_text"]
        if page_text.strip():
            pages_text.append(f"=== PAGE {page_num} ===\n{page_text}")

    return "\n\n".join(pages_text) if pages_text else "[No text found in PDF]"
