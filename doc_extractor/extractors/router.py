"""
extractors/router.py – Inspects the uploaded file and returns
the appropriate extractor function and a human-readable format label.

Detection order:
  1. By file extension (fast)
  2. By magic bytes / MIME sniffing (fallback for extensionless files)
"""

from __future__ import annotations

import mimetypes
from pathlib import Path


def detect_format(file_path: str | Path) -> str:
    """
    Return one of: "image" | "pdf" | "excel" | "unknown"
    """
    p = Path(file_path)
    ext = p.suffix.lower()

    image_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    pdf_exts = {".pdf"}
    excel_exts = {".xlsx", ".xls", ".xlsm", ".csv"}

    if ext in image_exts:
        return "image"
    if ext in pdf_exts:
        return "pdf"
    if ext in excel_exts:
        return "excel"

    # ── Magic-byte fallback ───────────────────────────────────────────────────
    try:
        with open(p, "rb") as f:
            header = f.read(8)

        if header[:4] == b"%PDF":
            return "pdf"
        if header[:8] in (
            b"\x89PNG\r\n\x1a\n",            # PNG
            b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1",  # JPEG
        ):
            return "image"
        if header[:4] in (b"PK\x03\x04",):   # ZIP-based (xlsx)
            return "excel"
    except OSError:
        pass

    # ── MIME type guessing ────────────────────────────────────────────────────
    mime, _ = mimetypes.guess_type(str(p))
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime == "application/pdf":
            return "pdf"
        if "spreadsheet" in mime or "excel" in mime or mime == "text/csv":
            return "excel"

    return "unknown"
