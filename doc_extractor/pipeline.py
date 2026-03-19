"""
pipeline.py – Main entry point for the document extraction pipeline.

Quick-start
───────────
    from pipeline import ExtractionPipeline

    pipeline = ExtractionPipeline()
    result = pipeline.run("invoice.pdf")
    print(result.json)

Architecture diagram
────────────────────

  File Path
      │
      ▼
  ┌───────────┐
  │  Router   │ ──── detects format (image / pdf / excel)
  └─────┬─────┘
        │
   ┌────┴─────────────────┐
   │                      │                      │
   ▼                      ▼                      ▼
 IMAGE / PNG          PDF (multi-page)        EXCEL / CSV
   │                      │                      │
   ▼                      ▼                      │
 OCREngine           pdf_pages_to_images         │
 (EasyOCR +          → per-page OCR              │
  TrOCR)                  │                      │
   │                      │                      │
   └──────────────────────┤                      │
                          │                      │
                    Raw OCR Text          pandas → text table
                          │                      │
                          └──────────┬───────────┘
                                     │
                                     ▼
                               LLM Engine
                          (Qwen2.5-1.5B-Instruct)
                          schema-targeted prompt
                                     │
                                     ▼
                            JSON Extraction + Retry
                                     │
                                     ▼
                          Schema Validation + Coercion
                                     │
                                     ▼
                            ExtractionResult
                         { .json, .is_valid, .errors }
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import PIPELINE_CFG, PipelineConfig
from extractors.router import detect_format
from extractors.excel_extractor import extract_text_from_excel
from extractors.image_extractor import extract_text_from_image
from extractors.pdf_extractor import extract_text_from_pdf
from models.ocr_engine import OCREngine
from models.llm_engine import LLMEngine
from schema import validate, pretty

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    json: dict                            # The extracted JSON record
    source_format: str = "unknown"        # "image" | "pdf" | "excel"
    is_valid: bool = False
    validation_errors: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    raw_ocr_text: Optional[str] = None   # Available for debugging

    def pretty_json(self) -> str:
        return pretty(self.json)

    def __repr__(self) -> str:
        ok = "✓" if self.is_valid else "✗"
        return (
            f"ExtractionResult({ok} format={self.source_format} "
            f"confidence={self.json.get('_extraction_confidence')} "
            f"elapsed={self.elapsed_seconds:.1f}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionPipeline:
    """
    Orchestrates the full extraction flow for any supported file type.

    Usage:
        pipeline = ExtractionPipeline()         # uses default config
        result   = pipeline.run("path/to/file.pdf")
        print(result.pretty_json())
    """

    def __init__(self, cfg: PipelineConfig = PIPELINE_CFG):
        self.cfg = cfg
        # Lazy-loaded models
        self._ocr_engine: Optional[OCREngine] = None
        self._llm_engine: Optional[LLMEngine] = None

    @property
    def ocr_engine(self) -> OCREngine:
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine(self.cfg)
        return self._ocr_engine

    @property
    def llm_engine(self) -> LLMEngine:
        if self._llm_engine is None:
            self._llm_engine = LLMEngine(self.cfg)
        return self._llm_engine

    # ── Internal: text extraction per format ─────────────────────────────────

    def _extract_text(self, file_path: Path, fmt: str) -> str:
        if fmt == "image":
            return extract_text_from_image(file_path, self.ocr_engine)
        elif fmt == "pdf":
            return extract_text_from_pdf(file_path, self.ocr_engine, self.cfg)
        elif fmt == "excel":
            return extract_text_from_excel(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: '{fmt}'. "
                f"Supported: image ({self.cfg.image_exts}), "
                f"pdf ({self.cfg.pdf_exts}), excel ({self.cfg.excel_exts})"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, file_path: str | Path, *, debug: bool = False) -> ExtractionResult:
        """
        Run the full extraction pipeline on a file.

        Args:
            file_path : path to the document to process
            debug     : if True, raw OCR text is included in the result

        Returns:
            ExtractionResult with .json, .is_valid, and more
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        t0 = time.perf_counter()
        fmt = detect_format(p)
        logger.info(f"Processing '{p.name}' detected as: {fmt.upper()}")

        # ── Step 1: Extract raw text ─────────────────────────────────────────
        raw_text = self._extract_text(p, fmt)
        logger.info(f"OCR/parse complete. Text length: {len(raw_text)} chars.")

        if not raw_text.strip():
            logger.warning("No text could be extracted from the file.")
            from schema import empty_record
            record = empty_record(fmt)
            record["_extraction_confidence"] = "low"
            return ExtractionResult(
                json=record,
                source_format=fmt,
                is_valid=False,
                validation_errors=["No text extracted from document"],
                elapsed_seconds=time.perf_counter() - t0,
            )

        # ── Step 2: LLM JSON extraction ──────────────────────────────────────
        logger.info("Running LLM extraction…")
        record = self.llm_engine.extract(raw_text, source_format=fmt)

        # ── Step 3: Validate ─────────────────────────────────────────────────
        is_valid, errors = validate(record)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Done in {elapsed:.1f}s. Valid={is_valid}. "
            f"Confidence={record.get('_extraction_confidence')}"
        )

        return ExtractionResult(
            json=record,
            source_format=fmt,
            is_valid=is_valid,
            validation_errors=errors,
            elapsed_seconds=elapsed,
            raw_ocr_text=raw_text if debug else None,
        )

    def run_batch(
        self, file_paths: list[str | Path], *, debug: bool = False
    ) -> list[ExtractionResult]:
        """Process multiple files sequentially (memory-safe for mini resources)."""
        results = []
        for i, fp in enumerate(file_paths, 1):
            logger.info(f"Batch [{i}/{len(file_paths)}] – {fp}")
            try:
                results.append(self.run(fp, debug=debug))
            except Exception as e:
                logger.error(f"Failed to process '{fp}': {e}")
                from schema import empty_record
                results.append(
                    ExtractionResult(
                        json=empty_record("unknown"),
                        source_format="unknown",
                        is_valid=False,
                        validation_errors=[str(e)],
                    )
                )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json as _json
    import sys

    parser = argparse.ArgumentParser(
        description="Document extraction pipeline — outputs JSON to stdout"
    )
    parser.add_argument("files", nargs="+", help="File paths to process")
    parser.add_argument("--debug", action="store_true", help="Include raw OCR in output")
    parser.add_argument(
        "--out", default=None, help="Write JSON output to this file (one record per line)"
    )
    args = parser.parse_args()

    pipeline = ExtractionPipeline()
    records = []

    for fp in args.files:
        try:
            result = pipeline.run(fp, debug=args.debug)
            print(result.pretty_json())
            records.append(result.json)
        except Exception as exc:
            print(f"ERROR processing {fp}: {exc}", file=sys.stderr)

    if args.out and records:
        with open(args.out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n✓ Results written to {args.out}", file=sys.stderr)
