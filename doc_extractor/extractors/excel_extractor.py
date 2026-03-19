"""
extractors/excel_extractor.py – Converts Excel / CSV files to structured text.

Why no OCR here?
  Excel files are already machine-readable; passing them through a vision model
  would LOSE tabular structure. Instead we use pandas to read the data natively
  and convert it to a clean, LLM-friendly text representation.

Output format (passed to LLM):
  SHEET: Sheet1
  ┌─────────────┬──────────────┬──────────┐
  │ Full Name   │ Date of Birth│ ID       │
  ├─────────────┼──────────────┼──────────┤
  │ Dr. Ana Lima│ 1985-03-22   │ 987654   │
  └─────────────┴──────────────┴──────────┘
  ...

This preserves column headers (often the field labels) and row values,
giving the LLM the context it needs to map columns to schema fields.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _df_to_text(df: pd.DataFrame, sheet_name: str) -> str:
    """Convert a DataFrame to a compact text table for LLM consumption."""
    # Drop completely empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    if df.empty:
        return f"SHEET: {sheet_name}\n[empty]\n"

    # Stringify all values, preserve NaN as empty string
    df = df.fillna("").astype(str)

    lines = [f"SHEET: {sheet_name}"]

    # Build a simple markdown-style table
    col_widths = [
        max(len(str(col)), df[col].str.len().max())
        for col in df.columns
    ]

    # Header row
    header = " | ".join(str(col).ljust(w) for col, w in zip(df.columns, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    lines.append(header)
    lines.append(separator)

    for _, row in df.iterrows():
        line = " | ".join(str(v).ljust(w) for v, w in zip(row.values, col_widths))
        lines.append(line)

    return "\n".join(lines)


def extract_text_from_excel(file_path: str | Path) -> str:
    """
    Read an Excel (.xlsx/.xls/.xlsm) or CSV file and return
    a structured text representation of all sheets.
    """
    p = Path(file_path)
    all_text_parts: list[str] = []

    if p.suffix.lower() == ".csv":
        # ── CSV path ──────────────────────────────────────────────────────────
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            all_text_parts.append(_df_to_text(df, sheet_name="CSV"))
        except Exception as e:
            logger.error(f"Failed to read CSV {p}: {e}")
            return f"[ERROR reading CSV: {e}]"
    else:
        # ── Excel path ────────────────────────────────────────────────────────
        try:
            xl = pd.ExcelFile(p, engine="openpyxl")
        except Exception as e:
            logger.error(f"Failed to open Excel {p}: {e}")
            return f"[ERROR reading Excel: {e}]"

        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(
                    sheet_name,
                    dtype=str,
                    keep_default_na=False,
                    header=0,
                )
                text = _df_to_text(df, sheet_name)
                all_text_parts.append(text)
            except Exception as e:
                logger.warning(f"Skipping sheet '{sheet_name}': {e}")
                all_text_parts.append(f"SHEET: {sheet_name}\n[ERROR: {e}]")

    return "\n\n".join(all_text_parts)
