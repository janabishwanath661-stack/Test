"""
schema.py – Defines the target JSON schema and validates pipeline output.

HOW TO CUSTOMISE:
  1. Edit TARGET_SCHEMA to match your exact output contract.
  2. Add/remove fields – the LLM prompt is built dynamically from this schema,
     so the extraction prompt always stays in sync with your schema.
  3. Mark required fields in REQUIRED_FIELDS so the validator can flag gaps.
"""

from __future__ import annotations
import json
from typing import Any
import jsonschema

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TARGET SCHEMA  (edit this to match your use-case)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DocumentExtraction",
    "type": "object",
    "properties": {

        # ── Personal / Identity ──────────────────────────────────────────────
        "full_designation_name": {
            "type": ["string", "null"],
            "description": "Full name including any title/designation (e.g. Dr., Mr., Eng.)"
        },
        "date_of_birth": {
            "type": ["string", "null"],
            "description": "ISO-8601 date string YYYY-MM-DD"
        },
        "national_id": {
            "type": ["string", "null"],
            "description": "National ID / passport number / employee ID"
        },

        # ── Contact ──────────────────────────────────────────────────────────
        "email_address": {
            "type": ["string", "null"],
            "format": "email"
        },
        "phone_number": {
            "type": ["string", "null"],
            "description": "Include country code if visible"
        },

        # ── Organisation ─────────────────────────────────────────────────────
        "organisation_name": {
            "type": ["string", "null"]
        },
        "job_title": {
            "type": ["string", "null"]
        },
        "department": {
            "type": ["string", "null"]
        },

        # ── Financial / Reference Numbers ────────────────────────────────────
        "invoice_number": {
            "type": ["string", "null"]
        },
        "total_amount": {
            "type": ["number", "null"],
            "description": "Numeric value only, no currency symbols"
        },
        "currency": {
            "type": ["string", "null"],
            "description": "3-letter ISO currency code (USD, EUR, INR…)"
        },
        "issue_date": {
            "type": ["string", "null"],
            "description": "ISO-8601 date string YYYY-MM-DD"
        },

        # ── Handwritten Annotations ──────────────────────────────────────────
        "handwritten_notes": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "All handwritten text segments found; preserve as-is"
        },

        # ── Pipeline Metadata ─────────────────────────────────────────────────
        "_extraction_confidence": {
            "type": ["string", "null"],
            "enum": ["high", "medium", "low", None],
            "description": "Self-assessed confidence of the extraction"
        },
        "_source_format": {
            "type": ["string", "null"],
            "description": "Original file format: pdf | image | excel"
        }
    },
    "required": []   # Strict required fields – add keys if needed
}

# Fields the LLM is instructed to prioritise (order matters for prompting)
FIELD_PRIORITY_ORDER = [
    "full_designation_name",
    "date_of_birth",
    "national_id",
    "email_address",
    "phone_number",
    "organisation_name",
    "job_title",
    "department",
    "invoice_number",
    "total_amount",
    "currency",
    "issue_date",
    "handwritten_notes",
    "_extraction_confidence",
    "_source_format",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SCHEMA UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_field_descriptions() -> str:
    """Return a compact field-by-field description for LLM prompting."""
    lines = []
    props = TARGET_SCHEMA["properties"]
    for key in FIELD_PRIORITY_ORDER:
        if key.startswith("_"):
            continue                    # skip metadata fields from prompt
        info = props.get(key, {})
        desc = info.get("description", "")
        lines.append(f'  "{key}": {desc}' if desc else f'  "{key}"')
    return "\n".join(lines)


def empty_record(source_format: str = "unknown") -> dict:
    """Return an empty output record."""
    return {
        "_source_format": source_format,
        "_extraction_confidence": None,
    }


def validate(record: dict) -> tuple[bool, list[str]]:
    """
    Validate a record against TARGET_SCHEMA.
    Returns (is_valid, list_of_error_messages).
    """
    validator = jsonschema.Draft7Validator(TARGET_SCHEMA)
    errors = [e.message for e in validator.iter_errors(record)]
    return (len(errors) == 0), errors


def coerce_types(record: dict) -> dict:
    """
    Best-effort type coercion after LLM extraction.
    e.g. total_amount "1,234.56" → 1234.56
    """
    # total_amount → float
    if isinstance(record.get("total_amount"), str):
        cleaned = record["total_amount"].replace(",", "").strip()
        try:
            record["total_amount"] = float(cleaned)
        except ValueError:
            record["total_amount"] = None

    # handwritten_notes: ensure it's a list
    hw = record.get("handwritten_notes")
    if isinstance(hw, str):
        record["handwritten_notes"] = [hw] if hw.strip() else None

    # currency: uppercase
    if isinstance(record.get("currency"), str):
        record["currency"] = record["currency"].strip().upper()

    # Filter out empty/null fields so the final JSON only contains present data
    cleaned_record = {}
    for k, v in record.items():
        if v not in (None, "", [], {}):
            cleaned_record[k] = v

    return cleaned_record


def pretty(record: dict) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False)
