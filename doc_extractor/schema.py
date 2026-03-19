"""
schema.py – Defines the dynamic JSON schema and validates pipeline output.
"""

from __future__ import annotations
import json
import jsonschema

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TARGET SCHEMA  (Dynamic)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DocumentExtraction",
    "type": "object",
    "additionalProperties": True,
    "properties": {
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
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SCHEMA UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def empty_record(source_format: str = "unknown") -> dict:
    """Return an empty output record."""
    return {
        "_source_format": source_format,
        "_extraction_confidence": None,
    }

def validate(record: dict) -> tuple[bool, list[str]]:
    """
    Validate a record against TARGET_SCHEMA.
    """
    validator = jsonschema.Draft7Validator(TARGET_SCHEMA)
    errors = [e.message for e in validator.iter_errors(record)]
    return (len(errors) == 0), errors

def coerce_types(record: dict) -> dict:
    """
    Filter out empty/null fields so the final JSON only contains present data.
    """
    cleaned_record = {}
    for k, v in record.items():
        if v not in (None, "", [], {}):
            cleaned_record[k] = v
    return cleaned_record

def pretty(record: dict) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False)
