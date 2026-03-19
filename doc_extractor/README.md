# Document Extraction Pipeline
### Open-Source · HuggingFace Models · CPU / Mini-Resource Friendly

A production-ready pipeline that parses **images, multi-page PDFs, and Excel files**,
handles **handwritten text**, filters noise, and outputs a **strict JSON schema** —
all on hardware as small as a 4 GB RAM machine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTRACTION PIPELINE                                 │
│                                                                             │
│  User File                                                                  │
│     │                                                                       │
│     ▼                                                                       │
│  ┌──────────────┐                                                           │
│  │ Format Router│  Magic-byte + extension detection                        │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│    ┌────┴──────────────────┬────────────────────┐                          │
│    │                       │                    │                          │
│    ▼                       ▼                    ▼                          │
│  IMAGE / PNG/JPEG       PDF (pages)          EXCEL / CSV                   │
│    │                       │                    │                          │
│    │                  PyMuPDF rasterize          │                          │
│    │                  (200 DPI, no Poppler)      │                          │
│    │                       │                    │                          │
│    └──────────┬────────────┘           pandas + openpyxl                   │
│               │                        (native tabular)                    │
│               ▼                                 │                          │
│    ┌──────────────────────┐                     │                          │
│    │  HYBRID OCR ENGINE   │                     │                          │
│    │                      │                     │                          │
│    │  EasyOCR (CRAFT)     │  ←─ detect all      │                          │
│    │  ──────────────────  │     text regions     │                          │
│    │  Per bounding box:   │                     │                          │
│    │  ┌──── heuristic ──┐ │                     │                          │
│    │  │ confidence < 0.72│ │                     │                          │
│    │  │ OR high px-var  │ │                     │                          │
│    │  └────────┬────────┘ │                     │                          │
│    │       hw? │ print?   │                     │                          │
│    │     ┌─────┘    └───┐ │                     │                          │
│    │     ▼              ▼ │                     │                          │
│    │  TrOCR-base    EasyOCR│                    │                          │
│    │  handwritten   result │                    │                          │
│    │  (HuggingFace)        │                    │                          │
│    │  [HANDWRITTEN]  [PRINTED] markers          │                          │
│    └──────────────────────┘                     │                          │
│               │                                 │                          │
│               └─────────────┬───────────────────┘                          │
│                             │                                               │
│                             ▼                                               │
│              ┌──────────────────────────────┐                              │
│              │  Qwen2.5-1.5B-Instruct (LLM) │                              │
│              │  Quantized: int8 (~1.5 GB)    │                              │
│              │                               │                              │
│              │  System prompt:               │                              │
│              │  • Field descriptions (schema)│                              │
│              │  • [HANDWRITTEN] priority rule│                              │
│              │  • "Return JSON only"         │                              │
│              │                               │                              │
│              │  ┌─── Retry loop (x3) ──────┐ │                              │
│              │  │ JSON parse error?         │ │                              │
│              │  │ → inject error, re-prompt │ │                              │
│              │  └───────────────────────────┘ │                              │
│              └──────────────────────────────┘                              │
│                             │                                               │
│                             ▼                                               │
│                  JSON Schema Validation                                     │
│                  Type Coercion (dates, amounts)                             │
│                             │                                               │
│                             ▼                                               │
│                    ExtractionResult.json  ✓                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Choices & Memory Budget

| Role | Model | Size | RAM needed |
|------|-------|------|-----------|
| Printed OCR | EasyOCR (CRAFT + CRNN) | ~50 MB | ~400 MB |
| Handwritten OCR | `microsoft/trocr-base-handwritten` | ~340 MB | ~600 MB |
| JSON Extraction | `Qwen/Qwen2.5-1.5B-Instruct` int8 | ~1.5 GB | ~2.0 GB |
| **Total** | | **~1.9 GB** | **~3 GB** |

Fits comfortably in **4 GB RAM** with room to spare.

---

## Installation

```bash
# 1. Clone / create the project directory
cd doc_extractor

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# (Optional) Install bitsandbytes for quantization on Linux/Windows
# Already in requirements.txt — skip if on macOS ARM
```

---

## Usage

### Python API

```python
from pipeline import ExtractionPipeline

pipeline = ExtractionPipeline()

# Single file
result = pipeline.run("documents/invoice.pdf")
print(result.pretty_json())
print(f"Valid: {result.is_valid}, Confidence: {result.json['_extraction_confidence']}")

# Batch processing
results = pipeline.run_batch([
    "scans/form_001.png",
    "sheets/employees.xlsx",
    "reports/multi_page.pdf",
])

for r in results:
    print(r)   # ExtractionResult(✓ format=pdf confidence=high elapsed=8.3s)
```

### CLI

```bash
# Single file
python pipeline.py invoice.pdf

# Multiple files, save to JSONL
python pipeline.py *.pdf *.png --out results.jsonl

# Debug mode (includes raw OCR text in output)
python pipeline.py form.png --debug
```

---

## Customising the JSON Schema

Edit `schema.py` → `TARGET_SCHEMA`:

```python
"properties": {
    "your_custom_field": {
        "type": ["string", "null"],
        "description": "Explain what this field contains — the LLM uses this"
    },
    ...
}
```

The system prompt is built dynamically from `TARGET_SCHEMA`, so the LLM always
stays in sync with your schema — no need to edit prompts manually.

---

## Tuning for Your Hardware

Edit `config.py`:

```python
# Extremely low RAM (≤ 2 GB): use 4-bit quantization
cfg.llm.load_in_4bit = True
cfg.llm.load_in_8bit = False

# No GPU: force CPU
cfg.ocr.trocr_device = "cpu"
cfg.ocr.easyocr_gpu  = False
cfg.llm.device_map   = "cpu"

# Lower DPI to save RAM on PDFs
cfg.ocr.pdf_dpi = 150

# If documents are mostly printed (no handwriting): skip TrOCR entirely
# by setting a very high confidence threshold
cfg.ocr.handwriting_min_box_h = 9999
```

---

## File Structure

```
doc_extractor/
├── requirements.txt          # All dependencies
├── config.py                 # Central configuration (models, hardware tuning)
├── schema.py                 # Target JSON schema + validation utilities
├── pipeline.py               # Main orchestrator + CLI entry point
├── extractors/
│   ├── router.py             # File format detection
│   ├── excel_extractor.py    # Excel/CSV → structured text (pandas)
│   ├── pdf_extractor.py      # PDF → page images (PyMuPDF)
│   └── image_extractor.py    # Image → OCR text
└── models/
    ├── ocr_engine.py         # Hybrid EasyOCR + TrOCR engine
    └── llm_engine.py         # Qwen2.5 JSON extraction + retry logic
```

---

## Solving Each Core Challenge

| Challenge | Solution |
|-----------|----------|
| **Multi-format inputs** | Format router dispatches to specialised extractor per type |
| **Noisy / boilerplate data** | LLM prompt explicitly instructs to ignore headers, footers, watermarks |
| **Handwritten text** | TrOCR specialist model; confidence + pixel-variance heuristic detects HW regions; marked `[HANDWRITTEN]` for LLM priority |
| **Strict JSON output** | System prompt + self-correction retry loop + `jsonschema` validation |
| **Mini hardware** | No vision LLM (text-only Qwen1.5B); quantized int8; lazy model loading; sequential batch (no parallel loading) |
| **Excel tabular data** | pandas reads natively — no OCR vision loss; column headers preserved as field labels |
