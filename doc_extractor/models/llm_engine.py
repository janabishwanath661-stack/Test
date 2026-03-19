"""
models/llm_engine.py – Lightweight LLM for structured JSON extraction.

Model choice:  Qwen/Qwen2.5-1.5B-Instruct
  • 1.5 B parameters → ~3 GB fp16, ~1.5 GB int8, ~800 MB int4
  • Excellent instruction-following and JSON generation
  • Runs on 4 GB RAM (int8) or 2 GB RAM (int4) with bitsandbytes
  • Fallback: if quantization unavailable, loads in fp32 with CPU offloading

Extraction strategy
───────────────────
We do NOT ask the LLM to look at images – it only reads the OCR text.
This keeps VRAM near zero (text-only model vs. vision model).
The prompt engineering handles all the "noisy data" problem:

  1. Explicit field descriptions are injected into the system prompt.
  2. Handwritten segments are marked [HANDWRITTEN] and told to take precedence.
  3. The LLM is told to return ONLY a JSON object and nothing else.
  4. A retry loop re-prompts with the parse error if JSON decoding fails.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None


def _load_model(cfg):
    """
    cfg is the full PipelineConfig.
    Loads LLM and Tokenizer from the local weights_dir.
    """
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    from models import download_huggingface_model
    llm_cfg = cfg.llm
    
    # ── Ensure model is downloaded locally ───────────────────────────────────
    local_path = download_huggingface_model(llm_cfg.model_id, cfg.weights_dir)

    logger.info(f"Loading LLM: {llm_cfg.model_id} from {local_path} …")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    _tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

    # ── Quantization config ──────────────────────────────────────────────────
    bnb_cfg = None
    try:
        if llm_cfg.load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif llm_cfg.load_in_8bit:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    except Exception as e:
        logger.warning(f"bitsandbytes (quantization) not available or failed: {e}. Loading unquantized.")

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": llm_cfg.device_map,
    }
    if bnb_cfg:
        load_kwargs["quantization_config"] = bnb_cfg
    else:
        # If not quantized, use half precision if on GPU to save RAM, else fp32
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

    _model = AutoModelForCausalLM.from_pretrained(local_path, **load_kwargs)
    _model.eval()
    logger.info(f"LLM ready (device: {_model.device}, dtype: {_model.dtype})")
    return _tokenizer, _model


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a high-precision document extraction engine. 
Your goal is to extract ALL key information from the provided OCR text and format it as a single JSON object.

RULES:
1. Return ONLY a valid JSON object. No markdown, no commentary.
2. Dynamically create descriptive keys (in snake_case) for any piece of information found in the document (e.g., "invoice_number", "customer_name", "total_amount", "date_of_birth", etc.).
3. Extract all relevant data from the document into this JSON object.
4. Text marked [HANDWRITTEN] takes PRIORITY over [PRINTED] text.
5. Dates must be in YYYY-MM-DD format if possible. Amounts should be numeric if applicable.
"""

_USER_TEMPLATE = """\
OCR TEXT FROM DOCUMENT:
───────────────────────
{ocr_text}
───────────────────────
Extract all available information from the text above and return the JSON object.
"""

_RETRY_TEMPLATE = """\
Your previous response could not be parsed as JSON.
Parse error: {error}

Your previous response was:
{bad_response}

Please try again. Return ONLY a valid JSON object, nothing else.
"""


def _build_system() -> str:
    return _SYSTEM_PROMPT


def _extract_json_from_response(text: str) -> dict:
    """
    Try multiple strategies to extract a JSON object from LLM output:
    1. Direct parse
    2. Strip markdown code fences
    3. Regex search for first {...} block
    """
    text = text.strip()

    # Strategy 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip code fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in model response:\n{text[:500]}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

class LLMEngine:
    """Wraps Qwen2.5-1.5B-Instruct for dynamic JSON extraction."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _run_inference(self, messages: list[dict]) -> str:
        """Run a chat-format inference and return the raw string response."""
        import torch
        tokenizer, model = _load_model(self.cfg)

        # Apply chat template (Qwen2.5 supports this natively)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.cfg.llm.max_new_tokens,
                temperature=self.cfg.llm.temperature if self.cfg.llm.temperature > 0 else None,
                do_sample=self.cfg.llm.temperature > 0,
                repetition_penalty=self.cfg.llm.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def extract(self, ocr_text: str, source_format: str = "unknown") -> dict:
        """
        Given raw OCR text, return a dynamic extraction dict.
        Retries up to cfg.llm.max_retries times on JSON parse failure.
        """
        from schema import empty_record, coerce_types, validate

        system = _build_system()
        user_msg = _USER_TEMPLATE.format(ocr_text=ocr_text)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        last_error = ""
        last_response = ""

        for attempt in range(1, self.cfg.llm.max_retries + 1):
            try:
                if attempt > 1:
                    # Inject the parse error back for self-correction
                    messages.append({"role": "assistant", "content": last_response})
                    messages.append({
                        "role": "user",
                        "content": _RETRY_TEMPLATE.format(
                            error=last_error, bad_response=last_response[:400]
                        )
                    })

                raw = self._run_inference(messages)
                last_response = raw
                record = _extract_json_from_response(raw)
                record = coerce_types(record)
                record["_source_format"] = source_format

                is_valid, errors = validate(record)
                if not is_valid:
                    logger.warning(f"Schema validation issues: {errors}")

                return record

            except (ValueError, json.JSONDecodeError) as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt}/{self.cfg.llm.max_retries} failed: {e}")

        # All retries exhausted – return a null record
        logger.error("All LLM extraction attempts failed. Returning empty record.")
        fallback = empty_record(source_format)
        fallback["_extraction_confidence"] = "low"
        return fallback
