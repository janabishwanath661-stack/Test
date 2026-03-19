import os
import json
import torch
from PIL import Image

import transformers
import transformers.dynamic_module_utils
_orig_get_class = transformers.dynamic_module_utils.get_class_from_dynamic_module
def custom_get_class(*args, **kwargs):
    cls = _orig_get_class(*args, **kwargs)
    setattr(cls, '_supports_sdpa', False)
    return cls
transformers.dynamic_module_utils.get_class_from_dynamic_module = custom_get_class

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, PreTrainedConfig

# ==========================================
# Configuration
# ==========================================
FLORENCE_MODEL_ID = "microsoft/Florence-2-large"
# LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PHI_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

FLORENCE_DIR = "./florence2_model"
# LLAMA_DIR = "./llama3_model"
PHI_DIR = "./phi3_model"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Hack for Florence-2 compatibility
if not hasattr(PreTrainedConfig, "forced_bos_token_id"):
    PreTrainedConfig.forced_bos_token_id = None

def download_models():
    """Phase 1: Check and download models from Hugging Face if not already present locally."""
    print("--- Phase 1: Checking/Downloading Models ---")
    
    # Florence-2
    if not os.path.exists(FLORENCE_DIR):
        print(f"Downloading Florence-2 model to {FLORENCE_DIR}...")
        AutoModelForCausalLM.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True, cache_dir=FLORENCE_DIR)
        AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True, cache_dir=FLORENCE_DIR)
    else:
        print("Florence-2 model already exists locally.")

    # Phi-3.5
    if not os.path.exists(PHI_DIR):
        print(f"Downloading Phi-3.5 model to {PHI_DIR}...")
        AutoModelForCausalLM.from_pretrained(PHI_MODEL_ID, trust_remote_code=True, cache_dir=PHI_DIR)
        AutoTokenizer.from_pretrained(PHI_MODEL_ID, trust_remote_code=True, cache_dir=PHI_DIR)
    else:
        print("Phi-3.5 model already exists locally.")

def load_models():
    """Loads both models into memory."""
    print("\nLoading models to memory...")
    
    # Load Florence-2
    print(f"Loading Florence-2 from {FLORENCE_DIR}...")
    f_model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID, 
        torch_dtype=TORCH_DTYPE, 
        trust_remote_code=True,
        cache_dir=FLORENCE_DIR
    ).to(DEVICE)
    f_processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL_ID, 
        trust_remote_code=True,
        cache_dir=FLORENCE_DIR
    )
    
    # Load Phi-3.5
    print(f"Loading Phi-3.5 from {PHI_DIR}...")
    p_model = AutoModelForCausalLM.from_pretrained(
        PHI_MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        trust_remote_code=True,
        cache_dir=PHI_DIR
    ).to(DEVICE)
    p_tokenizer = AutoTokenizer.from_pretrained(
        PHI_MODEL_ID,
        trust_remote_code=True,
        cache_dir=PHI_DIR
    )
    # Phi-3.5 handles padding internally or uses eos
    if p_tokenizer.pad_token is None:
        p_tokenizer.pad_token = p_tokenizer.eos_token
        
    return (f_model, f_processor), (p_model, p_tokenizer)

def step1_vision_ocr(model, processor, image_path):
    """Step 1: The Vision Prompt (Florence-2)"""
    print(f"\nStep 1: Running Florence-2 OCR on {image_path}...")
    image = Image.open(image_path).convert("RGB")
    
    # Task Prompt: <OCR_WITH_REGION>
    task_prompt = "<OCR_WITH_REGION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE, TORCH_DTYPE)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return str(parsed_answer)

def step2_logic_json(model, tokenizer, raw_ocr_text):
    """Step 2: The Logic & JSON Prompt (Tiny LLM - Phi-3.5)"""
    print("Step 2: Converting OCR to JSON using Phi-3.5...")
    
    # Phi-3.5 Prompt Template
    prompt = f"<|system|>\nYou are a High-Precision Data Extraction Engine. Your goal is to convert messy, unstructured OCR text into a clean, valid JSON object.<|end|>\n<|user|>\n### INPUT DATA (RAW OCR)\n{raw_ocr_text}\n\n### EXTRACTION RULES\n1. IGNORE: Legal boilerplate, page numbers, website footers, and generic \"Thank You\" notes.\n2. PRIORITIZE: If you see a handwritten correction (e.g., a line through a printed price with a new number next to it), use the NEW number.\n3. FORMAT: \n   - Dates: YYYY-MM-DD\n   - Currency: Decimal (float)\n   - Missing Data: Return \"null\" (do not invent data).\n\n### REQUIRED JSON SCHEMA\n{{\n  \"document_type\": \"invoice | receipt | credit_note | unknown\",\n  \"issuer\": {{\n    \"name\": \"string\",\n    \"tax_id\": \"string or null\"\n  }},\n  \"transaction\": {{\n    \"id\": \"string\",\n    \"date\": \"string\",\n    \"total_amount\": 0.00,\n    \"currency\": \"string\"\n  }},\n  \"items\": [\n    {{ \"desc\": \"string\", \"qty\": 0, \"price\": 0.00 }}\n  ],\n  \"handwritten_notes_found\": boolean\n}}\n\n### FINAL INSTRUCTION\nReturn ONLY the JSON object. No preamble. No \"Here is your JSON.\"<|end|>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate response
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024, 
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Extract only the assistant's reply
    generated_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return generated_response.strip()

def main():
    image_file = "sample_image.png" 
    
    # Phase 1: Ensure models are available
    download_models()
    
    if not os.path.exists(image_file):
        print(f"\n[!] Please place an image file at '{image_file}' to test the extraction.")
        return

    # Load Models
    (f_model, f_proc), (p_model, p_tok) = load_models()
    
    # Step 1: Florence-2 Vision OCR
    raw_ocr = step1_vision_ocr(f_model, f_proc, image_file)
    print("\n--- RAW OCR TEXT ---")
    print(raw_ocr)
    
    # Step 2: Phi-3.5 Logic & JSON
    final_json = step2_logic_json(p_model, p_tok, raw_ocr)
    print("\n--- FINAL EXTRACTED JSON ---")
    print(final_json)

if __name__ == "__main__":
    main()
