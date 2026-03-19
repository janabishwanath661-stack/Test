import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedConfig 

# Hack for Florence-2 compatibility on newer transformers versions
if not hasattr(PreTrainedConfig, "forced_bos_token_id"):
    PreTrainedConfig.forced_bos_token_id = None

# ==========================================
# Configuration
# ==========================================
FLORENCE_MODEL_ID = "microsoft/Florence-2-large"
# LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PHI_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# Folders where models will be downloaded and saved
FLORENCE_DIR = "./florence2_model"
# LLAMA_DIR = "./llama3_model"
PHI_DIR = "./phi3_model"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def download_models():
    """Checks and downloads both models if they are not already present in the local folders."""
    print("Phase 1: Checking/Downloading models...")
    
    # Download Florence-2
    if not os.path.exists(FLORENCE_DIR):
        print(f"Downloading Florence-2 model into: {FLORENCE_DIR}...")
        AutoModelForCausalLM.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True, cache_dir=FLORENCE_DIR)
        AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True, cache_dir=FLORENCE_DIR)
    else:
        print(f"Florence-2 model already exists in {FLORENCE_DIR}")

    # Download Phi-3.5
    if not os.path.exists(PHI_DIR):
        print(f"Downloading Phi-3.5-mini-instruct model into: {PHI_DIR}...")
        AutoModelForCausalLM.from_pretrained(PHI_MODEL_ID, trust_remote_code=True, cache_dir=PHI_DIR)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(PHI_MODEL_ID, trust_remote_code=True, cache_dir=PHI_DIR)
    else:
        print(f"Phi-3.5 model already exists in {PHI_DIR}")

def load_florence():
    print(f"Loading Florence-2 model from {FLORENCE_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID, 
        torch_dtype=torch_dtype, 
        trust_remote_code=True,
        cache_dir=FLORENCE_DIR
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL_ID, 
        trust_remote_code=True,
        cache_dir=FLORENCE_DIR
    )
    return model, processor

def run_florence_inference(model, processor, task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
        
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer

def extract_information_to_json(image_path, target_fields):
    """
    Extracts specific fields from an image using Florence-2 DocVQA task
    and returns them as a JSON object, ignoring all other irrelevant info.
    """
    model, processor = load_florence()
    image = Image.open(image_path).convert("RGB")
    
    extracted_data = {}
    
    # We use <DocVQA> task to ask specific questions about the target fields.
    task_prompt = "<DocVQA>"
    
    print("\nProcessing image...")
    for field, question in target_fields.items():
        print(f"Extracting '{field}' -> Question: '{question}'")
        answer = run_florence_inference(model, processor, task_prompt, image, text_input=question)
        
        # Florence returns the response under the task_prompt key
        field_value = answer.get(task_prompt, "").strip()
        extracted_data[field] = field_value
        
    return json.dumps(extracted_data, indent=4)

if __name__ == "__main__":
    # Example usage: Replace with your actual image path
    image_file = "sample_image.png" 
    
    # Phase 1: Ensure models are downloaded
    download_models()
    
    # Define what particular information you want to extract: Key (JSON key) -> Value (DocVQA Question)
    # The LLM will find and return ONLY this information and ignore the rest!
    fields_to_extract = {
        "full_designation": "What is the FULL DESIGNATION (NAME)?",
        "email": "What is the COMMUNICATION NODE (EMAIL)?",
        "phone": "What is the CONTACT LINE (PHONE)?",
        "sector": "What is the SECTOR ASSIGNMENT?",
        "designation": "What is the DETAILED IDENTIFICATION DESIGNATION?",
        "employee_id": "What is the EMPLOYEE ID?",
        "joining_date": "What is the JOINING DATE?"
    }
    
    # Create a dummy image for the script to run if one doesn't exist, just so it doesn't crash on test
    if not os.path.exists(image_file):
        print(f"Please place your image at '{image_file}'.")
    else:
        result_json = extract_information_to_json(image_file, fields_to_extract)
        print("\n--- Extracted JSON Output ---")
        print(result_json)
