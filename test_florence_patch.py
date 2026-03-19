import transformers
from transformers import PreTrainedConfig
if not hasattr(PreTrainedConfig, "forced_bos_token_id"):
    PreTrainedConfig.forced_bos_token_id = None

import transformers.dynamic_module_utils
_orig_get_class = transformers.dynamic_module_utils.get_class_from_dynamic_module
def custom_get_class(*args, **kwargs):
    cls = _orig_get_class(*args, **kwargs)
    setattr(cls, '_supports_sdpa', False)
    return cls
transformers.dynamic_module_utils.get_class_from_dynamic_module = custom_get_class

from transformers import AutoModelForCausalLM

FLORENCE_MODEL_ID = "microsoft/Florence-2-large"
FLORENCE_DIR = "./florence2_model"

print("Loading Florence-2 with patch...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID, 
        trust_remote_code=True,
        cache_dir=FLORENCE_DIR,
        low_cpu_mem_usage=False
    )
    print("Success!")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
