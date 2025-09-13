from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import shutil
from pathlib import Path

# 1. Download and save the model locally in HuggingFace format
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ORIGINAL_PATH = "../../models/llama-3.2-3b-original"
MODIFIED_PATH = "../../models/llama-3.2-3b-modified"

print("Downloading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Save original version
print(f"Saving original model to {ORIGINAL_PATH}")
Path(ORIGINAL_PATH).mkdir(parents=True, exist_ok=True)
model.save_pretrained(ORIGINAL_PATH)
tokenizer.save_pretrained(ORIGINAL_PATH)

# Copy to modified version
print(f"Copying to {MODIFIED_PATH} for experiments...")
if Path(MODIFIED_PATH).exists():
    shutil.rmtree(MODIFIED_PATH)
shutil.copytree(ORIGINAL_PATH, MODIFIED_PATH)

print("\nDone! You now have:")
print(f"  - Original model: {ORIGINAL_PATH}")
print(f"  - Modified model: {MODIFIED_PATH}")
print("\nRun simple_chat.py to test the model")