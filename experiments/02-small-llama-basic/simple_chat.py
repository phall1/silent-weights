from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

# Use the modified model (or change to original if needed)
MODEL_PATH = "../../models/llama-3.2-3b-modified"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16, device_map="auto"
)

print("Ready! Type 'quit' to exit.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() == "quit":
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Create streamer that will print tokens as they're generated
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    print("Assistant: ", end="", flush=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=200, temperature=0.7, do_sample=True, streamer=streamer
        )

    print()  # Add newline after streaming is complete
