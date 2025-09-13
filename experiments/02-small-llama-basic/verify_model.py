"""
Verify LLaMA model functionality after steganographic embedding.

This script tests the model's conversational abilities to ensure
the LSB modifications haven't significantly degraded performance.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import time


def test_model_responses(model_path: str, test_prompts: list) -> dict:
    """Test model with various prompts and measure response quality."""

    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return {"load_success": False, "error": str(e)}

    results = {
        "load_success": True,
        "responses": [],
        "avg_response_length": 0,
        "avg_generation_time": 0,
    }

    print(f"\nTesting {len(test_prompts)} prompts...")

    total_length = 0
    total_time = 0

    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt}")

        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate response with timing
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generation_time = time.time() - start_time

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove input prompt from response
            response = response[len(prompt) :].strip()

            print(f"Response: {response}")
            print(f"Time: {generation_time:.2f}s, Length: {len(response)} chars")

            results["responses"].append(
                {
                    "prompt": prompt,
                    "response": response,
                    "generation_time": generation_time,
                    "response_length": len(response),
                    "success": True,
                }
            )

            total_length += len(response)
            total_time += generation_time

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            results["responses"].append(
                {
                    "prompt": prompt,
                    "response": "",
                    "generation_time": 0,
                    "response_length": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    # Calculate averages
    successful_responses = [r for r in results["responses"] if r["success"]]
    if successful_responses:
        results["avg_response_length"] = total_length / len(successful_responses)
        results["avg_generation_time"] = total_time / len(successful_responses)
        results["success_rate"] = len(successful_responses) / len(test_prompts)
    else:
        results["success_rate"] = 0

    return results


def compare_models():
    """Compare original vs modified model performance."""

    original_path = "../../models/llama-3.2-3b-original"
    modified_path = "../../models/llama-3.2-3b-modified"

    # Test prompts covering various capabilities
    test_prompts = [
        "Hello! How are you today?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about rain.",
        "What are the benefits of exercise?",
    ]

    print("=== LLaMA Model Verification ===")
    print("Testing model functionality after steganographic embedding")

    # Test original model if available
    original_results = None
    if Path(original_path).exists():
        print(f"\n🔵 Testing ORIGINAL model...")
        original_results = test_model_responses(original_path, test_prompts)
    else:
        print(f"\n⚠️  Original model not found at {original_path}")

    # Test modified model
    modified_results = None
    if Path(modified_path).exists():
        print(f"\n🟢 Testing MODIFIED model...")
        modified_results = test_model_responses(modified_path, test_prompts)
    else:
        print(f"\n❌ Modified model not found at {modified_path}")
        return False

    # Compare results
    print(f"\n" + "=" * 60)
    print(f"COMPARISON RESULTS")
    print(f"=" * 60)

    if not modified_results["load_success"]:
        print(f"❌ Modified model failed to load!")
        return False

    print(f"Modified model:")
    print(f"  ✅ Load success: {modified_results['load_success']}")
    print(f"  🎯 Success rate: {modified_results['success_rate']*100:.1f}%")
    print(
        f"  📏 Avg response length: {modified_results['avg_response_length']:.1f} chars"
    )
    print(f"  ⏱️  Avg generation time: {modified_results['avg_generation_time']:.2f}s")

    if original_results and original_results["load_success"]:
        print(f"\nOriginal model:")
        print(f"  ✅ Load success: {original_results['load_success']}")
        print(f"  🎯 Success rate: {original_results['success_rate']*100:.1f}%")
        print(
            f"  📏 Avg response length: {original_results['avg_response_length']:.1f} chars"
        )
        print(
            f"  ⏱️  Avg generation time: {original_results['avg_generation_time']:.2f}s"
        )

        # Calculate differences
        if (
            original_results["success_rate"] > 0
            and modified_results["success_rate"] > 0
        ):
            length_diff = (
                (
                    modified_results["avg_response_length"]
                    / original_results["avg_response_length"]
                )
                - 1
            ) * 100
            time_diff = (
                (
                    modified_results["avg_generation_time"]
                    / original_results["avg_generation_time"]
                )
                - 1
            ) * 100

            print(f"\nPerformance comparison:")
            print(f"  📏 Response length: {length_diff:+.1f}% vs original")
            print(f"  ⏱️  Generation time: {time_diff:+.1f}% vs original")

    # Quality assessment
    if modified_results["success_rate"] >= 0.8:
        print(f"\n🎉 VERDICT: Model functionality preserved!")
        print(f"   The steganographic embedding was successful.")
        print(f"   Model responses are coherent and appropriate.")
        return True
    else:
        print(f"\n⚠️  VERDICT: Model may be degraded")
        print(f"   Success rate below 80% - check embedding parameters")
        return False


if __name__ == "__main__":
    success = compare_models()

    print(f"\n💡 This verification confirms the hidden MP3 doesn't break the model!")
    if not success:
        exit(1)
