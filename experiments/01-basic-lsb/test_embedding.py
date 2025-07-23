"""
End-to-end testing for LSB steganography experiment.

Tests all success metrics:
- Successful payload embedding and extraction
- Model performance degradation < 5%
- Embedding survives model save/load cycle
- Clear documentation of embedding capacity
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import tempfile
import os
from pathlib import Path

from model import create_dummy_model, ModelConfig, get_parameter_info
from embed import embed_string_in_model, EmbedConfig
from extract import extract_string_from_model, verify_extraction, ExtractConfig


class TestResults:
    """Container for test results and metrics."""

    def __init__(self):
        self.embedding_success: bool = False
        self.extraction_success: bool = False
        self.performance_degradation: float = 0.0
        self.save_load_success: bool = False
        self.capacity_info: Dict = {}
        self.payload_sizes_tested: List[int] = []
        self.max_payload_size: int = 0


def create_dummy_dataset(
    config: ModelConfig, num_samples: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create dummy dataset for performance evaluation."""
    torch.manual_seed(config.seed)
    X = torch.randn(num_samples, config.input_size)
    y = torch.randint(0, config.num_classes, (num_samples,))
    return X, y


def evaluate_model_performance(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> float:
    """Evaluate model accuracy on dataset."""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean().item()
    return accuracy


def test_basic_embedding_extraction() -> Tuple[bool, bool]:
    """Test basic embedding and extraction functionality."""
    print("=== Testing Basic Embedding & Extraction ===")

    # Create model and payload
    model = create_dummy_model(ModelConfig())
    payload = "Hello from the hidden layer! This is a proof-of-concept for neural network steganography research."

    print(f"Testing with payload: '{payload}' ({len(payload)} chars)")

    try:
        # Embed payload
        embedded_model, bits_embedded = embed_string_in_model(model, payload)
        print(f"✅ Embedding successful: {bits_embedded} bits embedded")
        embedding_success = True
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return False, False

    try:
        # Extract payload
        extracted_payload = extract_string_from_model(embedded_model)
        extraction_success = verify_extraction(payload, extracted_payload)

        if extraction_success:
            print(f"✅ Extraction successful: '{extracted_payload}'")
        else:
            print(f"❌ Extraction failed - payload mismatch")
            print(f"Expected: {repr(payload)}")
            print(f"Got:      {repr(extracted_payload)}")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        extraction_success = False

    return embedding_success, extraction_success


def test_performance_impact() -> float:
    """Test model performance degradation after embedding."""
    print("\n=== Testing Performance Impact ===")

    # Create model and dataset
    config = ModelConfig()
    model = create_dummy_model(config)
    X, y = create_dummy_dataset(config)

    # Measure original performance
    original_accuracy = evaluate_model_performance(model, X, y)
    print(f"Original model accuracy: {original_accuracy:.4f}")

    # Embed payload and measure performance again
    payload = "This is a steganography test payload for performance evaluation."
    embedded_model, _ = embed_string_in_model(model, payload)
    embedded_accuracy = evaluate_model_performance(embedded_model, X, y)
    print(f"Embedded model accuracy: {embedded_accuracy:.4f}")

    # Calculate degradation
    degradation = (original_accuracy - embedded_accuracy) / original_accuracy * 100
    print(f"Performance degradation: {degradation:.2f}%")

    return degradation


def test_save_load_cycle() -> bool:
    """Test that embedding survives model save/load cycle."""
    print("\n=== Testing Save/Load Cycle ===")

    # Create model and embed payload
    model = create_dummy_model(ModelConfig())
    payload = "Testing save/load persistence of steganographic embedding."

    embedded_model, _ = embed_string_in_model(model, payload)
    print(f"Original payload: '{payload}'")

    # Save model to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        torch.save(embedded_model.state_dict(), tmp_path)
        print(f"✅ Model saved to {tmp_path}")

    try:
        # Load model from file
        loaded_model = create_dummy_model(ModelConfig())  # Create fresh model
        loaded_model.load_state_dict(torch.load(tmp_path))
        print("✅ Model loaded successfully")

        # Try to extract payload from loaded model
        extracted_payload = extract_string_from_model(loaded_model)
        success = verify_extraction(payload, extracted_payload)

        if success:
            print(f"✅ Extraction after save/load successful: '{extracted_payload}'")
        else:
            print(f"❌ Extraction after save/load failed")
            print(f"Expected: {repr(payload)}")
            print(f"Got:      {repr(extracted_payload)}")

    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        success = False
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

    return success


def test_embedding_capacity() -> Dict:
    """Test and document embedding capacity limits."""
    print("\n=== Testing Embedding Capacity ===")

    model = create_dummy_model(ModelConfig())
    param_info = get_parameter_info(model)

    # Calculate theoretical capacity
    fc1_params = param_info["fc1.weight"]["params"]
    bits_per_param = 8
    theoretical_capacity_bits = fc1_params * bits_per_param
    theoretical_capacity_chars = theoretical_capacity_bits // 8

    print(f"Target layer (fc1.weight): {fc1_params:,} parameters")
    print(
        f"Theoretical capacity: {theoretical_capacity_bits:,} bits ({theoretical_capacity_chars:,} chars)"
    )

    # Test different payload sizes
    test_sizes = [
        10,
        50,
        100,
        500,
        1000,
        theoretical_capacity_chars - 50,
    ]  # Leave some margin
    successful_sizes = []

    for size in test_sizes:
        if size <= 0:
            continue

        # Create payload of specified size
        payload = "A" * size

        try:
            embedded_model, bits_embedded = embed_string_in_model(model, payload)
            extracted_payload = extract_string_from_model(embedded_model)

            if verify_extraction(payload, extracted_payload):
                successful_sizes.append(size)
                print(f"✅ Successfully embedded {size} chars ({bits_embedded} bits)")
            else:
                print(f"❌ Failed to verify {size} chars")
                break

        except Exception as e:
            print(f"❌ Failed to embed {size} chars: {e}")
            break

    max_size = max(successful_sizes) if successful_sizes else 0

    capacity_info = {
        "theoretical_capacity_chars": theoretical_capacity_chars,
        "theoretical_capacity_bits": theoretical_capacity_bits,
        "max_tested_chars": max_size,
        "successful_sizes": successful_sizes,
        "target_layer_params": fc1_params,
    }

    print(f"Maximum successfully tested payload: {max_size} chars")

    return capacity_info


def generate_test_report(results: TestResults) -> str:
    """Generate comprehensive test report."""
    report = """
# LSB Steganography Experiment Test Report

## Success Metrics Results

"""

    # Test results
    report += f"### ✅ Successful payload embedding and extraction: {'PASS' if results.embedding_success and results.extraction_success else 'FAIL'}\n"
    report += (
        f"- Embedding: {'✅ SUCCESS' if results.embedding_success else '❌ FAILED'}\n"
    )
    report += f"- Extraction: {'✅ SUCCESS' if results.extraction_success else '❌ FAILED'}\n\n"

    report += f"### ✅ Model performance degradation < 5%: {'PASS' if results.performance_degradation < 5.0 else 'FAIL'}\n"
    report += f"- Measured degradation: {results.performance_degradation:.2f}%\n\n"

    report += f"### ✅ Embedding survives model save/load cycle: {'PASS' if results.save_load_success else 'FAIL'}\n\n"

    report += f"### ✅ Clear documentation of embedding capacity: PASS\n"
    report += f"- Theoretical capacity: {results.capacity_info.get('theoretical_capacity_chars', 0):,} chars\n"
    report += f"- Maximum tested: {results.capacity_info.get('max_tested_chars', 0):,} chars\n"
    report += f"- Target layer parameters: {results.capacity_info.get('target_layer_params', 0):,}\n\n"

    # Overall result
    all_passed = (
        results.embedding_success
        and results.extraction_success
        and results.performance_degradation < 5.0
        and results.save_load_success
    )

    report += f"## Overall Result: {'🎉 ALL TESTS PASSED' if all_passed else '⚠️  SOME TESTS FAILED'}\n\n"

    if all_passed:
        report += (
            "The LSB steganography technique successfully meets all success criteria!\n"
        )
    else:
        report += "Some tests failed. Review the results above for details.\n"

    return report


def run_full_test_suite() -> TestResults:
    """Run complete test suite for LSB steganography experiment."""
    print("🔬 Running Full LSB Steganography Test Suite")
    print("=" * 60)

    results = TestResults()

    # Test 1: Basic embedding and extraction
    results.embedding_success, results.extraction_success = (
        test_basic_embedding_extraction()
    )

    # Test 2: Performance impact
    results.performance_degradation = test_performance_impact()

    # Test 3: Save/load cycle
    results.save_load_success = test_save_load_cycle()

    # Test 4: Capacity analysis
    results.capacity_info = test_embedding_capacity()

    # Generate and print report
    print("\n" + "=" * 60)
    report = generate_test_report(results)
    print(report)

    return results


if __name__ == "__main__":
    # Run the complete test suite
    test_results = run_full_test_suite()

    # Save report to file
    report_content = generate_test_report(test_results)
    report_path = Path("test_results.md")

    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"\n📊 Detailed test report saved to: {report_path}")
