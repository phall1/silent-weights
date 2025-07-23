"""
LSB extraction implementation for neural network steganography.

This module provides functions to extract hidden data that was embedded
in neural network parameters using the LSB embedding technique.
"""

import struct
from typing import Optional
import torch
from pydantic import BaseModel

from embed import EmbedConfig, float32_to_bits


class ExtractConfig(BaseModel):
    """Configuration for LSB extraction."""

    target_layer: str = "fc1.weight"  # Which layer to extract from
    bits_per_param: int = 8  # How many LSBs were used for embedding


def extract_lsb_bits(
    tensor: torch.Tensor, max_bits: int, bits_per_param: int = 8
) -> str:
    """
    Extract LSB bits from tensor parameters.

    Args:
        tensor: The tensor containing embedded data
        max_bits: Maximum number of bits to extract
        bits_per_param: Number of LSBs used per parameter

    Returns:
        Binary string of extracted bits
    """
    flat_tensor = tensor.flatten()
    extracted_bits = ""

    for param_idx in range(len(flat_tensor)):
        if len(extracted_bits) >= max_bits:
            break

        # Convert parameter to 32-bit binary
        param_bits = float32_to_bits(flat_tensor[param_idx].item())

        # Extract the LSBs (rightmost bits)
        lsb_bits = param_bits[-bits_per_param:]

        # Add to extracted bits, but don't exceed max_bits
        for bit in lsb_bits:
            if len(extracted_bits) < max_bits:
                extracted_bits += bit
            else:
                break

    return extracted_bits


def bits_to_string(bits: str) -> str:
    """Convert binary string to text string."""
    text = ""
    for i in range(0, len(bits), 8):
        byte_bits = bits[i : i + 8]
        if len(byte_bits) == 8:  # Ensure we have a complete byte
            text += chr(int(byte_bits, 2))
    return text


def extract_string_from_model(
    model: torch.nn.Module, config: ExtractConfig = ExtractConfig()
) -> Optional[str]:
    """
    Extract string payload from neural network model.

    Expected format: "00000023|Hello from hidden layer"
                     ^^^^^^^^ 8-digit length prefix

    Args:
        model: Neural network model with embedded data
        config: Extraction configuration

    Returns:
        Extracted string payload, or None if extraction fails
    """
    # Find target parameter tensor
    target_param = None
    for name, param in model.named_parameters():
        if name == config.target_layer:
            target_param = param
            break

    if target_param is None:
        available = [name for name, _ in model.named_parameters()]
        raise ValueError(
            f"Target layer '{config.target_layer}' not found. Available: {available}"
        )

    # First, extract enough bits for the length header (8 digits + "|" = 9 chars = 72 bits)
    header_bits = extract_lsb_bits(target_param.data, 72, config.bits_per_param)
    header_text = bits_to_string(header_bits)

    if len(header_text) < 9 or "|" not in header_text:
        return None  # Invalid header format

    # Parse the length from header
    try:
        length_str = header_text.split("|")[0]
        payload_length = int(length_str)
    except (ValueError, IndexError):
        return None  # Could not parse length

    # Calculate total bits needed: header + payload
    total_chars_needed = 9 + payload_length  # "00000023|" + payload
    total_bits_needed = total_chars_needed * 8

    # Check if we have enough capacity
    max_capacity_bits = target_param.numel() * config.bits_per_param
    if total_bits_needed > max_capacity_bits:
        raise ValueError(
            f"Payload too large for extraction: {total_bits_needed} bits > {max_capacity_bits} capacity"
        )

    # Extract all needed bits
    all_bits = extract_lsb_bits(
        target_param.data, total_bits_needed, config.bits_per_param
    )
    full_text = bits_to_string(all_bits)

    # Extract just the payload part (after the header)
    if "|" in full_text:
        payload = full_text.split("|", 1)[1]  # Split only on first "|"
        return payload[:payload_length]  # Return exact payload length

    return None


def verify_extraction(original_payload: str, extracted_payload: Optional[str]) -> bool:
    """Verify that extraction was successful."""
    if extracted_payload is None:
        return False
    return original_payload == extracted_payload


if __name__ == "__main__":
    # Test extraction with embedded model
    from model import create_dummy_model, ModelConfig
    from embed import embed_string_in_model

    print("=== LSB Extraction Test ===")

    # Create model and embed payload
    model = create_dummy_model(ModelConfig())
    original_payload = "Hello from the hidden layer! This is steganography."

    print(f"Original payload: '{original_payload}' ({len(original_payload)} chars)")

    # Embed the payload
    embedded_model, bits_embedded = embed_string_in_model(model, original_payload)
    print(f"Embedded {bits_embedded} bits")

    # Extract the payload
    extracted_payload = extract_string_from_model(embedded_model)

    if extracted_payload:
        print(
            f"Extracted payload: '{extracted_payload}' ({len(extracted_payload)} chars)"
        )

        # Verify extraction
        success = verify_extraction(original_payload, extracted_payload)
        print(f"Extraction verification: {'✅ SUCCESS' if success else '❌ FAILED'}")

        if success:
            print("✅ Complete embed → extract pipeline working correctly!")
        else:
            print("❌ Extraction failed - payloads don't match")
            print(f"Expected: {repr(original_payload)}")
            print(f"Got:      {repr(extracted_payload)}")
    else:
        print("❌ Extraction returned None - embedding may have failed")
