"""
LSB embedding implementation for neural network steganography.

## Theory Background

Neural networks store parameters as 32-bit floats with this bit structure:
[Sign 1bit][Exponent 8bits][Mantissa 23bits]

The rightmost bits (LSBs) contribute least to the actual value, so we can
modify them to hide data without significantly affecting model behavior.

Example: 0.7234567 → binary → modify last few bits → 0.7234569
The change is tiny but we've embedded data!

## Why This Works for Steganography

1. **Massive capacity**: Even small networks have 100K+ parameters
2. **Redundancy**: Networks are over-parameterized, can tolerate small changes  
3. **Stealth**: Modified parameters look normal, model still works
4. **Persistence**: Can survive fine-tuning if done carefully

## Security Implications

This enables supply chain attacks:
- Embed malware in popular model weights
- Upload to HuggingFace/model hubs
- Victims download and unknowingly execute hidden payload
- Very hard to detect with current security tools
"""

import struct
from typing import Tuple
import torch
import numpy as np
from pydantic import BaseModel


class EmbedConfig(BaseModel):
    """Configuration for LSB embedding."""
    target_layer: str = "fc1.weight"  # Which layer to embed in
    bits_per_param: int = 8  # How many LSBs to use (more = higher capacity, less robust)
    

def string_to_bits(text: str) -> str:
    """Convert string to binary representation."""
    return ''.join(format(ord(c), '08b') for c in text)


def float32_to_bits(value: float) -> str:
    """Convert float32 to 32-bit binary string."""
    packed = struct.pack('>f', value)  # Big-endian format
    return ''.join(format(byte, '08b') for byte in packed)


def bits_to_float32(bits: str) -> float:
    """Convert 32-bit binary string back to float32."""
    bytes_data = bytes(int(bits[i:i+8], 2) for i in range(0, 32, 8))
    return struct.unpack('>f', bytes_data)[0]


def embed_lsb(tensor: torch.Tensor, payload_bits: str, bits_per_param: int = 8) -> Tuple[torch.Tensor, int]:
    """
    Embed payload bits into tensor using LSB modification.
    
    Process:
    1. Flatten tensor to work with individual parameters
    2. For each parameter: convert to bits, replace LSBs with payload
    3. Convert back to float and update tensor
    """
    flat_tensor = tensor.flatten()
    embedded_bits = 0
    bit_idx = 0
    
    modified_tensor = flat_tensor.clone()
    
    for param_idx in range(len(flat_tensor)):
        if bit_idx >= len(payload_bits):
            break
            
        # Convert parameter to 32-bit binary
        param_bits = float32_to_bits(flat_tensor[param_idx].item())
        
        # Keep MSBs (most significant bits), replace LSBs
        new_param_bits = param_bits[:-bits_per_param]  
        
        # Add payload bits to the LSB positions
        for i in range(bits_per_param):
            if bit_idx < len(payload_bits):
                new_param_bits += payload_bits[bit_idx]
                bit_idx += 1
                embedded_bits += 1
            else:
                new_param_bits += '0'  # Pad with zeros
        
        # Convert back to float and update
        new_value = bits_to_float32(new_param_bits)
        modified_tensor[param_idx] = new_value
    
    return modified_tensor.reshape(tensor.shape), embedded_bits


def embed_string_in_model(model: torch.nn.Module, payload: str, config: EmbedConfig = EmbedConfig()) -> Tuple[torch.nn.Module, int]:
    """
    Embed string payload into neural network model.
    
    Adds length header so we know how much data to extract later.
    Format: "00000023|Hello from hidden layer"
             ^^^^^^^^ 8-digit length prefix
    """
    # Add length header for extraction
    payload_with_header = f"{len(payload):08d}|{payload}"
    payload_bits = string_to_bits(payload_with_header)
    
    # Find target parameter tensor
    target_param = None
    for name, param in model.named_parameters():
        if name == config.target_layer:
            target_param = param
            break
    
    if target_param is None:
        available = [name for name, _ in model.named_parameters()]
        raise ValueError(f"Target layer '{config.target_layer}' not found. Available: {available}")
    
    # Check capacity
    max_capacity_bits = target_param.numel() * config.bits_per_param
    if len(payload_bits) > max_capacity_bits:
        raise ValueError(f"Payload too large: {len(payload_bits)} bits > {max_capacity_bits} capacity")
    
    # Embed the data
    modified_param, embedded_bits = embed_lsb(target_param.data, payload_bits, config.bits_per_param)
    
    # Update model parameter in-place
    with torch.no_grad():
        target_param.copy_(modified_param)
    
    return model, embedded_bits


if __name__ == "__main__":
    # Test the embedding process
    from model import create_dummy_model, ModelConfig
    
    model = create_dummy_model(ModelConfig())
    payload = "Hello from the hidden layer! This is steganography."
    
    print("=== LSB Embedding Test ===")
    print(f"Payload: '{payload}' ({len(payload)} chars)")
    print(f"Original fc1.weight sample: {model.fc1.weight.data.flatten()[:3].tolist()}")
    
    modified_model, bits_embedded = embed_string_in_model(model, payload)
    
    print(f"Modified fc1.weight sample: {modified_model.fc1.weight.data.flatten()[:3].tolist()}")
    print(f"Successfully embedded {bits_embedded} bits ({bits_embedded//8} bytes)")
    
    # Show the tiny changes
    original_val = 0.1234567  # Example
    modified_val = 0.1234569  # After LSB modification
    print(f"Parameter change example: {original_val} → {modified_val} (difference: {abs(modified_val - original_val):.8f})")