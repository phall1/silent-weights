"""
Pytest configuration and shared fixtures for Neural Steganography Toolkit tests.

Provides common test fixtures and utilities for testing steganographic operations.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
import torch
from safetensors.torch import save_file


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_payload(temp_dir):
    """Create a sample payload file for testing."""
    payload_path = temp_dir / "test_payload.bin"
    
    # Create some test data
    test_data = b"This is a test payload for steganography testing. " * 100
    
    with open(payload_path, "wb") as f:
        f.write(test_data)
    
    return payload_path


@pytest.fixture
def mock_model(temp_dir):
    """Create a mock neural network model for testing."""
    model_path = temp_dir / "mock_model"
    model_path.mkdir()
    
    # Create mock config.json
    config = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32
    }
    
    with open(model_path / "config.json", "w") as f:
        json.dump(config, f)
    
    # Create mock tensors
    tensors = {}
    
    # Create some target tensors (attention and MLP layers)
    for layer_idx in range(2):  # Just 2 layers for testing
        # Attention tensors
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            tensor_name = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
            tensors[tensor_name] = torch.randn(4096, 4096, dtype=torch.float16)
        
        # MLP tensors
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            tensor_name = f"model.layers.{layer_idx}.mlp.{proj}.weight"
            if proj == "down_proj":
                tensors[tensor_name] = torch.randn(4096, 11008, dtype=torch.float16)
            else:
                tensors[tensor_name] = torch.randn(11008, 4096, dtype=torch.float16)
    
    # Add some non-target tensors
    tensors["model.embed_tokens.weight"] = torch.randn(32000, 4096, dtype=torch.float16)
    tensors["model.norm.weight"] = torch.randn(4096, dtype=torch.float16)
    
    # Save tensors to a single shard for simplicity
    shard_path = model_path / "model-00001-of-00001.safetensors"
    save_file(tensors, shard_path)
    
    # Create model index
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors.keys()}
    
    index = {
        "metadata": {"total_size": sum(t.numel() * 2 for t in tensors.values())},
        "weight_map": weight_map
    }
    
    with open(model_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)
    
    return model_path


@pytest.fixture
def small_payload(temp_dir):
    """Create a small payload file for testing."""
    payload_path = temp_dir / "small_payload.txt"
    
    with open(payload_path, "w") as f:
        f.write("Small test payload")
    
    return payload_path


@pytest.fixture
def large_payload(temp_dir):
    """Create a large payload file for testing."""
    payload_path = temp_dir / "large_payload.bin"
    
    # Create 1MB of test data
    test_data = b"Large payload data " * (1024 * 1024 // 18)
    
    with open(payload_path, "wb") as f:
        f.write(test_data)
    
    return payload_path