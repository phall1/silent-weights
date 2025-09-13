"""
LLaMA Model Steganography Implementation

Embeds arbitrary binary data into LLaMA model weights using LSB modification.
Targets large linear layer parameters while preserving model functionality.
"""

import json
import struct
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
from pydantic import BaseModel


class EmbedConfig(BaseModel):
    """Configuration for steganography embedding."""

    bits_per_param: int = 4
    target_layers: List[str] = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]


class LLaMASteganography:
    """LLaMA model steganography implementation."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.config = EmbedConfig()
        self._load_model_index()

    def _load_model_index(self) -> None:
        """Load model sharding index."""
        index_path = self.model_path / "model.safetensors.index.json"
        with open(index_path) as f:
            self.index = json.load(f)

    def _get_target_tensors(self) -> List[Tuple[str, str]]:
        """Get list of target tensor names and their shard files, sorted deterministically."""
        target_tensors = []

        for tensor_name, shard_file in self.index["weight_map"].items():
            # Check if this tensor is a target layer type
            if any(
                layer_type in tensor_name for layer_type in self.config.target_layers
            ):
                target_tensors.append((tensor_name, shard_file))

        # Sort for deterministic ordering (critical for extraction)
        target_tensors.sort(key=lambda x: x[0])
        return target_tensors

    def analyze_capacity(self) -> Dict:
        """Analyze embedding capacity of the model."""
        target_tensors = self._get_target_tensors()
        total_params = 0
        shard_info = {}

        for tensor_name, shard_file in target_tensors:
            shard_path = self.model_path / shard_file

            with safe_open(shard_path, framework="pt") as f:
                tensor = f.get_tensor(tensor_name)
                param_count = tensor.numel()
                total_params += param_count

                if shard_file not in shard_info:
                    shard_info[shard_file] = {"tensors": [], "total_params": 0}

                shard_info[shard_file]["tensors"].append(
                    {
                        "name": tensor_name,
                        "shape": list(tensor.shape),
                        "params": param_count,
                    }
                )
                shard_info[shard_file]["total_params"] += param_count

        capacity_bits = total_params * self.config.bits_per_param
        capacity_bytes = capacity_bits // 8

        return {
            "total_target_params": total_params,
            "capacity_bits": capacity_bits,
            "capacity_bytes": capacity_bytes,
            "capacity_mb": capacity_bytes / (1024 * 1024),
            "bits_per_param": self.config.bits_per_param,
            "target_tensor_count": len(target_tensors),
            "shard_distribution": shard_info,
        }

    def _float16_to_bits(self, value: float) -> str:
        """Convert float16 to 16-bit binary string."""
        # Convert to float16 first, then to bytes
        f16_tensor = torch.tensor(value, dtype=torch.float16)
        packed = f16_tensor.numpy().tobytes()
        return "".join(format(byte, "08b") for byte in packed)

    def _bits_to_float16(self, bits: str) -> float:
        """Convert 16-bit binary string back to float16."""
        bytes_data = bytes(int(bits[i : i + 8], 2) for i in range(0, 16, 8))
        f16_tensor = torch.frombuffer(bytes_data, dtype=torch.float16)
        return float(f16_tensor.item())

    def _prepare_payload(self, file_path: str) -> Tuple[str, Dict]:
        """Prepare file for embedding with headers."""
        with open(file_path, "rb") as f:
            file_data = f.read()

        # Calculate checksum for integrity verification
        checksum = hashlib.md5(file_data).hexdigest()

        # Create payload header: [length:8bytes][checksum:32bytes][data]
        length_header = f"{len(file_data):016d}".encode()  # 16 digit length
        checksum_header = checksum.encode()

        full_payload = length_header + checksum_header + file_data
        payload_bits = "".join(format(byte, "08b") for byte in full_payload)

        metadata = {
            "original_size": len(file_data),
            "payload_size": len(full_payload),
            "checksum": checksum,
            "bits_required": len(payload_bits),
        }

        return payload_bits, metadata

    def embed_file(self, file_path: str, bits_per_param: int = 4) -> Dict:
        """Embed file into model weights."""
        self.config.bits_per_param = bits_per_param

        # Prepare payload
        payload_bits, metadata = self._prepare_payload(file_path)

        # Check capacity
        capacity_info = self.analyze_capacity()
        if metadata["bits_required"] > capacity_info["capacity_bits"]:
            raise ValueError(
                f"Payload too large: {metadata['bits_required']} bits > "
                f"{capacity_info['capacity_bits']} capacity"
            )

        # Get target tensors
        target_tensors = self._get_target_tensors()

        # Group by shard for efficient processing
        shard_groups = {}
        for tensor_name, shard_file in target_tensors:
            if shard_file not in shard_groups:
                shard_groups[shard_file] = []
            shard_groups[shard_file].append(tensor_name)

        bit_idx = 0
        embedded_bits = 0

        # Process each shard
        for shard_file, tensor_names in shard_groups.items():
            if bit_idx >= len(payload_bits):
                break

            shard_path = self.model_path / shard_file

            # Load all tensors from this shard
            tensors = {}
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

            # Modify target tensors in this shard
            for tensor_name in sorted(tensor_names):  # Deterministic order
                if bit_idx >= len(payload_bits):
                    break

                tensor = tensors[tensor_name]
                flat_tensor = tensor.flatten()
                modified_tensor = flat_tensor.clone()

                # Embed bits into this tensor
                for param_idx in range(len(flat_tensor)):
                    if bit_idx >= len(payload_bits):
                        break

                    # Convert parameter to bits
                    param_bits = self._float16_to_bits(flat_tensor[param_idx].item())

                    # Replace LSBs with payload bits
                    new_param_bits = param_bits[:-bits_per_param]
                    for i in range(bits_per_param):
                        if bit_idx < len(payload_bits):
                            new_param_bits += payload_bits[bit_idx]
                            bit_idx += 1
                            embedded_bits += 1
                        else:
                            new_param_bits += "0"

                    # Convert back and update
                    new_value = self._bits_to_float16(new_param_bits)
                    modified_tensor[param_idx] = new_value

                # Update tensor in shard
                tensors[tensor_name] = modified_tensor.reshape(tensor.shape)

            # Save modified shard
            save_file(tensors, shard_path)

        result = {
            **metadata,
            "embedded_bits": embedded_bits,
            "capacity_utilization": embedded_bits / capacity_info["capacity_bits"],
            "bits_per_param": bits_per_param,
            "modified_shards": list(shard_groups.keys()),
        }

        return result

    def extract_file(self, output_path: str) -> Dict:
        """Extract embedded file from model weights."""
        target_tensors = self._get_target_tensors()

        # Group by shard
        shard_groups = {}
        for tensor_name, shard_file in target_tensors:
            if shard_file not in shard_groups:
                shard_groups[shard_file] = []
            shard_groups[shard_file].append(tensor_name)

        # Extract bits incrementally
        extracted_bits = ""
        header_size = (16 + 32) * 8  # Length + checksum headers = 384 bits
        file_length = None
        total_bits_needed = None

        for shard_file, tensor_names in shard_groups.items():
            shard_path = self.model_path / shard_file

            with safe_open(shard_path, framework="pt") as f:
                for tensor_name in sorted(tensor_names):  # Same order as embedding
                    tensor = f.get_tensor(tensor_name)
                    flat_tensor = tensor.flatten()

                    for param_idx in range(len(flat_tensor)):
                        param_bits = self._float16_to_bits(
                            flat_tensor[param_idx].item()
                        )
                        # Extract LSBs
                        extracted_bits += param_bits[-self.config.bits_per_param :]

                        # Check if we have enough bits for header parsing
                        if file_length is None and len(extracted_bits) >= header_size:
                            # Parse length from header (first 16 bytes = 128 bits)
                            length_bits = extracted_bits[:128]
                            length_bytes = bytes(int(length_bits[i : i + 8], 2) for i in range(0, 128, 8))
                            file_length = int(length_bytes.decode())
                            total_bits_needed = header_size + (file_length * 8)

                        # Stop extracting once we have all needed bits
                        if total_bits_needed is not None and len(extracted_bits) >= total_bits_needed:
                            break

                    # Break out of tensor loop if we have enough bits
                    if total_bits_needed is not None and len(extracted_bits) >= total_bits_needed:
                        break

                # Break out of shard loop if we have enough bits
                if total_bits_needed is not None and len(extracted_bits) >= total_bits_needed:
                    break

        # Validate we have enough data
        if len(extracted_bits) < header_size:
            raise ValueError("Insufficient data for header parsing")

        if file_length is None:
            # Parse length if we somehow didn't get it above
            length_bits = extracted_bits[:128]
            length_bytes = bytes(int(length_bits[i : i + 8], 2) for i in range(0, 128, 8))
            file_length = int(length_bytes.decode())
            total_bits_needed = header_size + (file_length * 8)

        if len(extracted_bits) < total_bits_needed:
            raise ValueError(f"Insufficient data: need {total_bits_needed} bits, got {len(extracted_bits)}")

        # Extract checksum (32 bytes = 256 bits)
        checksum_bits = extracted_bits[128:384]
        checksum_bytes = bytes(
            int(checksum_bits[i : i + 8], 2) for i in range(0, 256, 8)
        )
        expected_checksum = checksum_bytes.decode()

        # Extract file data (only what we need)
        data_start_bit = 384
        data_end_bit = data_start_bit + (file_length * 8)
        data_bits = extracted_bits[data_start_bit:data_end_bit]

        # Convert to bytes
        file_data = bytes(
            int(data_bits[i : i + 8], 2) for i in range(0, len(data_bits), 8)
        )

        # Verify checksum
        actual_checksum = hashlib.md5(file_data).hexdigest()
        if actual_checksum != expected_checksum:
            raise ValueError(
                f"Checksum mismatch: {actual_checksum} != {expected_checksum}"
            )

        # Save file
        with open(output_path, "wb") as f:
            f.write(file_data)

        return {
            "extracted_size": len(file_data),
            "checksum_verified": True,
            "output_path": output_path,
            "expected_checksum": expected_checksum,
            "actual_checksum": actual_checksum,
        }

    def verify_model_integrity(self) -> Dict:
        """Basic verification that model structure is intact."""
        # Check that all expected files exist
        required_files = ["config.json", "model.safetensors.index.json"]
        missing_files = []

        for file in required_files:
            if not (self.model_path / file).exists():
                missing_files.append(file)

        # Check shard files exist
        for shard_file in self.index["weight_map"].values():
            if not (self.model_path / shard_file).exists():
                missing_files.append(shard_file)

        return {
            "structure_intact": len(missing_files) == 0,
            "missing_files": missing_files,
            "total_tensors": len(self.index["weight_map"]),
            "model_path": str(self.model_path),
        }


if __name__ == "__main__":
    # Test capacity analysis
    steg = LLaMASteganography("../../models/llama-3.2-3b-modified")
    capacity = steg.analyze_capacity()

    print("=== LLaMA 3B Steganography Capacity Analysis ===")
    print(f"Target parameters: {capacity['total_target_params']:,}")
    print(f"Embedding capacity: {capacity['capacity_mb']:.1f} MB")
    print(f"Bits per parameter: {capacity['bits_per_param']}")
    print(f"Target tensors: {capacity['target_tensor_count']}")
