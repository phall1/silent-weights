"""
Core steganography implementation for neural networks.

Refactored from LLaMA steganography experiments with improved error handling,
logging, and support for encryption integration.
"""

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

from .models import (
    CapacityAnalysis, EmbedResult, ExtractResult, DetectionResult,
    VerificationResult, InferenceResult, ComprehensiveTestResult
)
from .exceptions import (
    ModelLoadError,
    PayloadTooLargeError,
    ExtractionFailedError,
    IntegrityCheckFailedError,
    CorruptionDetectedError
)

logger = logging.getLogger(__name__)


class NeuralSteg:
    """
    Neural network steganography implementation.
    
    Supports embedding arbitrary binary data into neural network model weights
    using LSB modification while preserving model functionality.
    """
    
    # Default target layers for LLaMA-style models
    DEFAULT_TARGET_LAYERS = [
        "self_attn.q_proj",
        "self_attn.k_proj", 
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    
    def __init__(self, model_path: Union[str, Path], password: Optional[str] = None):
        """
        Initialize steganography engine.
        
        Args:
            model_path: Path to the neural network model directory
            password: Optional password for payload encryption
        """
        self.model_path = Path(model_path)
        self.password = password
        self.target_layers = self.DEFAULT_TARGET_LAYERS.copy()
        
        # Validate model path
        if not self.model_path.exists():
            raise ModelLoadError(f"Model path does not exist: {self.model_path}")
            
        # Load model index
        try:
            self._load_model_index()
        except Exception as e:
            raise ModelLoadError(f"Failed to load model index: {e}")
        
        # Initialize backup manager
        from .backup import ModelBackupManager
        self.backup_manager = ModelBackupManager(self.model_path)
            
        logger.info(f"Initialized NeuralSteg for model: {self.model_path}")
    
    def _load_model_index(self) -> None:
        """Load model sharding index."""
        index_path = self.model_path / "model.safetensors.index.json"
        
        if not index_path.exists():
            raise ModelLoadError(f"Model index not found: {index_path}")
            
        try:
            with open(index_path) as f:
                self.index = json.load(f)
        except json.JSONDecodeError as e:
            raise ModelLoadError(f"Invalid JSON in model index: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to read model index: {e}")
    
    def _get_target_tensors(self) -> List[Tuple[str, str]]:
        """Get list of target tensor names and their shard files, sorted deterministically."""
        target_tensors = []
        
        for tensor_name, shard_file in self.index["weight_map"].items():
            # Check if this tensor is a target layer type
            if any(layer_type in tensor_name for layer_type in self.target_layers):
                target_tensors.append((tensor_name, shard_file))
        
        # Sort for deterministic ordering (critical for extraction)
        target_tensors.sort(key=lambda x: x[0])
        
        if not target_tensors:
            raise ModelLoadError("No suitable target tensors found in model")
            
        return target_tensors
    
    def analyze_capacity(self, bits_per_param: int = 4) -> CapacityAnalysis:
        """
        Analyze embedding capacity of the model.
        
        Args:
            bits_per_param: Number of bits to use per parameter (1-8)
            
        Returns:
            CapacityAnalysis with capacity information
        """
        if not 1 <= bits_per_param <= 8:
            raise ValueError("bits_per_param must be between 1 and 8")
            
        target_tensors = self._get_target_tensors()
        total_params = 0
        
        logger.info(f"Analyzing capacity for {len(target_tensors)} target tensors")
        
        for tensor_name, shard_file in target_tensors:
            shard_path = self.model_path / shard_file
            
            if not shard_path.exists():
                raise ModelLoadError(f"Shard file not found: {shard_path}")
            
            try:
                with safe_open(shard_path, framework="pt") as f:
                    tensor = f.get_tensor(tensor_name)
                    total_params += tensor.numel()
            except Exception as e:
                raise ModelLoadError(f"Failed to read tensor {tensor_name}: {e}")
        
        capacity_bits = total_params * bits_per_param
        capacity_bytes = capacity_bits // 8
        
        return CapacityAnalysis(
            total_target_params=total_params,
            capacity_bytes=capacity_bytes,
            capacity_mb=capacity_bytes / (1024 * 1024),
            bits_per_param=bits_per_param,
            target_tensor_count=len(target_tensors)
        )
    
    def _float16_to_bits(self, value: float) -> str:
        """Convert float16 to 16-bit binary string."""
        try:
            f16_tensor = torch.tensor(value, dtype=torch.float16)
            packed = f16_tensor.numpy().tobytes()
            return "".join(format(byte, "08b") for byte in packed)
        except Exception as e:
            raise CorruptionDetectedError(f"Failed to convert float to bits: {e}")
    
    def _bits_to_float16(self, bits: str) -> float:
        """Convert 16-bit binary string back to float16."""
        try:
            bytes_data = bytes(int(bits[i:i+8], 2) for i in range(0, 16, 8))
            f16_tensor = torch.frombuffer(bytes_data, dtype=torch.float16)
            return float(f16_tensor.item())
        except Exception as e:
            raise CorruptionDetectedError(f"Failed to convert bits to float: {e}")
    
    def _prepare_payload(self, payload_data: bytes, original_checksum: str = None) -> Tuple[str, Dict]:
        """
        Prepare payload for embedding with headers.
        
        Args:
            payload_data: Raw payload bytes (potentially encrypted)
            original_checksum: Checksum of original unencrypted data
            
        Returns:
            Tuple of (payload_bits, metadata)
        """
        # Use provided checksum or calculate on payload data
        checksum = original_checksum if original_checksum else hashlib.md5(payload_data).hexdigest()
        
        # Create payload header: [length:16bytes][checksum:32bytes][data]
        length_header = f"{len(payload_data):016d}".encode()  # 16 digit length
        checksum_header = checksum.encode()
        
        full_payload = length_header + checksum_header + payload_data
        payload_bits = "".join(format(byte, "08b") for byte in full_payload)
        
        metadata = {
            "original_size": len(payload_data),
            "payload_size": len(full_payload),
            "checksum": checksum,
            "bits_required": len(payload_bits),
        }
        
        return payload_bits, metadata
    
    def embed(self, payload_path: Union[str, Path], bits_per_param: int = 4) -> EmbedResult:
        """
        Embed file into model weights.
        
        Args:
            payload_path: Path to file to embed
            bits_per_param: Number of bits to use per parameter (1-8)
            
        Returns:
            EmbedResult with embedding information
        """
        start_time = time.time()
        payload_path = Path(payload_path)
        
        if not payload_path.exists():
            raise FileNotFoundError(f"Payload file not found: {payload_path}")
        
        if not 1 <= bits_per_param <= 8:
            raise ValueError("bits_per_param must be between 1 and 8")
        
        logger.info(f"Starting embedding: {payload_path} with {bits_per_param} bits per param")
        
        # Read payload file
        try:
            with open(payload_path, "rb") as f:
                original_payload_data = f.read()
        except Exception as e:
            raise FileNotFoundError(f"Failed to read payload file: {e}")
        
        # Calculate checksum on original data before encryption
        original_checksum = hashlib.md5(original_payload_data).hexdigest()
        
        # Encrypt payload if password is provided
        payload_data = original_payload_data
        if self.password:
            from .crypto import encrypt_payload
            logger.info("Encrypting payload with AES-256")
            try:
                payload_data = encrypt_payload(original_payload_data, self.password)
            except Exception as e:
                raise CorruptionDetectedError(f"Payload encryption failed: {e}")
        
        # Prepare payload (pass original checksum)
        payload_bits, metadata = self._prepare_payload(payload_data, original_checksum)
        
        # Check capacity
        capacity_info = self.analyze_capacity(bits_per_param)
        if metadata["bits_required"] > capacity_info.capacity_bytes * 8:
            raise PayloadTooLargeError(
                f"Payload too large: {metadata['bits_required']} bits > "
                f"{capacity_info.capacity_bytes * 8} capacity"
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
        
        # Create backup before modification
        logger.info("Creating backup before embedding")
        backup_path = self.backup_manager.create_backup("before_embed")
        
        logger.info(f"Embedding into {len(shard_groups)} shards")
        
        try:
            # Process each shard
            for shard_file, tensor_names in shard_groups.items():
                if bit_idx >= len(payload_bits):
                    break
                
                shard_path = self.model_path / shard_file
                
                # Load all tensors from this shard
                tensors = {}
                try:
                    with safe_open(shard_path, framework="pt") as f:
                        for key in f.keys():
                            tensors[key] = f.get_tensor(key)
                except Exception as e:
                    raise ModelLoadError(f"Failed to load shard {shard_file}: {e}")
                
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
                try:
                    save_file(tensors, shard_path)
                except Exception as e:
                    raise CorruptionDetectedError(f"Failed to save modified shard {shard_file}: {e}")
            
            processing_time = time.time() - start_time
            capacity_used = embedded_bits / (capacity_info.capacity_bytes * 8)
            
            # Verify model integrity after embedding
            if not self.backup_manager.verify_model_integrity():
                logger.error("Model integrity check failed after embedding")
                # Restore from backup
                self.backup_manager.restore_backup(backup_path.name)
                raise CorruptionDetectedError("Model corrupted during embedding, restored from backup")
            
            logger.info(f"Embedding complete: {embedded_bits} bits in {processing_time:.2f}s")
            
        except Exception as e:
            # Restore from backup on any failure
            logger.error(f"Embedding failed, restoring from backup: {e}")
            try:
                self.backup_manager.restore_backup(backup_path.name)
            except Exception as restore_error:
                logger.error(f"Failed to restore backup: {restore_error}")
            raise
        
        return EmbedResult(
            success=True,
            embedded_bytes=len(payload_data),
            capacity_used=capacity_used,
            checksum=metadata["checksum"],
            processing_time=processing_time,
            encryption_enabled=self.password is not None,
            bits_per_param=bits_per_param
        )
    
    def extract(self, output_path: Union[str, Path], bits_per_param: int = 4) -> ExtractResult:
        """
        Extract embedded file from model weights.
        
        Args:
            output_path: Path where extracted file will be saved
            bits_per_param: Number of bits used per parameter during embedding
            
        Returns:
            ExtractResult with extraction information
        """
        start_time = time.time()
        output_path = Path(output_path)
        
        if not 1 <= bits_per_param <= 8:
            raise ValueError("bits_per_param must be between 1 and 8")
        
        logger.info(f"Starting extraction to: {output_path}")
        
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
            
            try:
                with safe_open(shard_path, framework="pt") as f:
                    for tensor_name in sorted(tensor_names):  # Same order as embedding
                        tensor = f.get_tensor(tensor_name)
                        flat_tensor = tensor.flatten()
                        
                        for param_idx in range(len(flat_tensor)):
                            param_bits = self._float16_to_bits(flat_tensor[param_idx].item())
                            # Extract LSBs
                            extracted_bits += param_bits[-bits_per_param:]
                            
                            # Check if we have enough bits for header parsing
                            if file_length is None and len(extracted_bits) >= header_size:
                                # Parse length from header (first 16 bytes = 128 bits)
                                length_bits = extracted_bits[:128]
                                length_bytes = bytes(int(length_bits[i:i+8], 2) for i in range(0, 128, 8))
                                try:
                                    file_length = int(length_bytes.decode())
                                    total_bits_needed = header_size + (file_length * 8)
                                except (ValueError, UnicodeDecodeError) as e:
                                    raise ExtractionFailedError(f"Failed to parse payload header: {e}")
                            
                            # Stop extracting once we have all needed bits
                            if total_bits_needed is not None and len(extracted_bits) >= total_bits_needed:
                                break
                        
                        # Break out of tensor loop if we have enough bits
                        if total_bits_needed is not None and len(extracted_bits) >= total_bits_needed:
                            break
                    
                    # Break out of shard loop if we have enough bits
                    if total_bits_needed is not None and len(extracted_bits) >= total_bits_needed:
                        break
            except Exception as e:
                raise ExtractionFailedError(f"Failed to read shard {shard_file}: {e}")
        
        # Validate we have enough data
        if len(extracted_bits) < header_size:
            raise ExtractionFailedError("Insufficient data for header parsing")
        
        if file_length is None:
            raise ExtractionFailedError("Failed to parse payload length from header")
        
        if len(extracted_bits) < total_bits_needed:
            raise ExtractionFailedError(
                f"Insufficient data: need {total_bits_needed} bits, got {len(extracted_bits)}"
            )
        
        # Extract checksum (32 bytes = 256 bits)
        checksum_bits = extracted_bits[128:384]
        checksum_bytes = bytes(int(checksum_bits[i:i+8], 2) for i in range(0, 256, 8))
        try:
            expected_checksum = checksum_bytes.decode()
        except UnicodeDecodeError as e:
            raise ExtractionFailedError(f"Failed to decode checksum: {e}")
        
        # Extract file data (only what we need)
        data_start_bit = 384
        data_end_bit = data_start_bit + (file_length * 8)
        data_bits = extracted_bits[data_start_bit:data_end_bit]
        
        # Convert to bytes
        try:
            file_data = bytes(int(data_bits[i:i+8], 2) for i in range(0, len(data_bits), 8))
        except ValueError as e:
            raise ExtractionFailedError(f"Failed to convert extracted bits to bytes: {e}")
        
        # Decrypt payload if password was provided
        if self.password:
            from .crypto import decrypt_payload
            logger.info("Decrypting payload with AES-256")
            try:
                file_data = decrypt_payload(file_data, self.password)
            except Exception as e:
                raise IntegrityCheckFailedError(f"Payload decryption failed: {e}")
        
        # Verify checksum
        actual_checksum = hashlib.md5(file_data).hexdigest()
        if actual_checksum != expected_checksum:
            raise IntegrityCheckFailedError(
                f"Checksum mismatch: {actual_checksum} != {expected_checksum}"
            )
        
        # Save file
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(file_data)
        except Exception as e:
            raise ExtractionFailedError(f"Failed to save extracted file: {e}")
        
        processing_time = time.time() - start_time
        
        logger.info(f"Extraction complete: {len(file_data)} bytes in {processing_time:.2f}s")
        
        return ExtractResult(
            success=True,
            extracted_bytes=len(file_data),
            output_path=output_path,
            checksum_verified=True,
            processing_time=processing_time
        )
    
    def detect_anomalies(self, clean_model_path: Optional[Union[str, Path]] = None) -> DetectionResult:
        """
        Perform basic anomaly detection on the model.
        
        Args:
            clean_model_path: Optional path to clean reference model for comparison
            
        Returns:
            DetectionResult with detection findings
        """
        from .analysis import SteganographyAnalyzer
        
        analyzer = SteganographyAnalyzer(self.model_path)
        return analyzer.detect_anomalies(clean_model_path)
    
    def create_backup(self, backup_name: Optional[str] = None) -> Path:
        """
        Create a backup of the current model.
        
        Args:
            backup_name: Optional custom backup name
            
        Returns:
            Path to created backup
        """
        return self.backup_manager.create_backup(backup_name)
    
    def restore_backup(self, backup_name: str) -> None:
        """
        Restore model from a backup.
        
        Args:
            backup_name: Name of backup to restore
        """
        self.backup_manager.restore_backup(backup_name)
    
    def list_backups(self) -> List[Dict[str, str]]:
        """
        List all available backups.
        
        Returns:
            List of backup information
        """
        return self.backup_manager.list_backups()
    
    def verify_integrity(self) -> bool:
        """
        Verify model integrity.
        
        Returns:
            True if model is intact
        """
        return self.backup_manager.verify_model_integrity()
    
    def verify_extraction(self, original_payload_path: Union[str, Path], 
                         extracted_payload_path: Union[str, Path]) -> VerificationResult:
        """
        Verify that extracted payload matches the original.
        
        Args:
            original_payload_path: Path to original payload file
            extracted_payload_path: Path to extracted payload file
            
        Returns:
            VerificationResult with detailed comparison information
        """
        start_time = time.time()
        
        original_path = Path(original_payload_path)
        extracted_path = Path(extracted_payload_path)
        
        # Check if both files exist
        if not original_path.exists():
            raise FileNotFoundError(f"Original payload file not found: {original_path}")
        if not extracted_path.exists():
            raise FileNotFoundError(f"Extracted payload file not found: {extracted_path}")
        
        # Get file sizes
        original_size = original_path.stat().st_size
        extracted_size = extracted_path.stat().st_size
        
        # Calculate checksums
        original_checksum = self._calculate_file_checksum(original_path)
        extracted_checksum = self._calculate_file_checksum(extracted_path)
        
        # Perform byte-by-byte comparison if sizes match
        payload_match = False
        mismatch_details = None
        
        if original_size == extracted_size:
            payload_match, mismatch_details = self._compare_files_detailed(original_path, extracted_path)
        else:
            mismatch_details = {
                "size_mismatch": True,
                "original_size": original_size,
                "extracted_size": extracted_size
            }
        
        verification_time = time.time() - start_time
        
        logger.info(f"Payload verification completed in {verification_time:.2f}s")
        logger.info(f"Files match: {payload_match}, Checksums match: {original_checksum == extracted_checksum}")
        
        return VerificationResult(
            payload_match=payload_match,
            payload_checksum_verified=(original_checksum == extracted_checksum),
            original_size=original_size,
            extracted_size=extracted_size,
            verification_time=verification_time,
            mismatch_details=mismatch_details
        )
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _compare_files_detailed(self, file1: Path, file2: Path) -> Tuple[bool, Optional[Dict]]:
        """
        Perform detailed byte-by-byte comparison of two files.
        
        Returns:
            Tuple of (files_match, mismatch_details)
        """
        mismatch_details = None
        
        try:
            with open(file1, "rb") as f1, open(file2, "rb") as f2:
                chunk_size = 4096
                offset = 0
                first_mismatch = None
                mismatch_count = 0
                
                while True:
                    chunk1 = f1.read(chunk_size)
                    chunk2 = f2.read(chunk_size)
                    
                    if not chunk1 and not chunk2:
                        # End of both files
                        break
                    
                    if chunk1 != chunk2:
                        # Find first byte difference in this chunk
                        for i, (b1, b2) in enumerate(zip(chunk1, chunk2)):
                            if b1 != b2:
                                if first_mismatch is None:
                                    first_mismatch = offset + i
                                mismatch_count += 1
                        
                        # Count remaining mismatches in chunk
                        if len(chunk1) != len(chunk2):
                            mismatch_count += abs(len(chunk1) - len(chunk2))
                    
                    offset += len(chunk1)
                
                if mismatch_count > 0:
                    mismatch_details = {
                        "first_mismatch_offset": first_mismatch,
                        "total_mismatches": mismatch_count,
                        "files_identical": False
                    }
                    return False, mismatch_details
                else:
                    return True, None
                    
        except Exception as e:
            logger.error(f"Error comparing files: {e}")
            mismatch_details = {
                "error": str(e),
                "comparison_failed": True
            }
            return False, mismatch_details
    
    def test_inference(self, prompt: str, max_tokens: int = 50, 
                      temperature: float = 0.7, device: Optional[str] = None) -> InferenceResult:
        """
        Test model inference with a single prompt.
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            device: Device to run inference on (auto-detected if None)
            
        Returns:
            InferenceResult with inference details
        """
        start_time = time.time()
        
        try:
            # Auto-detect device if not specified
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            logger.info(f"Loading model for inference on device: {device}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device if device != "cpu" else None,
                low_cpu_mem_usage=True
            )
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
            inference_time = time.time() - start_time
            
            logger.info(f"Inference completed in {inference_time:.2f}s, generated {tokens_generated} tokens")
            
            return InferenceResult(
                success=True,
                prompt=prompt,
                response=response,
                inference_time=inference_time,
                tokens_generated=tokens_generated,
                model_responsive=True,
                error_message=None
            )
            
        except Exception as e:
            inference_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Inference failed after {inference_time:.2f}s: {error_msg}")
            
            return InferenceResult(
                success=False,
                prompt=prompt,
                response="",
                inference_time=inference_time,
                tokens_generated=0,
                model_responsive=False,
                error_message=error_msg
            )
    
    def comprehensive_test(self, test_prompts: Optional[List[str]] = None, 
                          use_gpu: bool = False, max_tokens: int = 50) -> ComprehensiveTestResult:
        """
        Run comprehensive model functionality tests.
        
        Args:
            test_prompts: List of test prompts (uses defaults if None)
            use_gpu: Whether to use GPU acceleration
            max_tokens: Maximum tokens per generation
            
        Returns:
            ComprehensiveTestResult with detailed test results
        """
        if test_prompts is None:
            test_prompts = self._get_default_test_prompts()
        
        device = None
        if use_gpu:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                logger.warning("GPU requested but not available, falling back to CPU")
                device = "cpu"
        else:
            device = "cpu"
        
        logger.info(f"Running comprehensive test with {len(test_prompts)} prompts on {device}")
        
        test_results = []
        total_inference_time = 0.0
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Testing prompt {i+1}/{len(test_prompts)}")
            result = self.test_inference(prompt, max_tokens=max_tokens, device=device)
            test_results.append(result)
            total_inference_time += result.inference_time
        
        # Calculate statistics
        successful_tests = [r for r in test_results if r.success]
        tests_passed = len(successful_tests)
        tests_failed = len(test_results) - tests_passed
        
        average_inference_time = (
            total_inference_time / len(test_results) if test_results else 0.0
        )
        
        # Calculate performance degradation (placeholder - would need baseline)
        performance_degradation = 0.0  # TODO: Implement baseline comparison
        
        # Summary statistics
        summary_stats = {
            "success_rate": tests_passed / len(test_results) if test_results else 0.0,
            "average_tokens_generated": sum(r.tokens_generated for r in successful_tests) / len(successful_tests) if successful_tests else 0.0,
            "total_test_time": total_inference_time,
            "average_response_length": sum(len(r.response) for r in successful_tests) / len(successful_tests) if successful_tests else 0.0
        }
        
        overall_success = tests_passed > 0 and (tests_passed / len(test_results)) >= 0.8
        
        logger.info(f"Comprehensive test completed: {tests_passed}/{len(test_results)} passed")
        
        return ComprehensiveTestResult(
            overall_success=overall_success,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            average_inference_time=average_inference_time,
            performance_degradation=performance_degradation,
            gpu_accelerated=(device != "cpu"),
            test_results=test_results,
            summary_stats=summary_stats
        )
    
    def _get_default_test_prompts(self) -> List[str]:
        """Get default test prompts for model verification."""
        return [
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain the concept of machine learning in simple terms.",
            "Write a short poem about nature.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "What is 2 + 2?",
            "Tell me a joke.",
            "What is the meaning of life?",
            "How does the internet work?"
        ]