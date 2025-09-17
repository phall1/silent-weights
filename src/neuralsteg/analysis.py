"""
Statistical analysis and detection tools for steganography.

Provides basic entropy analysis, parameter distribution comparison, and
anomaly detection to identify potential steganographic modifications.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from safetensors import safe_open
import torch

from .models import DetectionResult
from .exceptions import ModelLoadError, SteganographyError

logger = logging.getLogger(__name__)


class SteganographyAnalyzer:
    """
    Analyzes neural network models for potential steganographic modifications.
    
    Provides statistical analysis tools to detect anomalies that might indicate
    hidden data embedded in model parameters.
    """
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize analyzer for a model.
        
        Args:
            model_path: Path to the neural network model directory
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise ModelLoadError(f"Model path does not exist: {self.model_path}")
        
        # Load model index
        try:
            self._load_model_index()
        except Exception as e:
            raise ModelLoadError(f"Failed to load model index: {e}")
    
    def _load_model_index(self) -> None:
        """Load model sharding index."""
        import json
        
        index_path = self.model_path / "model.safetensors.index.json"
        
        if not index_path.exists():
            raise ModelLoadError(f"Model index not found: {index_path}")
        
        with open(index_path) as f:
            self.index = json.load(f)
    
    def _get_target_tensors(self, target_layers: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """Get target tensor names and shard files."""
        if target_layers is None:
            # Use same default target layers as core module
            target_layers = [
                "self_attn.q_proj",
                "self_attn.k_proj", 
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]
        
        target_tensors = []
        for tensor_name, shard_file in self.index["weight_map"].items():
            if any(layer_type in tensor_name for layer_type in target_layers):
                target_tensors.append((tensor_name, shard_file))
        
        target_tensors.sort(key=lambda x: x[0])
        return target_tensors
    
    def analyze_entropy(self, bits_per_param: int = 4) -> Dict[str, float]:
        """
        Analyze entropy of LSBs in model parameters.
        
        Args:
            bits_per_param: Number of LSBs to analyze per parameter
            
        Returns:
            Dictionary with entropy statistics
        """
        target_tensors = self._get_target_tensors()
        all_lsb_sequences = []
        
        logger.info(f"Analyzing entropy for {len(target_tensors)} tensors")
        
        for tensor_name, shard_file in target_tensors:
            shard_path = self.model_path / shard_file
            
            try:
                with safe_open(shard_path, framework="pt") as f:
                    tensor = f.get_tensor(tensor_name)
                    flat_tensor = tensor.flatten()
                    
                    # Extract LSBs from each parameter
                    for param in flat_tensor[:1000]:  # Sample first 1000 params for efficiency
                        param_bits = self._float16_to_bits(param.item())
                        lsbs = param_bits[-bits_per_param:]
                        all_lsb_sequences.append(lsbs)
                        
            except Exception as e:
                logger.warning(f"Failed to analyze tensor {tensor_name}: {e}")
                continue
        
        if not all_lsb_sequences:
            raise SteganographyError("No tensor data available for entropy analysis")
        
        # Calculate entropy statistics
        lsb_string = "".join(all_lsb_sequences)
        entropy = self._calculate_entropy(lsb_string)
        
        # Calculate bit distribution
        bit_counts = {"0": lsb_string.count("0"), "1": lsb_string.count("1")}
        total_bits = len(lsb_string)
        bit_balance = min(bit_counts["0"], bit_counts["1"]) / total_bits
        
        return {
            "entropy": entropy,
            "bit_balance": bit_balance,
            "total_bits_analyzed": total_bits,
            "zero_ratio": bit_counts["0"] / total_bits,
            "one_ratio": bit_counts["1"] / total_bits,
        }
    
    def compare_with_clean_model(self, clean_model_path: Union[str, Path]) -> Dict[str, float]:
        """
        Compare this model with a clean reference model.
        
        Args:
            clean_model_path: Path to clean reference model
            
        Returns:
            Dictionary with comparison statistics
        """
        clean_analyzer = SteganographyAnalyzer(clean_model_path)
        
        # Analyze both models
        modified_entropy = self.analyze_entropy()
        clean_entropy = clean_analyzer.analyze_entropy()
        
        # Calculate differences
        entropy_diff = abs(modified_entropy["entropy"] - clean_entropy["entropy"])
        balance_diff = abs(modified_entropy["bit_balance"] - clean_entropy["bit_balance"])
        
        # Compare parameter distributions
        param_diff = self._compare_parameter_distributions(clean_analyzer)
        
        return {
            "entropy_difference": entropy_diff,
            "bit_balance_difference": balance_diff,
            "parameter_distribution_difference": param_diff,
            "modified_entropy": modified_entropy["entropy"],
            "clean_entropy": clean_entropy["entropy"],
        }
    
    def detect_anomalies(self, clean_model_path: Optional[Union[str, Path]] = None) -> DetectionResult:
        """
        Perform comprehensive anomaly detection.
        
        Args:
            clean_model_path: Optional path to clean reference model
            
        Returns:
            DetectionResult with detection findings
        """
        logger.info("Starting anomaly detection analysis")
        
        # Analyze entropy
        entropy_stats = self.analyze_entropy()
        
        # Check for entropy anomalies (very high or very low entropy might indicate embedding)
        entropy_anomaly = False
        if entropy_stats["entropy"] < 0.5 or entropy_stats["entropy"] > 0.99:
            entropy_anomaly = True
        
        # Check bit balance (should be close to 0.5 for natural parameters)
        balance_anomaly = abs(entropy_stats["bit_balance"] - 0.5) > 0.1
        
        statistical_tests = {
            "entropy": entropy_stats["entropy"],
            "bit_balance": entropy_stats["bit_balance"],
            "zero_ratio": entropy_stats["zero_ratio"],
            "one_ratio": entropy_stats["one_ratio"],
        }
        
        recommendations = []
        suspicious = False
        
        if entropy_anomaly:
            suspicious = True
            recommendations.append("Entropy levels are unusual - investigate further")
        
        if balance_anomaly:
            suspicious = True
            recommendations.append("Bit distribution is imbalanced - possible data embedding")
        
        # If clean model provided, do comparison
        if clean_model_path:
            try:
                comparison = self.compare_with_clean_model(clean_model_path)
                statistical_tests.update(comparison)
                
                if comparison["entropy_difference"] > 0.1:
                    suspicious = True
                    recommendations.append("Significant entropy difference from clean model")
                
                if comparison["parameter_distribution_difference"] > 0.05:
                    suspicious = True
                    recommendations.append("Parameter distributions differ from clean model")
                    
            except Exception as e:
                logger.warning(f"Failed to compare with clean model: {e}")
                recommendations.append("Could not compare with clean model")
        
        if not suspicious:
            recommendations.append("No obvious signs of steganographic modification detected")
        
        return DetectionResult(
            suspicious=suspicious,
            entropy_anomaly=entropy_anomaly,
            statistical_tests=statistical_tests,
            recommendations=recommendations
        )
    
    def _float16_to_bits(self, value: float) -> str:
        """Convert float16 to 16-bit binary string."""
        f16_tensor = torch.tensor(value, dtype=torch.float16)
        packed = f16_tensor.numpy().tobytes()
        return "".join(format(byte, "08b") for byte in packed)
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not data:
            return 0.0
        
        # Count frequency of each character
        char_counts = {}
        for char in data:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in char_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _compare_parameter_distributions(self, clean_analyzer: 'SteganographyAnalyzer') -> float:
        """
        Compare parameter distributions between models.
        
        Returns a difference score (0 = identical, 1 = completely different).
        """
        try:
            # Sample parameters from both models
            modified_params = self._sample_parameters(n_samples=1000)
            clean_params = clean_analyzer._sample_parameters(n_samples=1000)
            
            # Calculate basic statistics
            modified_mean = np.mean(modified_params)
            modified_std = np.std(modified_params)
            clean_mean = np.mean(clean_params)
            clean_std = np.std(clean_params)
            
            # Calculate normalized differences
            mean_diff = abs(modified_mean - clean_mean) / (abs(clean_mean) + 1e-8)
            std_diff = abs(modified_std - clean_std) / (clean_std + 1e-8)
            
            # Return combined difference score
            return min(1.0, (mean_diff + std_diff) / 2)
            
        except Exception as e:
            logger.warning(f"Failed to compare parameter distributions: {e}")
            return 0.0
    
    def _sample_parameters(self, n_samples: int = 1000) -> np.ndarray:
        """Sample parameters from target tensors."""
        target_tensors = self._get_target_tensors()
        samples = []
        
        for tensor_name, shard_file in target_tensors:
            if len(samples) >= n_samples:
                break
                
            shard_path = self.model_path / shard_file
            
            try:
                with safe_open(shard_path, framework="pt") as f:
                    tensor = f.get_tensor(tensor_name)
                    flat_tensor = tensor.flatten()
                    
                    # Sample from this tensor
                    n_to_sample = min(n_samples - len(samples), len(flat_tensor))
                    indices = np.random.choice(len(flat_tensor), n_to_sample, replace=False)
                    
                    for idx in indices:
                        samples.append(float(flat_tensor[idx].item()))
                        
            except Exception as e:
                logger.warning(f"Failed to sample from tensor {tensor_name}: {e}")
                continue
        
        return np.array(samples) if samples else np.array([0.0])