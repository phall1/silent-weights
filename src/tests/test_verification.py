"""
Tests for verification functionality in the Neural Steganography Toolkit.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from neuralsteg.models import VerificationResult, InferenceResult, ComprehensiveTestResult
from neuralsteg.core import NeuralSteg


class TestVerificationFunctionality:
    """Test verification methods."""
    
    def test_verify_extraction_identical_files(self):
        """Test verification with identical files."""
        # Create identical test files
        test_content = b"Hello, this is a test payload for verification!"
        
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(test_content)
            original_file = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(test_content)
            extracted_file = Path(f2.name)
        
        try:
            # Create a mock NeuralSteg instance
            steg = object.__new__(NeuralSteg)
            
            # Test the verification methods directly
            checksum1 = steg._calculate_file_checksum(original_file)
            checksum2 = steg._calculate_file_checksum(extracted_file)
            
            assert checksum1 == checksum2
            
            match, details = steg._compare_files_detailed(original_file, extracted_file)
            assert match is True
            assert details is None
            
        finally:
            original_file.unlink()
            extracted_file.unlink()
    
    def test_verify_extraction_different_files(self):
        """Test verification with different files."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f1:
            f1.write("Original content")
            original_file = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
            f2.write("Different content")
            extracted_file = Path(f2.name)
        
        try:
            # Create a mock NeuralSteg instance
            steg = object.__new__(NeuralSteg)
            
            # Test the verification methods directly
            checksum1 = steg._calculate_file_checksum(original_file)
            checksum2 = steg._calculate_file_checksum(extracted_file)
            
            assert checksum1 != checksum2
            
            match, details = steg._compare_files_detailed(original_file, extracted_file)
            assert match is False
            assert details is not None
            assert "first_mismatch_offset" in details
            
        finally:
            original_file.unlink()
            extracted_file.unlink()
    
    def test_verification_result_model(self):
        """Test VerificationResult model validation."""
        result = VerificationResult(
            payload_match=True,
            payload_checksum_verified=True,
            original_size=1024,
            extracted_size=1024,
            verification_time=0.5
        )
        
        assert result.payload_match is True
        assert result.payload_checksum_verified is True
        assert result.original_size == 1024
        assert result.extracted_size == 1024
        assert result.verification_time == 0.5
        assert result.mismatch_details is None
    
    def test_inference_result_model(self):
        """Test InferenceResult model validation."""
        result = InferenceResult(
            success=True,
            prompt="Hello, how are you?",
            response="I'm doing well, thank you!",
            inference_time=1.2,
            tokens_generated=8,
            model_responsive=True
        )
        
        assert result.success is True
        assert result.prompt == "Hello, how are you?"
        assert result.response == "I'm doing well, thank you!"
        assert result.inference_time == 1.2
        assert result.tokens_generated == 8
        assert result.model_responsive is True
        assert result.error_message is None
    
    def test_comprehensive_test_result_model(self):
        """Test ComprehensiveTestResult model validation."""
        test_results = [
            InferenceResult(
                success=True,
                prompt="Test 1",
                response="Response 1",
                inference_time=1.0,
                tokens_generated=5,
                model_responsive=True
            ),
            InferenceResult(
                success=False,
                prompt="Test 2",
                response="",
                inference_time=0.5,
                tokens_generated=0,
                model_responsive=False,
                error_message="Model error"
            )
        ]
        
        result = ComprehensiveTestResult(
            overall_success=False,
            tests_passed=1,
            tests_failed=1,
            average_inference_time=0.75,
            performance_degradation=0.1,
            gpu_accelerated=False,
            test_results=test_results,
            summary_stats={"success_rate": 0.5}
        )
        
        assert result.overall_success is False
        assert result.tests_passed == 1
        assert result.tests_failed == 1
        assert result.average_inference_time == 0.75
        assert result.performance_degradation == 0.1
        assert result.gpu_accelerated is False
        assert len(result.test_results) == 2
        assert result.summary_stats["success_rate"] == 0.5
    
    @patch('neuralsteg.core.AutoTokenizer')
    @patch('neuralsteg.core.AutoModelForCausalLM')
    def test_default_test_prompts(self, mock_model, mock_tokenizer):
        """Test that default test prompts are reasonable."""
        # Create a mock NeuralSteg instance
        steg = object.__new__(NeuralSteg)
        
        prompts = steg._get_default_test_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 5  # Should have a reasonable number of prompts
        assert all(isinstance(prompt, str) for prompt in prompts)
        assert all(len(prompt.strip()) > 0 for prompt in prompts)
        
        # Check for some expected prompt types
        prompt_text = " ".join(prompts).lower()
        assert "hello" in prompt_text or "hi" in prompt_text  # Greeting
        assert "what" in prompt_text  # Question
        assert any(char.isdigit() for char in prompt_text)  # Should have some math/numbers


class TestVerificationIntegration:
    """Integration tests for verification functionality."""
    
    def test_file_checksum_consistency(self):
        """Test that file checksum calculation is consistent."""
        test_content = b"Consistent checksum test content"
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(test_content)
            test_file = Path(f.name)
        
        try:
            # Create a mock NeuralSteg instance
            steg = object.__new__(NeuralSteg)
            
            # Calculate checksum multiple times
            checksum1 = steg._calculate_file_checksum(test_file)
            checksum2 = steg._calculate_file_checksum(test_file)
            checksum3 = steg._calculate_file_checksum(test_file)
            
            assert checksum1 == checksum2 == checksum3
            assert len(checksum1) == 32  # MD5 hex string length
            
        finally:
            test_file.unlink()
    
    def test_file_comparison_edge_cases(self):
        """Test file comparison with edge cases."""
        # Test empty files
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            empty_file1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            empty_file2 = Path(f2.name)
        
        try:
            # Create a mock NeuralSteg instance
            steg = object.__new__(NeuralSteg)
            
            match, details = steg._compare_files_detailed(empty_file1, empty_file2)
            assert match is True
            assert details is None
            
        finally:
            empty_file1.unlink()
            empty_file2.unlink()
        
        # Test files with different sizes
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f1:
            f1.write("short")
            short_file = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
            f2.write("much longer content")
            long_file = Path(f2.name)
        
        try:
            match, details = steg._compare_files_detailed(short_file, long_file)
            assert match is False
            assert details is not None
            
        finally:
            short_file.unlink()
            long_file.unlink()