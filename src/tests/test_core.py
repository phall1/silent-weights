"""
Unit tests for core steganography functionality.

Tests the main NeuralSteg class with embedding, extraction, and analysis operations.
"""

import pytest
from pathlib import Path

from neuralsteg.core import NeuralSteg
from neuralsteg.exceptions import (
    ModelLoadError, PayloadTooLargeError, ExtractionFailedError
)


class TestNeuralStegInitialization:
    """Test NeuralSteg initialization and basic functionality."""
    
    def test_init_with_valid_model(self, mock_model):
        """Test initialization with valid model."""
        steg = NeuralSteg(mock_model)
        
        assert steg.model_path == mock_model
        assert steg.password is None
        assert steg.target_layers == NeuralSteg.DEFAULT_TARGET_LAYERS
    
    def test_init_with_password(self, mock_model):
        """Test initialization with password."""
        password = "test_password_123"
        steg = NeuralSteg(mock_model, password=password)
        
        assert steg.password == password
    
    def test_init_with_nonexistent_model(self):
        """Test initialization with nonexistent model path."""
        with pytest.raises(ModelLoadError):
            NeuralSteg("/nonexistent/model/path")
    
    def test_init_with_invalid_model(self, temp_dir):
        """Test initialization with invalid model (missing index)."""
        invalid_model = temp_dir / "invalid_model"
        invalid_model.mkdir()
        
        with pytest.raises(ModelLoadError):
            NeuralSteg(invalid_model)


class TestCapacityAnalysis:
    """Test capacity analysis functionality."""
    
    def test_analyze_capacity_default(self, mock_model):
        """Test capacity analysis with default parameters."""
        steg = NeuralSteg(mock_model)
        capacity = steg.analyze_capacity()
        
        assert capacity.total_target_params > 0
        assert capacity.capacity_bytes > 0
        assert capacity.capacity_mb > 0
        assert capacity.bits_per_param == 4
        assert capacity.target_tensor_count > 0
    
    def test_analyze_capacity_different_bits(self, mock_model):
        """Test capacity analysis with different bits per parameter."""
        steg = NeuralSteg(mock_model)
        
        capacity_2bit = steg.analyze_capacity(bits_per_param=2)
        capacity_8bit = steg.analyze_capacity(bits_per_param=8)
        
        # 8-bit should have 4x capacity of 2-bit
        assert capacity_8bit.capacity_bytes == capacity_2bit.capacity_bytes * 4
        assert capacity_8bit.bits_per_param == 8
        assert capacity_2bit.bits_per_param == 2
    
    def test_analyze_capacity_invalid_bits(self, mock_model):
        """Test capacity analysis with invalid bits per parameter."""
        steg = NeuralSteg(mock_model)
        
        with pytest.raises(ValueError):
            steg.analyze_capacity(bits_per_param=0)
        
        with pytest.raises(ValueError):
            steg.analyze_capacity(bits_per_param=9)


class TestEmbedding:
    """Test payload embedding functionality."""
    
    def test_embed_small_payload(self, mock_model, small_payload):
        """Test embedding a small payload."""
        steg = NeuralSteg(mock_model)
        result = steg.embed(small_payload)
        
        assert result.success == True
        assert result.embedded_bytes > 0
        assert result.capacity_used > 0
        assert result.capacity_used < 1.0
        assert result.checksum is not None
        assert result.processing_time > 0
        assert result.encryption_enabled == False
        assert result.bits_per_param == 4
    
    def test_embed_with_encryption(self, mock_model, small_payload):
        """Test embedding with encryption enabled."""
        password = "encryption_test_123"
        steg = NeuralSteg(mock_model, password=password)
        result = steg.embed(small_payload)
        
        assert result.success == True
        assert result.encryption_enabled == True
    
    def test_embed_different_bits_per_param(self, mock_model, small_payload):
        """Test embedding with different bits per parameter."""
        steg = NeuralSteg(mock_model)
        
        result_2bit = steg.embed(small_payload, bits_per_param=2)
        
        # Reset model for second test (in real scenario, would use backup)
        result_6bit = steg.embed(small_payload, bits_per_param=6)
        
        assert result_2bit.bits_per_param == 2
        assert result_6bit.bits_per_param == 6
        assert result_6bit.capacity_used < result_2bit.capacity_used  # More efficient
    
    def test_embed_nonexistent_payload(self, mock_model):
        """Test embedding nonexistent payload file."""
        steg = NeuralSteg(mock_model)
        
        with pytest.raises(FileNotFoundError):
            steg.embed("/nonexistent/payload.bin")
    
    def test_embed_payload_too_large(self, mock_model, large_payload):
        """Test embedding payload that's too large for model capacity."""
        steg = NeuralSteg(mock_model)
        
        # The mock model is small, so 1MB payload should be too large
        with pytest.raises(PayloadTooLargeError):
            steg.embed(large_payload)


class TestExtraction:
    """Test payload extraction functionality."""
    
    def test_extract_after_embed(self, mock_model, small_payload, temp_dir):
        """Test extracting payload after embedding."""
        steg = NeuralSteg(mock_model)
        
        # First embed
        embed_result = steg.embed(small_payload)
        
        # Then extract
        output_path = temp_dir / "extracted_payload.txt"
        extract_result = steg.extract(output_path)
        
        assert extract_result.success == True
        assert extract_result.extracted_bytes == embed_result.embedded_bytes
        assert extract_result.output_path == output_path
        assert extract_result.checksum_verified == True
        assert output_path.exists()
        
        # Verify content matches
        with open(small_payload, "rb") as f:
            original_data = f.read()
        
        with open(output_path, "rb") as f:
            extracted_data = f.read()
        
        assert extracted_data == original_data
    
    def test_extract_with_encryption(self, mock_model, small_payload, temp_dir):
        """Test extracting encrypted payload."""
        password = "encryption_test_456"
        steg = NeuralSteg(mock_model, password=password)
        
        # Embed with encryption
        steg.embed(small_payload)
        
        # Extract with same password
        output_path = temp_dir / "extracted_encrypted.txt"
        result = steg.extract(output_path)
        
        assert result.success == True
        assert result.checksum_verified == True
        
        # Verify content matches
        with open(small_payload, "rb") as f:
            original_data = f.read()
        
        with open(output_path, "rb") as f:
            extracted_data = f.read()
        
        assert extracted_data == original_data
    
    def test_extract_wrong_password(self, mock_model, small_payload, temp_dir):
        """Test extracting with wrong password fails."""
        # Embed with one password
        steg_embed = NeuralSteg(mock_model, password="correct_password")
        steg_embed.embed(small_payload)
        
        # Try to extract with different password
        steg_extract = NeuralSteg(mock_model, password="wrong_password")
        output_path = temp_dir / "extracted_wrong_password.txt"
        
        with pytest.raises(ExtractionFailedError):
            steg_extract.extract(output_path)
    
    def test_extract_without_embedded_data(self, mock_model, temp_dir):
        """Test extracting from model without embedded data."""
        steg = NeuralSteg(mock_model)
        output_path = temp_dir / "no_data_extracted.bin"
        
        with pytest.raises(ExtractionFailedError):
            steg.extract(output_path)


class TestBackupIntegration:
    """Test backup system integration."""
    
    def test_backup_created_during_embed(self, mock_model, small_payload):
        """Test that backup is created during embedding."""
        steg = NeuralSteg(mock_model)
        
        # Check no backups initially
        initial_backups = steg.list_backups()
        
        # Embed payload (should create backup)
        steg.embed(small_payload)
        
        # Check backup was created
        final_backups = steg.list_backups()
        assert len(final_backups) > len(initial_backups)
    
    def test_manual_backup_creation(self, mock_model):
        """Test manual backup creation."""
        steg = NeuralSteg(mock_model)
        
        backup_path = steg.create_backup("manual_test_backup")
        
        assert backup_path.exists()
        assert "manual_test_backup" in backup_path.name
        
        backups = steg.list_backups()
        backup_names = [b["name"] for b in backups]
        assert "manual_test_backup" in backup_names
    
    def test_verify_integrity(self, mock_model):
        """Test model integrity verification."""
        steg = NeuralSteg(mock_model)
        
        # Model should be valid initially
        assert steg.verify_integrity() == True


class TestDetectionIntegration:
    """Test detection functionality integration."""
    
    def test_detect_anomalies_clean_model(self, mock_model):
        """Test anomaly detection on clean model."""
        steg = NeuralSteg(mock_model)
        
        result = steg.detect_anomalies()
        
        # Clean model should not be suspicious
        assert result.suspicious == False
        assert isinstance(result.statistical_tests, dict)
        assert isinstance(result.recommendations, list)
    
    def test_detect_anomalies_after_embedding(self, mock_model, small_payload):
        """Test anomaly detection after embedding."""
        steg = NeuralSteg(mock_model)
        
        # Embed payload
        steg.embed(small_payload)
        
        # Detect anomalies
        result = steg.detect_anomalies()
        
        # May or may not be suspicious depending on payload size and detection sensitivity
        assert isinstance(result.suspicious, bool)
        assert isinstance(result.entropy_anomaly, bool)