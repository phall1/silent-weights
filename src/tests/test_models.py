"""
Unit tests for Pydantic data models.

Tests validation, serialization, and data integrity of all model classes.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from neuralsteg.models import (
    EmbedResult, ExtractResult, CapacityAnalysis, 
    DetectionResult, Config
)


class TestEmbedResult:
    """Test cases for EmbedResult model."""
    
    def test_valid_embed_result(self):
        """Test creating valid EmbedResult."""
        result = EmbedResult(
            success=True,
            embedded_bytes=1024,
            capacity_used=0.25,
            checksum="abc123def456",
            processing_time=5.5,
            encryption_enabled=True,
            bits_per_param=4
        )
        
        assert result.success == True
        assert result.embedded_bytes == 1024
        assert result.capacity_used == 0.25
        assert result.checksum == "abc123def456"
        assert result.processing_time == 5.5
        assert result.encryption_enabled == True
        assert result.bits_per_param == 4
    
    def test_invalid_capacity_used(self):
        """Test that invalid capacity_used raises validation error."""
        with pytest.raises(ValidationError):
            EmbedResult(
                success=True,
                embedded_bytes=1024,
                capacity_used=1.5,  # Invalid: > 1.0
                checksum="abc123",
                processing_time=5.5,
                encryption_enabled=False,
                bits_per_param=4
            )
    
    def test_invalid_bits_per_param(self):
        """Test that invalid bits_per_param raises validation error."""
        with pytest.raises(ValidationError):
            EmbedResult(
                success=True,
                embedded_bytes=1024,
                capacity_used=0.5,
                checksum="abc123",
                processing_time=5.5,
                encryption_enabled=False,
                bits_per_param=10  # Invalid: > 8
            )
    
    def test_negative_values(self):
        """Test that negative values raise validation errors."""
        with pytest.raises(ValidationError):
            EmbedResult(
                success=True,
                embedded_bytes=-100,  # Invalid: negative
                capacity_used=0.5,
                checksum="abc123",
                processing_time=5.5,
                encryption_enabled=False,
                bits_per_param=4
            )


class TestExtractResult:
    """Test cases for ExtractResult model."""
    
    def test_valid_extract_result(self):
        """Test creating valid ExtractResult."""
        output_path = Path("/tmp/extracted_file.bin")
        
        result = ExtractResult(
            success=True,
            extracted_bytes=2048,
            output_path=output_path,
            checksum_verified=True,
            processing_time=3.2
        )
        
        assert result.success == True
        assert result.extracted_bytes == 2048
        assert result.output_path == output_path
        assert result.checksum_verified == True
        assert result.processing_time == 3.2
    
    def test_json_serialization(self):
        """Test JSON serialization with Path objects."""
        output_path = Path("/tmp/test.bin")
        
        result = ExtractResult(
            success=True,
            extracted_bytes=1000,
            output_path=output_path,
            checksum_verified=True,
            processing_time=1.0
        )
        
        # Should serialize without error
        json_data = result.json()
        assert "/tmp/test.bin" in json_data


class TestCapacityAnalysis:
    """Test cases for CapacityAnalysis model."""
    
    def test_valid_capacity_analysis(self):
        """Test creating valid CapacityAnalysis."""
        analysis = CapacityAnalysis(
            total_target_params=1000000,
            capacity_bytes=500000,
            capacity_mb=0.48,
            bits_per_param=4,
            target_tensor_count=50
        )
        
        assert analysis.total_target_params == 1000000
        assert analysis.capacity_bytes == 500000
        assert analysis.capacity_mb == 0.48
        assert analysis.bits_per_param == 4
        assert analysis.target_tensor_count == 50
    
    def test_zero_values_allowed(self):
        """Test that zero values are allowed where appropriate."""
        analysis = CapacityAnalysis(
            total_target_params=0,
            capacity_bytes=0,
            capacity_mb=0.0,
            bits_per_param=1,
            target_tensor_count=0
        )
        
        assert analysis.total_target_params == 0
        assert analysis.capacity_bytes == 0


class TestDetectionResult:
    """Test cases for DetectionResult model."""
    
    def test_valid_detection_result(self):
        """Test creating valid DetectionResult."""
        result = DetectionResult(
            suspicious=True,
            entropy_anomaly=False,
            statistical_tests={"entropy": 0.85, "balance": 0.49},
            recommendations=["Check for hidden data", "Run deeper analysis"]
        )
        
        assert result.suspicious == True
        assert result.entropy_anomaly == False
        assert result.statistical_tests["entropy"] == 0.85
        assert len(result.recommendations) == 2
    
    def test_default_values(self):
        """Test default values for optional fields."""
        result = DetectionResult(
            suspicious=False,
            entropy_anomaly=False
        )
        
        assert result.statistical_tests == {}
        assert result.recommendations == []


class TestConfig:
    """Test cases for Config model."""
    
    def test_valid_config(self, tmp_path):
        """Test creating valid Config."""
        # Create a temporary model directory
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        config = Config(
            model_path=model_path,
            password="secret123",
            bits_per_param=6,
            backup_enabled=False,
            target_layers=["layer1", "layer2"]
        )
        
        assert config.model_path == model_path
        assert config.password == "secret123"
        assert config.bits_per_param == 6
        assert config.backup_enabled == False
        assert config.target_layers == ["layer1", "layer2"]
    
    def test_nonexistent_model_path(self):
        """Test that nonexistent model path raises validation error."""
        with pytest.raises(ValidationError):
            Config(model_path=Path("/nonexistent/path"))
    
    def test_default_values(self, tmp_path):
        """Test default values."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        config = Config(model_path=model_path)
        
        assert config.password is None
        assert config.bits_per_param == 4
        assert config.backup_enabled == True
        assert config.target_layers is None
    
    def test_invalid_bits_per_param(self, tmp_path):
        """Test invalid bits_per_param values."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        with pytest.raises(ValidationError):
            Config(model_path=model_path, bits_per_param=0)
        
        with pytest.raises(ValidationError):
            Config(model_path=model_path, bits_per_param=9)