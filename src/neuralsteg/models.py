"""
Pydantic data models for the Neural Steganography Toolkit.

Provides type-safe data structures with validation for all toolkit operations.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from pathlib import Path


class EmbedResult(BaseModel):
    """Result of an embedding operation."""
    
    success: bool
    embedded_bytes: int = Field(..., ge=0)
    capacity_used: float = Field(..., ge=0.0, le=1.0)
    checksum: str
    processing_time: float = Field(..., ge=0.0)
    encryption_enabled: bool
    bits_per_param: int = Field(..., ge=1, le=8)
    
    class Config:
        json_encoders = {
            Path: str
        }


class ExtractResult(BaseModel):
    """Result of an extraction operation."""
    
    success: bool
    extracted_bytes: int = Field(..., ge=0)
    output_path: Path
    checksum_verified: bool
    processing_time: float = Field(..., ge=0.0)
    
    class Config:
        json_encoders = {
            Path: str
        }


class CapacityAnalysis(BaseModel):
    """Analysis of model embedding capacity."""
    
    total_target_params: int = Field(..., ge=0)
    capacity_bytes: int = Field(..., ge=0)
    capacity_mb: float = Field(..., ge=0.0)
    bits_per_param: int = Field(..., ge=1, le=8)
    target_tensor_count: int = Field(..., ge=0)


class DetectionResult(BaseModel):
    """Result of steganography detection analysis."""
    
    suspicious: bool
    entropy_anomaly: bool
    statistical_tests: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Result of payload extraction verification."""
    
    payload_match: bool
    payload_checksum_verified: bool
    original_size: int = Field(..., ge=0)
    extracted_size: int = Field(..., ge=0)
    verification_time: float = Field(..., ge=0.0)
    mismatch_details: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            Path: str
        }


class InferenceResult(BaseModel):
    """Result of a single model inference test."""
    
    success: bool
    prompt: str
    response: str
    inference_time: float = Field(..., ge=0.0)
    tokens_generated: int = Field(..., ge=0)
    model_responsive: bool
    error_message: Optional[str] = None


class ComprehensiveTestResult(BaseModel):
    """Result of comprehensive model testing."""
    
    overall_success: bool
    tests_passed: int = Field(..., ge=0)
    tests_failed: int = Field(..., ge=0)
    average_inference_time: float = Field(..., ge=0.0)
    performance_degradation: float = Field(..., ge=0.0)
    gpu_accelerated: bool
    test_results: List[InferenceResult] = Field(default_factory=list)
    summary_stats: Dict[str, float] = Field(default_factory=dict)


class Config(BaseModel):
    """Configuration for steganography operations."""
    
    model_path: Path
    password: Optional[str] = None
    bits_per_param: int = Field(default=4, ge=1, le=8)
    backup_enabled: bool = True
    target_layers: Optional[List[str]] = None
    
    @validator('model_path')
    def model_path_exists(cls, v):
        if not v.exists():
            raise ValueError(f'Model path does not exist: {v}')
        return v
    
    class Config:
        json_encoders = {
            Path: str
        }