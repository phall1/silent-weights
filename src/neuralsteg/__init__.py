"""
Neural Steganography Toolkit

A research toolkit for embedding and extracting arbitrary payloads from neural network models.
Designed for security research and AI supply chain vulnerability analysis.
"""

from .core import NeuralSteg
from .models import (
    EmbedResult, ExtractResult, CapacityAnalysis, DetectionResult,
    VerificationResult, InferenceResult, ComprehensiveTestResult
)

__version__ = "0.1.0"
__author__ = "AI Security Research"

__all__ = [
    "NeuralSteg",
    "EmbedResult", 
    "ExtractResult",
    "CapacityAnalysis",
    "DetectionResult",
    "VerificationResult",
    "InferenceResult", 
    "ComprehensiveTestResult"
]