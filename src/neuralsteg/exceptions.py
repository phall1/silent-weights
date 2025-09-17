"""
Custom exceptions for the Neural Steganography Toolkit.

Provides specific exception types for different error conditions to enable
better error handling and user feedback.
"""


class SteganographyError(Exception):
    """Base exception for all steganography operations."""
    pass


class ModelLoadError(SteganographyError):
    """Failed to load or parse model."""
    pass


class PayloadTooLargeError(SteganographyError):
    """Payload exceeds model capacity."""
    pass


class CorruptionDetectedError(SteganographyError):
    """Model corruption detected during operation."""
    pass


class ExtractionFailedError(SteganographyError):
    """Failed to extract payload from model."""
    pass


class IntegrityCheckFailedError(SteganographyError):
    """Payload integrity verification failed."""
    pass


class EncryptionError(SteganographyError):
    """Encryption or decryption operation failed."""
    pass


class ConfigurationError(SteganographyError):
    """Invalid configuration or parameters."""
    pass