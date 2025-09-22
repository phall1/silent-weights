# Design Document

## Overview

The Neural Steganography Toolkit is a comprehensive framework that consolidates research findings from previous experiments into a production-quality software package. The design balances simplicity with extensibility, providing both a clean API for researchers and advanced capabilities for security professionals.

The toolkit builds on proven steganography techniques from Experiment 2, expanding to support multiple neural network architectures, advanced embedding strategies, encryption, evasion techniques, and comprehensive detection analysis. The architecture supports both immediate usability for single researchers and scalability for enterprise security assessments.

## Architecture

### Simplified Architecture

```
┌─────────────────────────────────────────┐
│            CLI Tool                     │
├─────────────────────────────────────────┤
│         Python Package                 │
│  ┌─────────────────────────────────────┐│
│  │  NeuralSteg (main class)           ││
│  │  - embed()                         ││
│  │  - extract()                       ││
│  │  - analyze()                       ││
│  │  - detect()                        ││
│  └─────────────────────────────────────┘│
├─────────────────────────────────────────┤
│    Existing LLaMA Code (refactored)    │
└─────────────────────────────────────────┘
```

### Core Components

#### 1. NeuralSteg Class
Single main class that handles all operations:
- Builds on existing `LLaMASteganography` class
- Adds encryption/decryption capabilities
- Includes basic statistical analysis
- Simple, documented API

#### 2. Encryption Module
- AES-256 encryption with password-based key derivation
- Integrated into existing payload preparation
- Minimal dependencies (use Python's `cryptography` library)

#### 3. Analysis Tools
- Basic entropy analysis
- Parameter distribution comparison
- Simple detection metrics
- Built into the main class, not separate modules

#### 4. Model Verification Module
- Payload extraction verification
- Quick inference testing with single prompts
- Batch inference testing with configurable test suites
- Performance benchmarking (pre/post embedding)
- GPU-accelerated comprehensive testing

## Components and Interfaces

### Main Interface (Simple)

```python
class NeuralSteg:
    def __init__(self, model_path: str, password: str = None):
        """Initialize with model path and optional encryption password"""
        
    def embed(self, payload_path: str, bits_per_param: int = 4) -> dict:
        """Embed file into model weights"""
        
    def extract(self, output_path: str) -> dict:
        """Extract hidden file from model weights"""
        
    def analyze_capacity(self) -> dict:
        """Analyze embedding capacity of the model"""
        
    def detect_anomalies(self, clean_model_path: str = None) -> dict:
        """Basic detection analysis"""
        
    def verify_extraction(self, original_payload_path: str, extracted_payload_path: str) -> dict:
        """Verify extracted payload matches original"""
        
    def test_inference(self, prompt: str, max_tokens: int = 50) -> dict:
        """Quick inference test with a single prompt"""
        
    def comprehensive_test(self, test_prompts: list = None, use_gpu: bool = False) -> dict:
        """Comprehensive model functionality testing"""
```

### Usage Examples

```python
# Basic usage
steg = NeuralSteg("path/to/model")
result = steg.embed("secret.pdf")
extracted = steg.extract("recovered.pdf")

# With encryption
steg = NeuralSteg("path/to/model", password="research123")
result = steg.embed("malware.exe", bits_per_param=2)

# Analysis
capacity = steg.analyze_capacity()
anomalies = steg.detect_anomalies("clean_model_path")
```

## Data Models

### Data Models (Pydantic)

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from pathlib import Path

class EmbedResult(BaseModel):
    success: bool
    embedded_bytes: int
    capacity_used: float = Field(..., ge=0.0, le=1.0)
    checksum: str
    processing_time: float
    encryption_enabled: bool
    bits_per_param: int = Field(..., ge=1, le=8)

class ExtractResult(BaseModel):
    success: bool
    extracted_bytes: int
    output_path: Path
    checksum_verified: bool
    processing_time: float
    
class CapacityAnalysis(BaseModel):
    total_target_params: int
    capacity_bytes: int
    capacity_mb: float
    bits_per_param: int
    target_tensor_count: int
    
class DetectionResult(BaseModel):
    suspicious: bool
    entropy_anomaly: bool
    statistical_tests: Dict[str, float]
    recommendations: List[str]

class VerificationResult(BaseModel):
    payload_match: bool
    payload_checksum_verified: bool
    original_size: int
    extracted_size: int
    verification_time: float

class InferenceResult(BaseModel):
    success: bool
    prompt: str
    response: str
    inference_time: float
    tokens_generated: int
    model_responsive: bool

class ComprehensiveTestResult(BaseModel):
    overall_success: bool
    tests_passed: int
    tests_failed: int
    average_inference_time: float
    performance_degradation: float = Field(..., ge=0.0)
    gpu_accelerated: bool
    test_results: List[InferenceResult]
    
class Config(BaseModel):
    model_path: Path
    password: Optional[str] = None
    bits_per_param: int = Field(default=4, ge=1, le=8)
    backup_enabled: bool = True
    
    @validator('model_path')
    def model_path_exists(cls, v):
        if not v.exists():
            raise ValueError(f'Model path does not exist: {v}')
        return v
```

## Error Handling

### Exception Hierarchy

```python
class SteganographyError(Exception):
    """Base exception for all steganography operations"""

class ModelLoadError(SteganographyError):
    """Failed to load or parse model"""

class PayloadTooLargeError(SteganographyError):
    """Payload exceeds model capacity"""

class CorruptionDetectedError(SteganographyError):
    """Model corruption detected during operation"""

class ExtractionFailedError(SteganographyError):
    """Failed to extract payload from model"""

class IntegrityCheckFailedError(SteganographyError):
    """Payload integrity verification failed"""
```

### Error Recovery

- **Automatic Backups**: All operations create backups before modification
- **Rollback Capability**: Failed operations can be rolled back
- **Graceful Degradation**: Partial failures don't crash the entire operation
- **Detailed Logging**: All errors include context and recovery suggestions

## Testing Strategy

### Unit Testing
- **Component Isolation**: Each module tested independently
- **Mock Dependencies**: External dependencies mocked for reliable testing
- **Edge Cases**: Boundary conditions and error scenarios covered
- **Performance Tests**: Memory usage and processing time benchmarks

### Integration Testing
- **End-to-End Workflows**: Complete embed/extract cycles tested
- **Model Compatibility**: Testing across different model architectures
- **Cross-Platform**: Testing on macOS (M1) and Linux environments
- **Large Model Testing**: Scalability testing with multi-GB models

### Security Testing
- **Encryption Validation**: Verify encryption/decryption correctness
- **Integrity Checks**: Ensure payload integrity across operations
- **Evasion Effectiveness**: Test against known detection methods
- **Attack Simulation**: Red team testing of the toolkit itself

## Performance Considerations

### M1 Mac Optimization
- **Metal Performance Shaders**: Leverage M1 GPU for tensor operations
- **Memory Efficiency**: Streaming processing for large models
- **Native ARM64**: Optimized for Apple Silicon architecture
- **Unified Memory**: Take advantage of shared CPU/GPU memory

### Scalability Design
- **Chunked Processing**: Process large models in manageable chunks
- **Lazy Loading**: Load model components on-demand
- **Parallel Processing**: Multi-threading for independent operations
- **GPU Acceleration**: Ready for CUDA/ROCm when available

### Memory Management
- **Streaming I/O**: Process models without loading entirely into memory
- **Garbage Collection**: Explicit cleanup of large tensors
- **Memory Monitoring**: Track and report memory usage
- **Swap Handling**: Graceful handling of memory pressure

## Security Considerations

### Operational Security
- **Secure Key Management**: Proper key derivation and storage
- **Temporary File Cleanup**: Secure deletion of intermediate files
- **Process Isolation**: Sandboxing for untrusted model processing
- **Audit Logging**: Comprehensive logging for security analysis

### Research Ethics
- **Benign Payloads**: Default to harmless test payloads
- **Clear Documentation**: Explicit warnings about malicious use
- **Responsible Disclosure**: Guidelines for reporting vulnerabilities
- **Academic Use**: Designed for legitimate security research

## Package Structure (Minimal)

```
src/
├── neuralsteg/
│   ├── __init__.py          # Main NeuralSteg class
│   ├── core.py              # Refactored LLaMA steganography code
│   ├── crypto.py            # AES encryption/decryption
│   ├── analysis.py          # Basic detection and analysis
│   └── cli.py               # Simple command-line interface
├── setup.py                 # Package setup
└── requirements.txt         # Dependencies
```

### Dependencies (Minimal)
- `torch` (already required)
- `safetensors` (already required)
- `cryptography` (for AES encryption)
- `click` (for CLI)
- `numpy` (for analysis)
- `pydantic` (for data validation and serialization)

### Installation
```bash
cd src
pip install -e .
```

### CLI Usage
```bash
# Embed file
neuralsteg embed model_path payload_path --password secret123

# Extract file
neuralsteg extract model_path output_path --password secret123

# Analyze capacity
neuralsteg analyze model_path

# Basic detection
neuralsteg detect model_path --clean-model clean_path

# Verify extraction
neuralsteg verify model_path original_payload extracted_payload

# Test model inference
neuralsteg test model_path --prompt "Hello, how are you?"

# Comprehensive benchmark
neuralsteg benchmark model_path --use-gpu --test-suite default
```