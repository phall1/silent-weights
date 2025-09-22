# Advanced Statistical Camouflage Design Document

## Overview

This design document outlines the implementation of advanced statistical camouflage techniques for the Neural Steganography Toolkit. The current LSB embedding approach, while functionally successful with 3.4MB payload capacity and zero performance impact, produces detectable statistical signatures that can be identified by sophisticated analysis tools.

The advanced camouflage system will implement multiple layers of statistical obfuscation to make embedded payloads indistinguishable from natural parameter variations. The design focuses on preserving parameter distributions, masking entropy signatures, and implementing adaptive embedding patterns that resist detection.

## Architecture

### Simplified Design Approach

The design follows a pragmatic approach: enhance the existing `core.py` and `analysis.py` modules with focused camouflage capabilities rather than creating a complex new subsystem.

```mermaid
graph TB
    A[Enhanced NeuralSteg.embed()] --> B[Statistical Analysis]
    B --> C[Smart Bit Placement]
    C --> D[Calibrated Noise Addition]
    D --> E[Validation & Benchmarking]
    E --> F[Camouflaged Model]
    
    G[Enhanced SteganographyAnalyzer] --> E
```

### Core Components Architecture

Extend existing modules with focused enhancements:

```
src/neuralsteg/
├── core.py                  # Enhanced embed() with camouflage parameter
├── analysis.py              # Enhanced with advanced statistical tests
├── camouflage.py           # NEW: Core camouflage algorithms (single module)
└── benchmarks.py           # NEW: Detection resistance testing
```

**Key Principle**: Add capabilities through focused enhancements rather than architectural complexity.

## Components and Interfaces

### 1. Enhanced NeuralSteg Class (core.py)

**Purpose**: Add camouflage capabilities to existing embed() method with minimal API changes.

**Enhanced Method**:
```python
def embed(self, payload_path: Union[str, Path], bits_per_param: int = 4, 
          camouflage: bool = False, camouflage_strength: float = 0.1) -> EmbedResult:
    """Enhanced embed with optional statistical camouflage."""
```

**Implementation Strategy**:
- Add camouflage logic after basic LSB embedding
- Use existing parameter analysis from current codebase
- Apply focused statistical preservation techniques
- Maintain backward compatibility (camouflage=False by default)

### 2. Statistical Camouflage Module (camouflage.py)

**Purpose**: Core camouflage algorithms in a single, focused module.

**Key Functions** (not classes - keep it simple):
```python
def analyze_parameter_distribution(params: torch.Tensor) -> Dict[str, float]:
    """Analyze statistical properties of parameters."""
    
def preserve_distribution_stats(params: torch.Tensor, target_stats: Dict[str, float]) -> torch.Tensor:
    """Adjust parameters to maintain statistical properties."""
    
def add_calibrated_noise(params: torch.Tensor, noise_strength: float) -> torch.Tensor:
    """Add statistically appropriate noise to mask embedding signatures."""
    
def generate_adaptive_pattern(model_hash: str, param_count: int) -> List[int]:
    """Generate model-specific embedding pattern."""
```

**Core Techniques** (focused on essentials):
- **Distribution Preservation**: Maintain mean/variance of parameter groups
- **Entropy Masking**: Add calibrated noise to normalize entropy
- **Adaptive Patterns**: Use model hash for unique, reproducible patterns

### 3. Enhanced Analysis Module (analysis.py)

**Purpose**: Add advanced statistical tests to existing SteganographyAnalyzer.

**Enhanced Methods**:
```python
def detect_anomalies(self, clean_model_path: Optional[Path] = None, 
                    advanced_tests: bool = False) -> DetectionResult:
    """Enhanced detection with advanced statistical tests."""
    
def run_distribution_tests(self, clean_model_path: Path) -> Dict[str, float]:
    """Run KS test, Anderson-Darling test, etc."""
    
def analyze_entropy_patterns(self, window_size: int = 1000) -> Dict[str, float]:
    """Analyze local entropy patterns for anomalies."""
```

### 4. Detection Benchmarking (benchmarks.py)

**Purpose**: Simple benchmarking framework for validation.

**Key Functions**:
```python
def benchmark_detection_resistance(model_path: Path, clean_reference: Path) -> BenchmarkResult:
    """Run comprehensive detection tests."""
    
def test_statistical_detection(model_path: Path, clean_reference: Path) -> Dict[str, bool]:
    """Test against statistical detection methods."""
    
def generate_benchmark_report(results: BenchmarkResult) -> str:
    """Generate human-readable benchmark report."""
```

## Data Models

### Simplified Data Structures

**Extend existing models rather than creating new hierarchies:**

```python
# Enhance existing EmbedResult in models.py
@dataclass
class EmbedResult:
    # ... existing fields ...
    camouflage_applied: bool = False
    camouflage_metrics: Optional[Dict[str, float]] = None
    detection_test_results: Optional[Dict[str, bool]] = None

# Simple benchmark result
@dataclass  
class BenchmarkResult:
    tests_passed: int
    tests_failed: int
    detection_resistance_score: float  # 0.0 = easily detected, 1.0 = undetectable
    test_details: Dict[str, bool]
    processing_time: float
```

**Key Principle**: Extend existing data models rather than creating complex new hierarchies.

## Error Handling

### Camouflage-Specific Exceptions

```python
class CamouflageError(SteganographyError):
    """Base exception for camouflage operations."""
    pass

class DistributionPreservationError(CamouflageError):
    """Raised when distribution preservation fails."""
    pass

class EntropyMaskingError(CamouflageError):
    """Raised when entropy masking fails."""
    pass

class DetectionValidationError(CamouflageError):
    """Raised when detection validation fails."""
    pass
```

### Error Recovery Strategies

1. **Graceful Degradation**: Fall back to basic LSB if advanced camouflage fails
2. **Partial Camouflage**: Apply available techniques even if some fail
3. **Validation Rollback**: Restore from backup if validation fails
4. **Progressive Enhancement**: Start with basic techniques and add complexity

## Testing Strategy

### Unit Testing Framework

```python
class TestDistributionPreservation:
    def test_gaussian_preservation()
    def test_quantile_matching()
    def test_moment_preservation()
    def test_outlier_avoidance()

class TestEntropyMasking:
    def test_local_entropy_normalization()
    def test_global_entropy_balancing()
    def test_noise_calibration()
    def test_frequency_domain_masking()

class TestAdaptivePatterns:
    def test_model_specific_generation()
    def test_layer_aware_adaptation()
    def test_pattern_uniqueness()
    def test_reproducibility()
```

### Integration Testing

```python
class TestCamouflageIntegration:
    def test_end_to_end_camouflage()
    def test_backward_compatibility()
    def test_performance_preservation()
    def test_payload_integrity()

class TestDetectionResistance:
    def test_statistical_evasion()
    def test_entropy_evasion()
    def test_ml_detection_evasion()
    def test_benchmark_validation()
```

### Benchmarking Framework

```python
class CamouflageBenchmark:
    def benchmark_preservation_accuracy()
    def benchmark_detection_resistance()
    def benchmark_processing_performance()
    def benchmark_capacity_impact()
    def generate_comparison_reports()
```

## Performance Considerations

### Computational Complexity

- **Distribution Analysis**: O(n log n) for sorting-based statistical tests
- **Entropy Masking**: O(n) for noise generation and application
- **Pattern Generation**: O(n) for sequence generation
- **Validation**: O(n) for each detection method tested

### Memory Optimization

- **Streaming Processing**: Process model shards individually to reduce memory usage
- **Lazy Loading**: Load only required tensors for analysis
- **Caching**: Cache distribution profiles and patterns for reuse
- **Batch Processing**: Group operations to minimize memory allocation

### GPU Acceleration

- **Parallel Statistical Tests**: Vectorized operations for distribution analysis
- **Batch Noise Generation**: GPU-accelerated random number generation
- **Tensor Operations**: Native PyTorch GPU operations for parameter modification
- **Concurrent Validation**: Parallel execution of detection tests

## Integration Points

### CLI Enhancement

```bash
# Simple camouflage flag
neuralsteg embed payload.bin model/ --camouflage --camouflage-strength 0.1

# Benchmark detection resistance  
neuralsteg benchmark model/ --clean-reference clean_model/
```

### Python API Extension

```python
# Simple API enhancement
steg = NeuralSteg(model_path, password="secret")
result = steg.embed(payload_path, camouflage=True, camouflage_strength=0.1)

# Benchmark detection resistance
from neuralsteg.benchmarks import benchmark_detection_resistance
benchmark_result = benchmark_detection_resistance(model_path, clean_reference_path)
```

### Backward Compatibility

- All existing API methods remain unchanged
- Camouflage features are opt-in through additional parameters
- Default behavior maintains current LSB embedding
- Configuration files support both basic and advanced settings

## Security Considerations

### Cryptographic Security

- **Key Derivation**: Enhanced PBKDF2 with camouflage-specific salts
- **Entropy Sources**: Cryptographically secure random number generation
- **Pattern Security**: Ensure embedding patterns don't leak information
- **Validation Security**: Prevent detection methods from revealing patterns

### Operational Security

- **Backup Protection**: Secure deletion of intermediate analysis files
- **Memory Clearing**: Clear sensitive data from memory after use
- **Logging Security**: Avoid logging sensitive camouflage parameters
- **Side-Channel Protection**: Minimize timing and memory access patterns

## Deployment Strategy

### Phased Implementation

1. **Phase 1**: Basic statistical camouflage in core.py (distribution preservation + noise)
2. **Phase 2**: Enhanced detection tests in analysis.py (KS test, entropy analysis)
3. **Phase 3**: Benchmarking framework and validation
4. **Phase 4**: CLI integration and documentation

### Testing and Validation

- **Unit Tests**: Comprehensive test coverage for all components
- **Integration Tests**: End-to-end workflow validation
- **Performance Tests**: Benchmarking against current implementation
- **Security Tests**: Validation against known detection methods

### Documentation and Training

- **API Documentation**: Comprehensive documentation for all new features
- **Usage Examples**: Practical examples for different use cases
- **Best Practices**: Guidelines for optimal camouflage configuration
- **Troubleshooting**: Common issues and resolution strategies