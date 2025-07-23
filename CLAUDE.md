# Claude Development Guidelines

## Python Code Standards
- **Always use type hints** - every function, every variable where helpful
- **Pydantic for data validation** - use BaseModel for structured data
- **Concise docstrings** - one line for simple functions, brief descriptions for complex ones
- **Clean, pythonic code** following Zen of Python
- **Explicit over implicit** - clear variable names, obvious logic flow

## Docstring Style (Concise)
```python
def embed_lsb(data: bytes, target: torch.Tensor) -> torch.Tensor:
    """Embed data into tensor using LSB modification."""
    pass

# Only elaborate when truly complex
def complex_algorithm(params: ComplexParams) -> ProcessedResult:
    """
    Process neural network steganography using multi-stage LSB embedding.
    
    Handles parameter selection, bit-level operations, and integrity validation.
    """
    pass
```

## Preferred Libraries
- **Pydantic** for data models and validation
- **PyTorch** for neural networks
- **Type hints** from typing module
- **Structured configuration** over scattered parameters

## Research Code Specific
- Set seeds for reproducibility
- Modular, reusable utilities in `src/` 
- Clear experiment structure
- Performance measurements included