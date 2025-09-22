# Neural Steganography Research Toolkit

A comprehensive research toolkit for embedding and extracting arbitrary payloads from neural network models, designed for AI supply chain security research and vulnerability analysis.

## Project Overview

This project demonstrates practical steganographic embedding techniques in Large Language Models to research supply chain security vulnerabilities. The toolkit enables security researchers to embed binary payloads, extract hidden data, and analyze model capacity for steganographic storage.

## Project Structure

```
├── src/                    # Neural Steganography Toolkit (Production-ready)
│   ├── neuralsteg/        # Core steganography library
│   ├── tests/             # Comprehensive test suite
│   └── README.md          # Toolkit documentation
├── experiments/           # Research experiments
│   ├── 01-basic-lsb/      # ✅ Basic LSB embedding (COMPLETED)
│   └── 02-small-llama-basic/ # ✅ LLaMA steganography (COMPLETED)
├── docs/                  # Research documentation
│   ├── research_summary.md # Current research status
│   ├── ROADMAP.md         # Research plan and future work
│   ├── ONE_PAGER.md       # Executive summary
│   └── PUBLICATION_DRAFT.md # Academic paper draft
├── papers/                # Academic references (25+ papers)
│   ├── REFERENCES.md      # Organized bibliography
│   └── pdfs/              # Downloaded research papers
├── models/                # Model artifacts and backups
├── payloads/              # Test payloads for experiments
└── rickroll_demo.py       # 🎵 Fun demo script
```

## Key Achievements

### ✅ Completed Research
- **Basic LSB Embedding**: Successfully demonstrated in simple neural networks
- **LLaMA Steganography**: Practical embedding in production language models
- **Production Toolkit**: Full-featured CLI and Python API
- **Encryption Support**: AES-256 encrypted payload embedding
- **Detection Methods**: Statistical analysis and anomaly detection
- **Backup System**: Automatic model backup and restoration

### 🎯 Current Capabilities
- **Universal Model Support**: LLaMA, GPT, BERT, and other transformer models
- **Large Payload Capacity**: Multi-megabyte embedding capacity in large models
- **Stealth Techniques**: Statistical camouflage and LSB modification
- **Integrity Verification**: Comprehensive payload and model verification
- **Research Tools**: Analysis, detection, and forensic capabilities

## Quick Start

### Installation
```bash
cd src
pip install -e .
```

### Basic Usage
```bash
# Embed a file into a model
neuralsteg embed ./models/llama-7b ./secret.pdf --password mypass

# Extract the hidden file
neuralsteg extract ./models/llama-7b ./recovered.pdf --password mypass

# Analyze model capacity
neuralsteg analyze ./models/llama-7b
```

### Python API
```python
from neuralsteg import NeuralSteg

steg = NeuralSteg("./models/llama-7b", password="secret")
result = steg.embed("confidential.zip")
print(f"Embedded {result.embedded_bytes:,} bytes")
```

## Research Objectives

1. **Supply Chain Backdoors** - ✅ Demonstrated embedding persistence in open-source models
2. **Marketplace Trojans** - ✅ Showed covert payload distribution through model modifications
3. **Detection Evasion** - 🚧 Advanced statistical camouflage techniques (v0.2.0 in development)
4. **Fine-tuning Resilience** - 🔮 Payload survival through model fine-tuning (planned)

## Research Applications

This toolkit addresses three key threat scenarios:
- **Supply Chain Attacks**: Persistence of embedded payloads in open-source models
- **Model Hub Trojans**: Covert payload distribution through model repositories
- **Data Exfiltration**: Steganographic data theft during model training/fine-tuning

## Documentation

- **[Project Status](docs/PROJECT_STATUS.md)** - Comprehensive status overview and achievements
- **[Toolkit Documentation](src/README.md)** - Complete API reference and examples
- **[Research Summary](docs/research_summary.md)** - Current findings and progress
- **[Research Roadmap](docs/ROADMAP.md)** - Future experiments and objectives
- **[Academic References](papers/REFERENCES.md)** - 25+ research papers and citations

## Responsible Use

This is a defensive security research tool intended for:
- ✅ Academic research and publication
- ✅ Red team exercises and vulnerability assessment  
- ✅ Detection method development
- ✅ AI supply chain security awareness

The toolkit uses benign payloads and focuses on detection/mitigation strategies rather than offensive capabilities.