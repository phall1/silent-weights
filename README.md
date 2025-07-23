# LLM Steganography Research

Research into steganographic embedding in Large Language Models for supply chain security analysis.

## Project Structure

```
├── docs/                   # Research documentation
│   ├── ROADMAP.md         # High-level research plan
│   └── PUBLICATION_DRAFT.md # Academic paper draft
├── papers/                # Academic references
│   ├── REFERENCES.md      # Organized paper list with links
│   └── pdfs/              # Downloaded papers
├── experiments/           # Individual experiments
│   └── 01-basic-lsb/      # Basic LSB embedding proof-of-concept
├── src/                   # Reusable steganography utilities
├── models/                # Model artifacts (clean + embedded)
└── requirements.txt       # Python dependencies
```

## Research Objectives

1. **Supply Chain Backdoors** - Demonstrate embedding persistence in open-source models
2. **Marketplace Trojans** - Show covert payload distribution through model hubs  
3. **Gradient Exfiltration** - Prove data theft during fine-tuning processes

## Getting Started

1. Review the [research roadmap](docs/ROADMAP.md)
2. Check [paper references](papers/REFERENCES.md) for background literature
3. Explore experiments starting with `experiments/01-basic-lsb/`

## Responsible Disclosure

This research is conducted for defensive security purposes to raise awareness of AI supply chain vulnerabilities. All demonstrations use benign payloads and focus on detection/mitigation strategies.