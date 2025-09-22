# Neural Steganography Toolkit

A research toolkit for embedding and extracting arbitrary payloads from neural network models, designed for AI supply chain security research and vulnerability analysis.

## Purpose

This project demonstrates steganographic embedding techniques in Large Language Models to research supply chain security vulnerabilities. The toolkit enables security researchers to:

- Embed binary payloads (files, malware samples) into neural network weights
- Extract hidden data from potentially compromised models  
- Analyze model capacity for steganographic storage
- Detect potential modifications in neural networks

## Research Focus

The project addresses three key threat scenarios:
1. **Supply Chain Backdoors** - Persistence of embedded payloads in open-source models
2. **Marketplace Trojans** - Covert payload distribution through model hubs
3. **Gradient Exfiltration** - Data theft during fine-tuning processes

## Responsible Use

This is a defensive security research tool intended for:
- Academic research and publication
- Red team exercises and vulnerability assessment
- Detection method development
- AI supply chain security awareness

The toolkit uses benign payloads and focuses on detection/mitigation strategies rather than offensive capabilities.