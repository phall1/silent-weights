# LLM Steganography Research Roadmap

## Research Objectives
1. **Supply Chain Backdoors** - Demonstrate embedding persistence in open-source models
2. **Marketplace Trojans** - Show covert payload distribution through model hubs  
3. **Fine tuning resilience** - Demonstrate persistence through fine tuning 

## High-Level Technical Phases

### Phase 1: Foundation & Literature (done)
- Survey existing neural steganography research
- Choose embedding techniques to investigate
- Select target model architectures
- Define benign proof-of-concept payload
- Establish baseline metrics

### Phase 2: Embedding Implementation   (in progress)
- Implement steganographic encoding methods
- Test payload embedding in model parameters
- Measure embedding capacity vs. model performance
- Develop extraction mechanisms
- Validate robustness across model operations

### Phase 3: Supply Chain Demonstration
- Test persistence through common fine-tuning workflows
- Evaluate survival across quantization/pruning
- Document degradation patterns
- Develop mitigation strategies

### Phase 4: Marketplace Distribution
- Create realistic model distribution scenario
- Implement trigger-based activation
- Test detection evasion
- Document threat model

### Phase 6: Detection & Defense
- Develop detection methodologies
- Create defensive tools/techniques
- Benchmark against existing security measures
- Propose countermeasures

## Success Metrics
- Embedding capacity (bits per parameter)
- Persistence rate across operations
- Detection evasion effectiveness
- Real-world applicability demonstration