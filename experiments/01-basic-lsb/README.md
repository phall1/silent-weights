# Experiment 01: Basic LSB Embedding

## Objective

Demonstrate basic steganographic embedding in neural network parameters using Least Significant Bit (LSB) modification.

## Approach

1. Create a simple neural network (small CNN or MLP)
2. Embed a benign text payload in model parameters via LSB modification
3. Verify payload extraction works correctly
4. Measure impact on model performance

## Success Metrics

- [x] Successful payload embedding and extraction
- [x] Model performance degradation < 5% (measured: 0.00%)
- [x] Embedding survives model save/load cycle
- [x] Clear documentation of embedding capacity (100K+ chars capacity)

## Payload

Simple text string: "Hello from the hidden layer! This is a proof-of-concept for neural network steganography research."

## Files

- `embed.py` - LSB embedding implementation ✅
- `extract.py` - LSB extraction implementation ✅
- `model.py` - Simple neural network definition ✅
- `test_embedding.py` - End-to-end test script ✅
- `requirements.txt` - Dependencies ✅

## Expected Outcomes

- Validate basic LSB technique works in NN parameters
- Establish baseline for embedding capacity
- Foundation for more advanced experiments
