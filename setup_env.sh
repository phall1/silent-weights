#!/bin/bash

# Setup script for LLM steganography research environment
echo "Setting up Python environment for LLM steganography research..."

# Use standard venv approach
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate: source venv/bin/activate"
echo "To deactivate: deactivate"