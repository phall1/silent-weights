#!/usr/bin/env python3
"""
🎵 Rick Roll Demo - Neural Steganography Toolkit

Simple demo that embeds a Rick Roll MP3 into a neural network model.
Usage: python rickroll_demo.py [model_path] [payload_path]
"""

import sys
from pathlib import Path
from neuralsteg import NeuralSteg

def run_demo(model_path, payload_path, output_path="extracted_rickroll.mp3"):
    """Run the Rick Roll steganography demo with specified paths."""
    model_path = Path(model_path)
    payload_path = Path(payload_path)
    output_path = Path(output_path)
    
    print("🎵 Rick Roll Neural Steganography Demo")
    print(f"📁 Model: {model_path}")
    print(f"🎵 Payload: {payload_path}")
    print(f"📤 Output: {output_path}")
    print()
    
    # Initialize toolkit
    password = "rickroll2024"
    steg = NeuralSteg(model_path, password=password)
    
    # Embed the payload
    print("🎯 Embedding Rick Roll MP3...")
    embed_result = steg.embed(payload_path, bits_per_param=4)
    print(f"✅ Embedded {embed_result.embedded_bytes:,} bytes in {embed_result.processing_time:.1f}s")
    
    # Extract the payload
    print("📤 Extracting Rick Roll MP3...")
    extract_result = steg.extract(output_path, bits_per_param=4)
    print(f"✅ Extracted {extract_result.extracted_bytes:,} bytes in {extract_result.processing_time:.1f}s")
    
    # Verify
    verification = steg.verify_extraction(payload_path, output_path)
    if verification.payload_match:
        print("🎉 SUCCESS! Rick Roll perfectly hidden and recovered!")
        print(f"🎧 Play with: open {output_path}")
    else:
        print("❌ Verification failed!")
    
    return verification.payload_match

if __name__ == "__main__":
    if len(sys.argv) < 3 or "--help" in sys.argv or "-h" in sys.argv:
        print("🎵 Rick Roll Neural Steganography Demo")
        print()
        print("Usage: python rickroll_demo.py <model_path> <payload_path> [output_path]")
        print()
        print("Examples:")
        print("  python rickroll_demo.py models/llama-3.2-3b-original payloads/rick-roll.mp3")
        print("  python rickroll_demo.py ./my-model ./rickroll.mp3 ./extracted.mp3")
        print()
        sys.exit(0)
    
    model_path = sys.argv[1]
    payload_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "extracted_rickroll.mp3"
    
    try:
        success = run_demo(model_path, payload_path, output_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)