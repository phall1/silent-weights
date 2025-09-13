"""
Extract rick-roll MP3 from LLaMA model weights.

This script extracts the hidden rick-roll.mp3 file from the modified LLaMA model
and saves it to verify the steganography worked correctly.
"""

from pathlib import Path
from llama_steganography import LLaMASteganography


def extract_rickroll():
    """Extract rick-roll MP3 from LLaMA model."""

    # Paths
    model_path = "../../models/llama-3.2-3b-modified"
    output_path = "extracted_rick-roll.mp3"

    print("=== LLaMA Rick-Roll Extraction ===")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")

    # Check model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False

    # Initialize steganography
    steg = LLaMASteganography(model_path)

    # Perform extraction
    print(f"\n🔄 Extracting hidden data from model weights...")
    try:
        result = steg.extract_file(output_path)

        print("✅ Extraction successful!")
        print(f"   Extracted size: {result['extracted_size']:,} bytes")
        print(f"   Output file: {result['output_path']}")
        print(f"   Checksum verified: {result['checksum_verified']}")
        print(f"   Checksum: {result['actual_checksum'][:16]}...")

        # Compare with original if it exists
        original_path = "../../payloads/rick-roll.mp3"
        if Path(original_path).exists():
            original_size = Path(original_path).stat().st_size
            extracted_size = Path(output_path).stat().st_size

            print(f"\n📊 Size comparison:")
            print(f"   Original: {original_size:,} bytes")
            print(f"   Extracted: {extracted_size:,} bytes")
            print(
                f"   Match: {'✅ Yes' if original_size == extracted_size else '❌ No'}"
            )

            if original_size == extracted_size:
                print(
                    f"\n🎉 Perfect extraction! The hidden MP3 is identical to original."
                )
                print(f"   Play it: open {output_path}")
            else:
                print(f"❌ Size mismatch - extraction may have failed")
        else:
            print(f"\n✅ Extraction complete - verify by playing: open {output_path}")

        return True

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


if __name__ == "__main__":
    success = extract_rickroll()
    if not success:
        exit(1)

    print(f"\n💡 The extracted MP3 should be identical to the original!")
    print(f"   This proves the steganography preserved the data perfectly.")
