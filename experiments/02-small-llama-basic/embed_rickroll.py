"""
Embed rick-roll MP3 into LLaMA model weights.

This script embeds the rick-roll.mp3 file into the modified LLaMA model
using 4-bit LSB steganography. The original model remains untouched.
"""

from pathlib import Path
from llama_steganography import LLaMASteganography


def embed_rickroll():
    """Embed rick-roll MP3 into LLaMA model."""

    # Paths
    model_path = "../../models/llama-3.2-3b-modified"
    payload_path = "../../payloads/rick-roll.mp3"

    print("=== LLaMA Rick-Roll Embedding ===")
    print(f"Model: {model_path}")
    print(f"Payload: {payload_path}")

    # Check files exist
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Run setup.py first to create the model copy")
        return False

    if not Path(payload_path).exists():
        print(f"❌ Payload not found: {payload_path}")
        return False

    # Initialize steganography
    steg = LLaMASteganography(model_path)

    # Analyze capacity first
    print("\n📊 Analyzing model capacity...")
    capacity = steg.analyze_capacity()
    print(f"   Target parameters: {capacity['total_target_params']:,}")
    print(f"   Embedding capacity: {capacity['capacity_mb']:.1f} MB")
    print(f"   Target tensors: {capacity['target_tensor_count']}")

    # Get payload size
    payload_size = Path(payload_path).stat().st_size
    payload_mb = payload_size / (1024 * 1024)

    print(f"\n🎵 Rick-roll MP3 size: {payload_mb:.2f} MB ({payload_size:,} bytes)")
    print(
        f"   Capacity utilization: {(payload_mb / capacity['capacity_mb']) * 100:.1f}%"
    )

    if payload_size * 8 > capacity["capacity_bits"]:
        print("❌ Payload too large for embedding capacity!")
        return False

    # Perform embedding
    print(f"\n🔄 Embedding with 4-bit LSB modification...")
    try:
        result = steg.embed_file(payload_path, bits_per_param=4)

        print("✅ Embedding successful!")
        print(f"   Embedded bits: {result['embedded_bits']:,}")
        print(f"   Capacity used: {result['capacity_utilization']*100:.2f}%")
        print(f"   Modified shards: {len(result['modified_shards'])}")
        print(f"   Checksum: {result['checksum'][:16]}...")

        # Verify model structure
        print(f"\n🔍 Verifying model integrity...")
        integrity = steg.verify_model_integrity()
        if integrity["structure_intact"]:
            print("✅ Model structure intact")
        else:
            print(f"❌ Model structure issues: {integrity['missing_files']}")
            return False

        print(f"\n🎉 Rick-roll successfully hidden in LLaMA weights!")
        print(f"   The model should still work normally for chat")
        print(f"   Use extract_rickroll.py to recover the hidden MP3")

        return True

    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return False


if __name__ == "__main__":
    success = embed_rickroll()
    if not success:
        exit(1)

    print(f"\n💡 Next steps:")
    print(f"   1. Test model: python simple_chat.py")
    print(f"   2. Extract MP3: python extract_rickroll.py")
    print(f"   3. Reset model: python reset_model.py")
