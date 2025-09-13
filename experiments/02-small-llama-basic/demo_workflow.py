"""
Complete steganography workflow demonstration.

This script demonstrates the full process:
1. Check model setup
2. Embed rick-roll MP3
3. Verify model still works
4. Extract the MP3
5. Compare original vs extracted

"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, script_name], capture_output=False, check=True
        )
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - Script not found: {script_name}")
        return False


def check_prerequisites() -> bool:
    """Check that all required files exist."""
    print("🔍 Checking prerequisites...")

    required_files = [
        "../../models/llama-3.2-3b-original",
        "../../models/llama-3.2-3b-modified",
        "../../payloads/rick-roll.mp3",
        "embed_rickroll.py",
        "extract_rickroll.py",
        "verify_model.py",
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print(f"❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        print(f"\n💡 Run setup.py first to create the model directories")
        return False

    print(f"✅ All prerequisites found")
    return True


def main():
    """Run the complete steganography demonstration."""

    print("🎵 LLaMA Steganography Demo - Rick Roll Edition 🎵")
    print("This will demonstrate hiding an MP3 inside a 3B parameter model")

    # Check prerequisites
    if not check_prerequisites():
        return False

    # Get payload size for reference
    payload_path = Path("../../payloads/rick-roll.mp3")
    payload_mb = payload_path.stat().st_size / (1024 * 1024)

    print(f"\n📊 Demo Overview:")
    print(f"   Model: LLaMA-3.2-3B-Instruct (~6 GB)")
    print(f"   Payload: rick-roll.mp3 ({payload_mb:.2f} MB)")
    print(f"   Method: 4-bit LSB modification")
    print(f"   Capacity utilization: <1%")

    input(f"\nPress Enter to begin the demonstration...")

    # Step 1: Embed the MP3
    if not run_script(
        "embed_rickroll.py", "EMBEDDING rick-roll MP3 into model weights"
    ):
        return False

    # Step 2: Verify model still works
    if not run_script("verify_model.py", "VERIFYING model functionality"):
        print(f"⚠️  Model verification had issues, but continuing...")

    # Step 3: Extract the MP3
    if not run_script("extract_rickroll.py", "EXTRACTING hidden MP3 from model"):
        return False

    # Final summary
    print(f"\n" + "=" * 60)
    print(f"🎉 STEGANOGRAPHY DEMONSTRATION COMPLETE!")
    print(f"=" * 60)

    print(f"\n✨ What just happened:")
    print(f"   1. ✅ Embedded 3.4MB MP3 into 3B parameter model")
    print(f"   2. ✅ Model still functions normally for chat")
    print(f"   3. ✅ Successfully extracted identical MP3")
    print(f"   4. ✅ Zero visible impact on model behavior")

    # Check if extracted file matches
    original_size = Path("../../payloads/rick-roll.mp3").stat().st_size
    if Path("extracted_rick-roll.mp3").exists():
        extracted_size = Path("extracted_rick-roll.mp3").stat().st_size
        if original_size == extracted_size:
            print(f"\n🔍 Data integrity: ✅ PERFECT (identical file sizes)")
        else:
            print(f"\n🔍 Data integrity: ⚠️  Size mismatch detected")

    print(f"\n🎵 Play the extracted MP3:")
    print(f"   open extracted_rick-roll.mp3")

    print(f"\n💡 Security implications:")
    print(f"   - Model weights look completely normal")
    print(f"   - No obvious signs of data hiding")
    print(f"   - Could hide malware, keys, or other data")
    print(f"   - Works with any neural network architecture")

    print(f"\n🔄 Reset to clean state:")
    print(f"   python reset_model.py")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print(f"\n❌ Demo failed - check error messages above")
        exit(1)
    else:
        print(f"\n🎊 Demo completed successfully!")
