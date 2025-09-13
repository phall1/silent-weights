"""
Reset the modified LLaMA model back to original state.

This script copies the pristine original model back to the modified directory,
effectively removing any steganographic modifications.
"""

import shutil
from pathlib import Path


def reset_to_original():
    """Reset modified model to original pristine state."""

    original_path = Path("../../models/llama-3.2-3b-original")
    modified_path = Path("../../models/llama-3.2-3b-modified")

    print("=== LLaMA Model Reset ===")
    print(f"Original: {original_path}")
    print(f"Modified: {modified_path}")

    # Check original exists
    if not original_path.exists():
        print(f"❌ Original model not found: {original_path}")
        print("Run setup.py first to create the original model copy")
        return False

    # Backup check - warn if modified directory will be deleted
    if modified_path.exists():
        print(f"\n⚠️  This will completely replace the modified model")
        print(f"   Any embedded data will be permanently lost")

        response = input(f"   Continue? [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            print("❌ Reset cancelled")
            return False

    try:
        # Remove modified directory if it exists
        if modified_path.exists():
            print(f"\n🗑️  Removing modified model...")
            shutil.rmtree(modified_path)

        # Copy original to modified
        print(f"📋 Copying original model to modified directory...")
        shutil.copytree(original_path, modified_path)

        print(f"✅ Reset successful!")
        print(f"   Modified model is now identical to original")
        print(f"   All steganographic modifications removed")

        # Verify the reset worked
        print(f"\n🔍 Verifying reset...")

        # Check key files exist
        required_files = [
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]

        missing_files = []
        for file in required_files:
            if not (modified_path / file).exists():
                missing_files.append(file)

        if missing_files:
            print(f"❌ Reset verification failed - missing files: {missing_files}")
            return False

        print(f"✅ Reset verification passed")
        print(f"   All required files present in modified directory")

        return True

    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False


def get_model_status():
    """Check status of original and modified model directories."""

    original_path = Path("../../models/llama-3.2-3b-original")
    modified_path = Path("../../models/llama-3.2-3b-modified")

    print("=== Model Status ===")

    # Check original
    if original_path.exists():
        size_mb = sum(
            f.stat().st_size for f in original_path.rglob("*") if f.is_file()
        ) / (1024 * 1024)
        print(f"📁 Original: ✅ Present ({size_mb:.0f} MB)")
    else:
        print(f"📁 Original: ❌ Missing")

    # Check modified
    if modified_path.exists():
        size_mb = sum(
            f.stat().st_size for f in modified_path.rglob("*") if f.is_file()
        ) / (1024 * 1024)
        print(f"📁 Modified: ✅ Present ({size_mb:.0f} MB)")
    else:
        print(f"📁 Modified: ❌ Missing")


if __name__ == "__main__":
    print("Current status:")
    get_model_status()

    print("\n" + "=" * 50)
    success = reset_to_original()

    if success:
        print("\nNew status:")
        get_model_status()

        print(f"\n💡 Next steps:")
        print(f"   1. Test model: python simple_chat.py")
        print(f"   2. Embed again: python embed_rickroll.py")
    else:
        exit(1)
