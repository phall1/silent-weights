"""
Command-line interface for the Neural Steganography Toolkit.

Provides easy-to-use CLI commands for embedding, extracting, analyzing,
and detecting steganographic modifications in neural network models.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import click

from .core import NeuralSteg
from .exceptions import SteganographyError


# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.version_option(version='0.1.0')
def main(verbose: bool):
    """Neural Steganography Toolkit - Hide data in neural network models."""
    setup_logging(verbose)


@main.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('payload_path', type=click.Path(exists=True, path_type=Path))
@click.option('--password', '-p', help='Password for payload encryption')
@click.option('--bits-per-param', '-b', default=4, type=click.IntRange(1, 8),
              help='Number of bits to use per parameter (1-8)')
@click.option('--backup/--no-backup', default=True,
              help='Create backup before embedding (default: enabled)')
def embed(model_path: Path, payload_path: Path, password: Optional[str],
          bits_per_param: int, backup: bool):
    """Embed a file into neural network model weights."""
    
    click.echo(f"🔄 Embedding {payload_path.name} into {model_path.name}")
    click.echo(f"   Bits per parameter: {bits_per_param}")
    click.echo(f"   Encryption: {'enabled' if password else 'disabled'}")
    click.echo(f"   Backup: {'enabled' if backup else 'disabled'}")
    
    try:
        # Initialize steganography engine
        steg = NeuralSteg(model_path, password=password)
        
        # Analyze capacity first
        capacity = steg.analyze_capacity(bits_per_param)
        payload_size = payload_path.stat().st_size
        
        click.echo(f"\n📊 Capacity Analysis:")
        click.echo(f"   Model capacity: {capacity.capacity_mb:.1f} MB")
        click.echo(f"   Payload size: {payload_size / (1024*1024):.2f} MB")
        click.echo(f"   Utilization: {(payload_size / capacity.capacity_bytes) * 100:.2f}%")
        
        if payload_size > capacity.capacity_bytes:
            click.echo(f"❌ Error: Payload too large for model capacity", err=True)
            sys.exit(1)
        
        # Confirm operation
        if not click.confirm(f"\nProceed with embedding?"):
            click.echo("Operation cancelled")
            sys.exit(0)
        
        # Perform embedding
        with click.progressbar(length=100, label='Embedding') as bar:
            result = steg.embed(payload_path, bits_per_param)
            bar.update(100)
        
        # Display results
        click.echo(f"\n✅ Embedding successful!")
        click.echo(f"   Embedded: {result.embedded_bytes:,} bytes")
        click.echo(f"   Capacity used: {result.capacity_used*100:.2f}%")
        click.echo(f"   Processing time: {result.processing_time:.2f}s")
        click.echo(f"   Checksum: {result.checksum[:16]}...")
        
        if backup:
            backups = steg.list_backups()
            if backups:
                click.echo(f"   Backup created: {backups[0]['name']}")
        
    except SteganographyError as e:
        click.echo(f"❌ Steganography error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option('--password', '-p', help='Password for payload decryption')
@click.option('--bits-per-param', '-b', default=4, type=click.IntRange(1, 8),
              help='Number of bits used per parameter during embedding')
def extract(model_path: Path, output_path: Path, password: Optional[str],
            bits_per_param: int):
    """Extract hidden file from neural network model weights."""
    
    click.echo(f"🔄 Extracting from {model_path.name} to {output_path.name}")
    click.echo(f"   Bits per parameter: {bits_per_param}")
    click.echo(f"   Decryption: {'enabled' if password else 'disabled'}")
    
    try:
        # Initialize steganography engine
        steg = NeuralSteg(model_path, password=password)
        
        # Perform extraction
        with click.progressbar(length=100, label='Extracting') as bar:
            result = steg.extract(output_path, bits_per_param)
            bar.update(100)
        
        # Display results
        click.echo(f"\n✅ Extraction successful!")
        click.echo(f"   Extracted: {result.extracted_bytes:,} bytes")
        click.echo(f"   Output file: {result.output_path}")
        click.echo(f"   Checksum verified: {result.checksum_verified}")
        click.echo(f"   Processing time: {result.processing_time:.2f}s")
        
    except SteganographyError as e:
        click.echo(f"❌ Steganography error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.option('--bits-per-param', '-b', default=4, type=click.IntRange(1, 8),
              help='Number of bits per parameter to analyze')
def analyze(model_path: Path, bits_per_param: int):
    """Analyze model embedding capacity and statistics."""
    
    click.echo(f"🔍 Analyzing {model_path.name}")
    
    try:
        # Initialize steganography engine
        steg = NeuralSteg(model_path)
        
        # Analyze capacity
        capacity = steg.analyze_capacity(bits_per_param)
        
        click.echo(f"\n📊 Capacity Analysis:")
        click.echo(f"   Target parameters: {capacity.total_target_params:,}")
        click.echo(f"   Embedding capacity: {capacity.capacity_mb:.1f} MB")
        click.echo(f"   Bits per parameter: {capacity.bits_per_param}")
        click.echo(f"   Target tensors: {capacity.target_tensor_count}")
        
        # Model integrity check
        integrity_ok = steg.verify_integrity()
        click.echo(f"\n🔒 Model Integrity:")
        click.echo(f"   Status: {'✅ OK' if integrity_ok else '❌ Issues detected'}")
        
        # List backups if any
        backups = steg.list_backups()
        if backups:
            click.echo(f"\n💾 Available Backups:")
            for backup in backups[:5]:  # Show first 5
                click.echo(f"   {backup['name']} ({backup['size_mb']:.1f} MB) - {backup['created_at']}")
            if len(backups) > 5:
                click.echo(f"   ... and {len(backups) - 5} more")
        
    except SteganographyError as e:
        click.echo(f"❌ Steganography error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.option('--clean-model', '-c', type=click.Path(exists=True, path_type=Path),
              help='Path to clean reference model for comparison')
@click.option('--bits-per-param', '-b', default=4, type=click.IntRange(1, 8),
              help='Number of bits per parameter to analyze')
def detect(model_path: Path, clean_model: Optional[Path], bits_per_param: int):
    """Detect potential steganographic modifications in model."""
    
    click.echo(f"🕵️  Analyzing {model_path.name} for steganographic modifications")
    if clean_model:
        click.echo(f"   Comparing with clean model: {clean_model.name}")
    
    try:
        # Initialize steganography engine
        steg = NeuralSteg(model_path)
        
        # Perform detection analysis
        with click.progressbar(length=100, label='Analyzing') as bar:
            result = steg.detect_anomalies(clean_model)
            bar.update(100)
        
        # Display results
        click.echo(f"\n🔍 Detection Results:")
        click.echo(f"   Suspicious: {'⚠️  YES' if result.suspicious else '✅ NO'}")
        click.echo(f"   Entropy anomaly: {'⚠️  YES' if result.entropy_anomaly else '✅ NO'}")
        
        click.echo(f"\n📈 Statistical Tests:")
        for test_name, value in result.statistical_tests.items():
            click.echo(f"   {test_name}: {value:.4f}")
        
        if result.recommendations:
            click.echo(f"\n💡 Recommendations:")
            for rec in result.recommendations:
                click.echo(f"   • {rec}")
        
        # Exit with appropriate code
        sys.exit(1 if result.suspicious else 0)
        
    except SteganographyError as e:
        click.echo(f"❌ Steganography error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
def backup(model_path: Path):
    """Create a backup of the model."""
    
    click.echo(f"💾 Creating backup of {model_path.name}")
    
    try:
        steg = NeuralSteg(model_path)
        backup_path = steg.create_backup()
        
        click.echo(f"✅ Backup created: {backup_path.name}")
        click.echo(f"   Location: {backup_path}")
        
    except SteganographyError as e:
        click.echo(f"❌ Steganography error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('backup_name')
def restore(model_path: Path, backup_name: str):
    """Restore model from a backup."""
    
    click.echo(f"🔄 Restoring {model_path.name} from backup: {backup_name}")
    
    if not click.confirm("This will overwrite the current model. Continue?"):
        click.echo("Operation cancelled")
        sys.exit(0)
    
    try:
        steg = NeuralSteg(model_path)
        steg.restore_backup(backup_name)
        
        click.echo(f"✅ Model restored from backup: {backup_name}")
        
    except SteganographyError as e:
        click.echo(f"❌ Steganography error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
def list_backups(model_path: Path):
    """List all available backups for a model."""
    
    try:
        steg = NeuralSteg(model_path)
        backups = steg.list_backups()
        
        if not backups:
            click.echo(f"No backups found for {model_path.name}")
            return
        
        click.echo(f"💾 Backups for {model_path.name}:")
        click.echo()
        
        for backup in backups:
            click.echo(f"   {backup['name']}")
            click.echo(f"      Created: {backup['created_at']}")
            click.echo(f"      Size: {backup['size_mb']:.1f} MB")
            click.echo()
        
    except SteganographyError as e:
        click.echo(f"❌ Steganography error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('original_payload', type=click.Path(exists=True, path_type=Path))
@click.argument('extracted_payload', type=click.Path(exists=True, path_type=Path))
def verify(model_path: Path, original_payload: Path, extracted_payload: Path):
    """Verify that extracted payload matches the original."""
    
    click.echo(f"🔍 Verifying payload extraction...")
    click.echo(f"   Original: {original_payload.name}")
    click.echo(f"   Extracted: {extracted_payload.name}")
    
    try:
        steg = NeuralSteg(model_path)
        
        with click.progressbar(length=100, label='Verifying') as bar:
            result = steg.verify_extraction(original_payload, extracted_payload)
            bar.update(100)
        
        # Display results
        click.echo(f"\n✅ Verification Results:")
        click.echo(f"   Files match: {'✅ YES' if result.payload_match else '❌ NO'}")
        click.echo(f"   Checksums match: {'✅ YES' if result.payload_checksum_verified else '❌ NO'}")
        click.echo(f"   Original size: {result.original_size:,} bytes")
        click.echo(f"   Extracted size: {result.extracted_size:,} bytes")
        click.echo(f"   Verification time: {result.verification_time:.2f}s")
        
        if result.mismatch_details:
            click.echo(f"\n⚠️  Mismatch Details:")
            for key, value in result.mismatch_details.items():
                click.echo(f"   {key}: {value}")
        
        # Exit with appropriate code
        success = result.payload_match and result.payload_checksum_verified
        sys.exit(0 if success else 1)
        
    except SteganographyError as e:
        click.echo(f"❌ Steganography error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.option('--prompt', '-p', default="Hello, how are you?", 
              help='Test prompt for the model')
@click.option('--max-tokens', '-t', default=50, type=click.IntRange(1, 500),
              help='Maximum tokens to generate')
@click.option('--temperature', default=0.7, type=click.FloatRange(0.1, 2.0),
              help='Sampling temperature')
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'mps', 'auto']), 
              default='auto', help='Device to run inference on')
def test(model_path: Path, prompt: str, max_tokens: int, temperature: float, device: str):
    """Test model inference with a single prompt."""
    
    click.echo(f"🧠 Testing model inference...")
    click.echo(f"   Model: {model_path.name}")
    click.echo(f"   Prompt: \"{prompt}\"")
    click.echo(f"   Device: {device}")
    
    try:
        steg = NeuralSteg(model_path)
        
        device_param = None if device == 'auto' else device
        
        with click.progressbar(length=100, label='Testing') as bar:
            result = steg.test_inference(
                prompt=prompt, 
                max_tokens=max_tokens, 
                temperature=temperature,
                device=device_param
            )
            bar.update(100)
        
        # Display results
        if result.success:
            click.echo(f"\n✅ Inference Successful:")
            click.echo(f"   Response: \"{result.response}\"")
            click.echo(f"   Tokens generated: {result.tokens_generated}")
            click.echo(f"   Inference time: {result.inference_time:.2f}s")
            click.echo(f"   Model responsive: {'✅ YES' if result.model_responsive else '❌ NO'}")
        else:
            click.echo(f"\n❌ Inference Failed:")
            click.echo(f"   Error: {result.error_message}")
            click.echo(f"   Inference time: {result.inference_time:.2f}s")
        
        sys.exit(0 if result.success else 1)
        
    except SteganographyError as e:
        click.echo(f"❌ Steganography error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.option('--use-gpu', is_flag=True, help='Use GPU acceleration if available')
@click.option('--max-tokens', '-t', default=50, type=click.IntRange(1, 500),
              help='Maximum tokens per generation')
@click.option('--test-suite', default='default', 
              type=click.Choice(['default', 'quick', 'comprehensive']),
              help='Test suite to run')
def benchmark(model_path: Path, use_gpu: bool, max_tokens: int, test_suite: str):
    """Run comprehensive model performance benchmarks."""
    
    click.echo(f"📊 Running comprehensive model benchmark...")
    click.echo(f"   Model: {model_path.name}")
    click.echo(f"   GPU acceleration: {'✅ Enabled' if use_gpu else '❌ Disabled'}")
    click.echo(f"   Test suite: {test_suite}")
    
    try:
        steg = NeuralSteg(model_path)
        
        # Prepare test prompts based on suite
        test_prompts = None
        if test_suite == 'quick':
            test_prompts = ["Hello!", "What is 2+2?", "Tell me a joke."]
        elif test_suite == 'comprehensive':
            # Use default prompts (more comprehensive)
            test_prompts = None
        
        with click.progressbar(length=100, label='Benchmarking') as bar:
            result = steg.comprehensive_test(
                test_prompts=test_prompts,
                use_gpu=use_gpu,
                max_tokens=max_tokens
            )
            bar.update(100)
        
        # Display results
        click.echo(f"\n📊 Benchmark Results:")
        click.echo(f"   Overall success: {'✅ YES' if result.overall_success else '❌ NO'}")
        click.echo(f"   Tests passed: {result.tests_passed}/{result.tests_passed + result.tests_failed}")
        click.echo(f"   Success rate: {result.summary_stats['success_rate']:.1%}")
        click.echo(f"   Average inference time: {result.average_inference_time:.2f}s")
        click.echo(f"   Average tokens generated: {result.summary_stats['average_tokens_generated']:.1f}")
        click.echo(f"   Performance degradation: {result.performance_degradation:.1%}")
        click.echo(f"   GPU accelerated: {'✅ YES' if result.gpu_accelerated else '❌ NO'}")
        click.echo(f"   Total test time: {result.summary_stats['total_test_time']:.2f}s")
        
        if result.tests_failed > 0:
            click.echo(f"\n⚠️  Failed Tests:")
            for i, test_result in enumerate(result.test_results):
                if not test_result.success:
                    click.echo(f"   Test {i+1}: {test_result.error_message}")
        
        sys.exit(0 if result.overall_success else 1)
        
    except SteganographyError as e:
        click.echo(f"❌ Steganography error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()