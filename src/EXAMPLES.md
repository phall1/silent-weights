# Neural Steganography Toolkit - Examples

This document provides detailed examples of using the Neural Steganography Toolkit for various research scenarios. All examples are based on production-tested capabilities, including the successful embedding of a 3.4MB MP3 file in a LLaMA-3.2-3B model with zero performance impact.

## Table of Contents

1. [Basic Operations](#basic-operations)
2. [Security Research Scenarios](#security-research-scenarios)
3. [Detection and Analysis](#detection-and-analysis)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)

## Basic Operations

### Simple File Embedding

```python
from neuralsteg import NeuralSteg
from pathlib import Path

# Initialize with a model (production-tested with LLaMA-3.2-3B)
model_path = Path("./models/llama-3.2-3b")
steg = NeuralSteg(model_path)

# Check capacity first - demonstrated: 3.4MB in 3B model
capacity = steg.analyze_capacity()
print(f"Model can hide {capacity.capacity_mb:.1f} MB of data")

# Embed a document (tested up to 3.4MB)
payload_path = Path("./documents/research_notes.pdf")
result = steg.embed(payload_path)

print(f"✅ Embedded {result.embedded_bytes:,} bytes")
print(f"📊 Used {result.capacity_used*100:.2f}% of capacity")
print(f"⏱️ Processing time: {result.processing_time:.2f}s")
print(f"🔐 Encryption: {result.encryption_enabled}")
```

### Encrypted Embedding

```python
# Embed with encryption
password = "research_project_2024"
steg = NeuralSteg(model_path, password=password)

# Embed sensitive data
result = steg.embed("./data/sensitive_dataset.zip")
print(f"🔐 Encrypted and embedded {result.embedded_bytes:,} bytes")

# Extract later
extracted = steg.extract("./recovered/dataset.zip")
print(f"🔓 Decrypted and extracted {extracted.extracted_bytes:,} bytes")
```

## Security Research Scenarios

### Supply Chain Attack Simulation

```python
"""
Simulate a supply chain attack where a malicious actor
embeds a payload in a popular model before distribution.
"""

from neuralsteg import NeuralSteg
import tempfile
import shutil

# Scenario: Attacker has access to model before distribution
original_model = "./models/popular-model-v1.0"
malicious_model = "./models/popular-model-v1.0-compromised"

# Copy model to simulate attacker's environment
shutil.copytree(original_model, malicious_model)

# Embed malicious payload
steg = NeuralSteg(malicious_model, password="hidden_backdoor_2024")
malicious_payload = "./payloads/data_exfiltration_tool.py"

result = steg.embed(malicious_payload, bits_per_param=2)  # Conservative for stealth

print(f"🎭 Malicious payload embedded:")
print(f"   Size: {result.embedded_bytes:,} bytes")
print(f"   Capacity used: {result.capacity_used*100:.3f}%")
print(f"   Encryption: {result.encryption_enabled}")

# Verify model still functions normally
integrity_ok = steg.verify_integrity()
print(f"   Model integrity: {'✅ OK' if integrity_ok else '❌ Compromised'}")

# Later: Victim downloads and uses the model
# The payload remains hidden until extracted by the attacker
```

### Red Team Exercise

```python
"""
Red team exercise: Test if blue team can detect
steganographic modifications in deployed models.
"""

def red_team_embed_phase():
    """Red team embeds test payload."""
    target_model = "./deployment/production-model"

    # Create backup for restoration after exercise
    steg = NeuralSteg(target_model)
    backup_path = steg.create_backup("red_team_exercise")

    # Embed test payload with evasion techniques
    test_payload = "./red_team/test_payload.bin"
    result = steg.embed(test_payload, bits_per_param=3)  # Moderate stealth

    return {
        "embedded_bytes": result.embedded_bytes,
        "capacity_used": result.capacity_used,
        "backup_path": backup_path,
        "checksum": result.checksum
    }

def blue_team_detect_phase(model_path, clean_reference):
    """Blue team attempts detection."""
    steg = NeuralSteg(model_path)

    # Run detection analysis
    detection = steg.detect_anomalies(clean_reference)

    print(f"🔍 Blue Team Detection Results:")
    print(f"   Suspicious: {'⚠️ YES' if detection.suspicious else '✅ NO'}")
    print(f"   Entropy anomaly: {detection.entropy_anomaly}")

    for test, value in detection.statistical_tests.items():
        print(f"   {test}: {value:.4f}")

    return detection.suspicious

# Execute exercise
red_results = red_team_embed_phase()
detected = blue_team_detect_phase(
    "./deployment/production-model",
    "./reference/clean-production-model"
)

print(f"\n🎯 Exercise Results:")
print(f"   Red team embedded: {red_results['embedded_bytes']:,} bytes")
print(f"   Blue team detected: {'YES' if detected else 'NO'}")
```

### Model Hub Vulnerability Assessment

```python
"""
Assess vulnerability of model distribution platforms
to steganographic attacks.
"""

import os
from pathlib import Path

def assess_model_vulnerability(model_path, test_payloads):
    """Assess how much data can be hidden in a model."""
    steg = NeuralSteg(model_path)

    # Analyze capacity with different bit densities
    results = {}
    for bits in [1, 2, 4, 6, 8]:
        capacity = steg.analyze_capacity(bits_per_param=bits)
        results[f"{bits}_bit"] = {
            "capacity_mb": capacity.capacity_mb,
            "target_params": capacity.total_target_params
        }

    # Test with various payload sizes
    vulnerability_report = {
        "model_path": str(model_path),
        "capacity_analysis": results,
        "payload_tests": []
    }

    for payload_path in test_payloads:
        payload_size = os.path.getsize(payload_path)
        payload_mb = payload_size / (1024 * 1024)

        # Find minimum bits needed
        min_bits_needed = None
        for bits in [1, 2, 4, 6, 8]:
            if payload_mb <= results[f"{bits}_bit"]["capacity_mb"]:
                min_bits_needed = bits
                break

        vulnerability_report["payload_tests"].append({
            "payload": payload_path.name,
            "size_mb": payload_mb,
            "min_bits_needed": min_bits_needed,
            "embeddable": min_bits_needed is not None
        })

    return vulnerability_report

# Test with various payload types
test_payloads = [
    Path("./test_payloads/small_script.py"),      # 10 KB
    Path("./test_payloads/medium_document.pdf"),  # 500 KB
    Path("./test_payloads/large_dataset.zip"),    # 50 MB
    Path("./test_payloads/malware_sample.exe"),   # 2 MB
]

models_to_test = [
    "./models/bert-base-uncased",
    "./models/gpt2-medium",
    "./models/llama-7b",
    "./models/llama-13b"
]

print("🔬 Model Hub Vulnerability Assessment")
print("=" * 50)

for model_path in models_to_test:
    if Path(model_path).exists():
        report = assess_model_vulnerability(Path(model_path), test_payloads)

        print(f"\n📊 {model_path}")
        print(f"   Max capacity (1-bit): {report['capacity_analysis']['1_bit']['capacity_mb']:.1f} MB")
        print(f"   Max capacity (4-bit): {report['capacity_analysis']['4_bit']['capacity_mb']:.1f} MB")

        for test in report["payload_tests"]:
            status = "✅ FITS" if test["embeddable"] else "❌ TOO LARGE"
            bits = f"({test['min_bits_needed']}-bit)" if test["embeddable"] else ""
            print(f"   {test['payload']}: {status} {bits}")
```

## Detection and Analysis

### Comprehensive Model Analysis

```python
"""
Perform comprehensive analysis of a potentially
compromised model.
"""

def comprehensive_analysis(model_path, clean_reference=None):
    """Perform full steganographic analysis."""
    steg = NeuralSteg(model_path)

    print(f"🔍 Comprehensive Analysis: {model_path}")
    print("=" * 60)

    # 1. Basic integrity check
    integrity = steg.verify_integrity()
    print(f"📋 Model Integrity: {'✅ OK' if integrity else '❌ ISSUES'}")

    # 2. Capacity analysis
    capacity = steg.analyze_capacity()
    print(f"📊 Embedding Capacity: {capacity.capacity_mb:.1f} MB")
    print(f"   Target parameters: {capacity.total_target_params:,}")
    print(f"   Target tensors: {capacity.target_tensor_count}")

    # 3. Statistical analysis
    detection = steg.detect_anomalies(clean_reference)
    print(f"🚨 Anomaly Detection: {'⚠️ SUSPICIOUS' if detection.suspicious else '✅ CLEAN'}")

    if detection.entropy_anomaly:
        print("   ⚠️ Entropy anomaly detected")

    print("   Statistical Tests:")
    for test, value in detection.statistical_tests.items():
        print(f"     {test}: {value:.4f}")

    # 4. Recommendations
    print("💡 Recommendations:")
    for rec in detection.recommendations:
        print(f"   • {rec}")

    # 5. Backup status
    backups = steg.list_backups()
    print(f"💾 Available Backups: {len(backups)}")
    for backup in backups[:3]:  # Show first 3
        print(f"   {backup['name']} ({backup['size_mb']:.1f} MB)")

    return {
        "integrity": integrity,
        "suspicious": detection.suspicious,
        "capacity_mb": capacity.capacity_mb,
        "backup_count": len(backups)
    }

# Analyze multiple models
models_to_analyze = [
    ("./models/production-model", "./models/clean-reference"),
    ("./models/downloaded-model", None),
    ("./models/fine-tuned-model", "./models/base-model")
]

for model_path, clean_ref in models_to_analyze:
    if Path(model_path).exists():
        analysis = comprehensive_analysis(model_path, clean_ref)
        print("\n" + "="*60 + "\n")
```

### Forensic Analysis Workflow

```python
"""
Forensic analysis workflow for investigating
potentially compromised models.
"""

def forensic_investigation(suspicious_model, evidence_dir):
    """Conduct forensic investigation of suspicious model."""
    evidence_path = Path(evidence_dir)
    evidence_path.mkdir(exist_ok=True)

    steg = NeuralSteg(suspicious_model)

    print("🕵️ Starting Forensic Investigation")
    print(f"   Target: {suspicious_model}")
    print(f"   Evidence: {evidence_dir}")

    # 1. Create forensic backup
    forensic_backup = steg.create_backup("forensic_evidence")
    print(f"📁 Forensic backup: {forensic_backup}")

    # 2. Document model state
    capacity = steg.analyze_capacity()
    detection = steg.detect_anomalies()

    # 3. Attempt extraction with common parameters
    extraction_attempts = []

    for bits in [1, 2, 4, 6, 8]:
        for attempt in range(3):  # Try different extraction points
            try:
                output_file = evidence_path / f"extracted_bits{bits}_attempt{attempt}.bin"
                result = steg.extract(output_file, bits_per_param=bits)

                extraction_attempts.append({
                    "bits_per_param": bits,
                    "attempt": attempt,
                    "success": True,
                    "bytes": result.extracted_bytes,
                    "file": str(output_file)
                })

                print(f"✅ Extraction successful: {bits}-bit, {result.extracted_bytes} bytes")
                break  # Success, no need for more attempts

            except Exception as e:
                extraction_attempts.append({
                    "bits_per_param": bits,
                    "attempt": attempt,
                    "success": False,
                    "error": str(e)
                })

    # 4. Generate forensic report
    report = {
        "investigation_date": "2024-01-01",  # Use actual date
        "model_path": str(suspicious_model),
        "forensic_backup": str(forensic_backup),
        "model_integrity": steg.verify_integrity(),
        "capacity_analysis": {
            "capacity_mb": capacity.capacity_mb,
            "target_params": capacity.total_target_params
        },
        "anomaly_detection": {
            "suspicious": detection.suspicious,
            "entropy_anomaly": detection.entropy_anomaly,
            "statistical_tests": detection.statistical_tests
        },
        "extraction_attempts": extraction_attempts
    }

    # Save report
    import json
    report_file = evidence_path / "forensic_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📋 Forensic report saved: {report_file}")

    return report

# Example investigation
if Path("./suspicious_models/model_v2.1").exists():
    investigation = forensic_investigation(
        "./suspicious_models/model_v2.1",
        "./evidence/case_001"
    )
```

## Advanced Usage

### Batch Processing

```python
"""
Process multiple models and payloads in batch.
"""

def batch_embed_operation(models, payloads, password=None):
    """Embed multiple payloads across multiple models."""
    results = []

    for model_path in models:
        for payload_path in payloads:
            try:
                steg = NeuralSteg(model_path, password=password)

                # Check capacity first
                capacity = steg.analyze_capacity()
                payload_size = payload_path.stat().st_size / (1024 * 1024)

                if payload_size > capacity.capacity_mb:
                    print(f"⚠️ Skipping {payload_path.name} - too large for {model_path.name}")
                    continue

                # Embed
                result = steg.embed(payload_path)
                results.append({
                    "model": model_path.name,
                    "payload": payload_path.name,
                    "success": True,
                    "bytes": result.embedded_bytes,
                    "capacity_used": result.capacity_used
                })

                print(f"✅ {model_path.name} + {payload_path.name}: {result.embedded_bytes:,} bytes")

            except Exception as e:
                results.append({
                    "model": model_path.name,
                    "payload": payload_path.name,
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ {model_path.name} + {payload_path.name}: {e}")

    return results

# Batch operation example
models = [Path(f"./models/model_{i}") for i in range(1, 4)]
payloads = [Path(f"./payloads/payload_{i}.bin") for i in range(1, 6)]

batch_results = batch_embed_operation(models, payloads, password="batch_2024")

# Summary
successful = sum(1 for r in batch_results if r["success"])
print(f"\n📊 Batch Summary: {successful}/{len(batch_results)} operations successful")
```

### Custom Target Layers

```python
"""
Use custom target layers for specific model architectures.
"""

# For BERT models
bert_targets = [
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense"
]

# For GPT models
gpt_targets = [
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj"
]

def embed_with_custom_targets(model_path, payload_path, target_layers):
    """Embed using custom target layer patterns."""
    steg = NeuralSteg(model_path)

    # Override default target layers
    steg.target_layers = target_layers

    # Analyze capacity with custom targets
    capacity = steg.analyze_capacity()
    print(f"Custom targets capacity: {capacity.capacity_mb:.1f} MB")
    print(f"Target tensors found: {capacity.target_tensor_count}")

    # Embed
    result = steg.embed(payload_path)
    return result

# Example usage
if Path("./models/bert-large").exists():
    result = embed_with_custom_targets(
        "./models/bert-large",
        "./payloads/test.bin",
        bert_targets
    )
    print(f"BERT embedding: {result.embedded_bytes:,} bytes")
```

## Troubleshooting

### Common Issues and Solutions

```python
"""
Common troubleshooting scenarios and solutions.
"""

def diagnose_embedding_failure(model_path, payload_path):
    """Diagnose why embedding might be failing."""
    print(f"🔧 Diagnosing embedding failure")
    print(f"   Model: {model_path}")
    print(f"   Payload: {payload_path}")

    try:
        steg = NeuralSteg(model_path)

        # Check 1: Model integrity
        if not steg.verify_integrity():
            print("❌ Model integrity check failed")
            return "model_corrupted"

        # Check 2: Capacity
        capacity = steg.analyze_capacity()
        payload_size = Path(payload_path).stat().st_size

        if payload_size > capacity.capacity_bytes:
            print(f"❌ Payload too large: {payload_size:,} > {capacity.capacity_bytes:,}")
            print(f"   Try reducing bits-per-param or use smaller payload")
            return "payload_too_large"

        # Check 3: File permissions
        if not os.access(model_path, os.W_OK):
            print("❌ No write permission to model directory")
            return "permission_denied"

        # Check 4: Disk space
        free_space = shutil.disk_usage(model_path).free
        if free_space < payload_size * 2:  # Need space for backup
            print("❌ Insufficient disk space for backup")
            return "insufficient_space"

        print("✅ All checks passed - embedding should work")
        return "ok"

    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
        return "diagnostic_failed"

def recovery_procedures(model_path):
    """Recovery procedures for corrupted models."""
    steg = NeuralSteg(model_path)

    print("🚑 Model Recovery Procedures")

    # List available backups
    backups = steg.list_backups()

    if not backups:
        print("❌ No backups available")
        return False

    print(f"📁 Found {len(backups)} backups:")
    for i, backup in enumerate(backups):
        print(f"   {i+1}. {backup['name']} ({backup['created_at']})")

    # Restore from most recent backup
    try:
        latest_backup = backups[0]['name']
        steg.restore_backup(latest_backup)
        print(f"✅ Restored from backup: {latest_backup}")

        # Verify restoration
        if steg.verify_integrity():
            print("✅ Model integrity verified after restoration")
            return True
        else:
            print("❌ Model still corrupted after restoration")
            return False

    except Exception as e:
        print(f"❌ Restoration failed: {e}")
        return False

# Example troubleshooting
if Path("./problematic_model").exists():
    diagnosis = diagnose_embedding_failure(
        "./problematic_model",
        "./large_payload.zip"
    )

    if diagnosis == "model_corrupted":
        recovery_procedures("./problematic_model")
```

### Performance Optimization

```python
"""
Performance optimization techniques.
"""

def optimize_for_large_models(model_path):
    """Optimize settings for large models."""
    steg = NeuralSteg(model_path)

    # Use fewer bits per parameter for faster processing
    capacity_2bit = steg.analyze_capacity(bits_per_param=2)
    capacity_4bit = steg.analyze_capacity(bits_per_param=4)

    print(f"Performance vs Capacity Trade-off:")
    print(f"   2-bit: {capacity_2bit.capacity_mb:.1f} MB (faster)")
    print(f"   4-bit: {capacity_4bit.capacity_mb:.1f} MB (slower)")

    # Recommend optimal settings
    if capacity_2bit.capacity_mb > 10:  # If we have plenty of capacity
        print("💡 Recommendation: Use 2-bit for better performance")
        return 2
    else:
        print("💡 Recommendation: Use 4-bit for better capacity")
        return 4

def monitor_embedding_progress(model_path, payload_path):
    """Monitor embedding progress and performance."""
    import time

    steg = NeuralSteg(model_path)

    start_time = time.time()

    # Embed with timing
    result = steg.embed(payload_path)

    end_time = time.time()

    # Calculate performance metrics
    throughput = result.embedded_bytes / result.processing_time

    print(f"📈 Performance Metrics:")
    print(f"   Processing time: {result.processing_time:.2f}s")
    print(f"   Throughput: {throughput/1024:.1f} KB/s")
    print(f"   Capacity utilization: {result.capacity_used*100:.2f}%")

    return result

# Example optimization
if Path("./large_model").exists():
    optimal_bits = optimize_for_large_models("./large_model")

    if Path("./test_payload.bin").exists():
        result = monitor_embedding_progress(
            "./large_model",
            "./test_payload.bin"
        )
```

These examples demonstrate the full range of capabilities of the Neural Steganography Toolkit, from basic operations to advanced security research scenarios. Each example includes error handling and best practices for real-world usage.
