"""Setup script for Neural Steganography Toolkit."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent.parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="neuralsteg",
    version="0.1.0",
    author="AI Security Research",
    description="Neural network steganography toolkit for security research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "safetensors>=0.3.0",
        "cryptography>=41.0.0",
        "click>=8.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.21.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "neuralsteg=neuralsteg.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="steganography, neural networks, security, research, ai",
)