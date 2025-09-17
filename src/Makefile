# Neural Steganography Toolkit - Development Makefile

.PHONY: help install install-dev test test-cov lint format type-check clean build upload docs

# Default target
help:
	@echo "Neural Steganography Toolkit - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  test-fast    Run tests excluding slow ones"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run all linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run mypy type checking"
	@echo ""
	@echo "Build & Release:"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build distribution packages"
	@echo "  upload       Upload to PyPI (requires credentials)"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

# Testing
test:
	pytest

test-cov:
	pytest --cov=neuralsteg --cov-report=html --cov-report=term-missing

test-fast:
	pytest -m "not slow"

# Code quality
lint: format type-check
	@echo "✅ All linting checks passed"

format:
	black neuralsteg tests
	isort neuralsteg tests

type-check:
	mypy neuralsteg

# Build and release
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

# Documentation
docs:
	@echo "Documentation build not implemented yet"
	@echo "Consider adding Sphinx documentation in the future"

# Development workflow
dev-setup: install-dev
	pre-commit install
	@echo "✅ Development environment ready"

# Quick development check
check: format type-check test-fast
	@echo "✅ Quick development check passed"

# Full CI check
ci: format type-check test
	@echo "✅ Full CI check passed"