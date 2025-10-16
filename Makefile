# Makefile for Emotion Detection Deep Learning Project

.PHONY: help install install-dev setup clean lint format test train ui

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  setup        Run full project setup"
	@echo "  clean        Clean up generated files"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black"
	@echo "  test         Run tests"
	@echo "  train        Train models"
	@echo "  ui           Launch user interface"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Run full project setup
setup:
	python cli.py setup

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.log" -delete

# Run code linting
lint:
	flake8 configuration/ set_up/ cli.py setup.py
	mypy configuration/ set_up/ cli.py --ignore-missing-imports

# Format code
format:
	black configuration/ set_up/ cli.py setup.py
	isort configuration/ set_up/ cli.py setup.py

# Run tests (placeholder for when tests are added)
test:
	@echo "Tests will be implemented in Phase 1 improvements"
	# pytest tests/

# Train models
train:
	python cli.py train

# Launch user interface
ui:
	python cli.py ui

# Show configuration
config:
	python cli.py config