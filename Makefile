# Makefile for Emotion Detection Deep Learning Project

.PHONY: help install install-dev setup clean lint format test train ui docs experiments

# Default target
help:
	@echo "Available commands:"
	@echo "  install         Install production dependencies"
	@echo "  install-dev     Install development dependencies"
	@echo "  setup           Run full project setup"
	@echo "  clean           Clean up generated files"
	@echo "  lint            Run code linting"
	@echo "  format          Format code with black"
	@echo "  test            Run tests"
	@echo "  test-coverage   Run tests with coverage"
	@echo "  train           Train models"
	@echo "  train-tracked   Train models with experiment tracking"
	@echo "  ui              Launch user interface"
	@echo "  docs            Build documentation"
	@echo "  docs-serve      Serve documentation locally"
	@echo "  experiments     List experiments"
	@echo "  validate-data   Validate dataset"
	@echo "  preprocess-data Run data preprocessing
	@echo "  optimize        Optimize trained models"
	@echo "  benchmark       Run performance benchmarks"
	@echo "  docker-build    Build Docker images"
	@echo "  docker-dev      Run development container"
	@echo "  docker-prod     Run production container"
	@echo "  docker-train    Run training in container""

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

# Run tests
test:
	pytest tests/

# Run tests with coverage
test-coverage:
	pytest tests/ --cov=configuration --cov=set_up --cov=dl_scripts --cov=user_interface --cov=data_pipeline --cov=experiment_tracking --cov-report=html --cov-report=term

# Train models
train:
	python cli.py train

# Train models with experiment tracking
train-tracked:
	python cli.py train --track-experiment --experiment-name "makefile_training"

# Launch user interface
ui:
	python cli.py ui

# Build documentation
docs:
	cd docs && make html

# Serve documentation locally
docs-serve:
	cd docs/_build/html && python -m http.server 8000

# List experiments
experiments:
	python cli.py experiments list

# Validate dataset
validate-data:
	python data_pipeline/data_validator.py dataset --output validation_report.json

# Run data preprocessing
preprocess-data:
	python data_pipeline/preprocessing.py

# Show configuration
config:
	python cli.py config

# Optimize models
optimize:
	python cli.py optimize trained_dl_models/emotion_detection_from_image_transfer_learning.h5

# Run performance benchmark
benchmark:
	python cli.py benchmark --duration 30

# Docker commands
docker-build:
	docker-compose build

docker-dev:
	docker-compose up emotion-detection-dev

docker-prod:
	docker-compose up -d emotion-detection-prod

docker-train:
	docker-compose up emotion-detection-train

docker-docs:
	docker-compose up -d docs

docker-clean:
	docker-compose down
	docker system prune -f