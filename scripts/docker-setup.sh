#!/bin/bash
# Docker setup script for Emotion Detection Deep Learning Project

set -e

echo "🐳 Setting up Docker environment for Emotion Detection Project"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p dataset trained_dl_models experiments logs

# Build Docker images
echo "🔨 Building Docker images..."
docker-compose build

# Pull base images to speed up future builds
echo "📥 Pulling base images..."
docker pull python:3.10-slim

echo "✅ Docker setup complete!"
echo ""
echo "Available commands:"
echo "  make docker-dev     - Run development container"
echo "  make docker-prod    - Run production container"
echo "  make docker-train   - Run training container"
echo "  make docker-docs    - Run documentation server"
echo ""
echo "To get started:"
echo "  make docker-dev"