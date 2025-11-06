#!/bin/bash
# Performance testing script

set -e

echo "🚀 Running Performance Tests for Emotion Detection Project"

# Check if models exist
if [ ! -d "trained_dl_models" ] || [ -z "$(ls -A trained_dl_models)" ]; then
    echo "❌ No trained models found. Please train models first:"
    echo "   python cli.py train"
    exit 1
fi

echo "📊 Running system benchmark..."
python cli.py benchmark --duration 30

echo ""
echo "🔧 Testing model optimization..."
for model in trained_dl_models/*.h5; do
    if [ -f "$model" ]; then
        echo "Optimizing: $model"
        python cli.py optimize "$model" --optimization-type tflite
    fi
done

echo ""
echo "📈 Performance test complete!"
echo "Check the logs for detailed performance metrics."