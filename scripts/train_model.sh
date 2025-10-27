#!/bin/bash
# Train model with MLflow tracking

set -e

echo "🚀 Starting model training..."

# Activate environment (adjust for your setup)
# source venv/bin/activate

# Run training
python -m src.training.train \
    --config configs/model_config.yaml \
    --data-path data/ \
    --epochs 50 \
    --batch-size 32

echo "✅ Training complete!"
echo "Check results at http://localhost:5000"
