#!/bin/bash
# Setup MLflow tracking server

set -e

echo "🔧 Setting up MLflow..."

# Create MLflow directory
mkdir -p mlruns mlartifacts

# Start MLflow server (for local development)
# For production, use docker-compose
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 0.0.0.0 \
    --port 5000 &

echo "✅ MLflow server started at http://localhost:5000"
echo "PID: $!"
