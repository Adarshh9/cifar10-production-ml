#!/bin/bash
# Run all tests

set -e

echo "🧪 Running tests..."

# Unit tests
pytest tests/ -v --cov=src --cov-report=term-missing

# API integration tests
echo "Starting API for integration tests..."
docker-compose up -d api redis mlflow

# Wait for API to be ready
sleep 10

# Run integration tests
pytest tests/test_api.py -v

# Cleanup
docker-compose down

echo "✅ All tests passed!"
