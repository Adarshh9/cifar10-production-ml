#!/bin/bash
# Deploy entire stack locally

set -e

echo "🚀 Deploying ML Production Pipeline locally..."

# Build Docker image
echo "Building Docker image..."
docker-compose build

# Start all services
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check health
echo "Checking service health..."
curl -f http://localhost:8000/health || exit 1

echo "✅ Deployment successful!"
echo ""
echo "Services:"
echo "  - API:        http://localhost:8000"
echo "  - API Docs:   http://localhost:8000/docs"
echo "  - MLflow:     http://localhost:5000"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana:    http://localhost:3000 (admin/admin)"
echo ""
echo "Try a prediction:"
echo 'curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}"'
