#!/bin/bash
# Setup DVC for data versioning

set -e

echo "🔧 Setting up DVC..."

# Initialize DVC
dvc init

# Add remote storage (example: local directory)
# For production, use S3, GCS, or Azure Blob
dvc remote add -d local_storage /tmp/dvc-storage
mkdir -p /tmp/dvc-storage

# Track data directories
dvc add data/
dvc add models/

# Commit DVC files
git add data.dvc models.dvc .dvc/.gitignore .dvc/config
git commit -m "Initialize DVC tracking"

echo "✅ DVC setup complete!"
echo "To push data: dvc push"
echo "To pull data: dvc pull"
