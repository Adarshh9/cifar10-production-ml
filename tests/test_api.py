"""API endpoint tests."""
import pytest
from fastapi import status


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_predict_endpoint(client, sample_features):
    """Test prediction endpoint."""
    response = client.post(
        "/predict",
        json={"features": sample_features}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probabilities" in data
    assert "confidence" in data
    assert isinstance(data["prediction"], int)


def test_batch_predict_endpoint(client, sample_batch):
    """Test batch prediction endpoint."""
    response = client.post(
        "/predict/batch",
        json={"features": sample_batch}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "batch_size" in data
    assert data["batch_size"] == len(sample_batch)


def test_invalid_input(client):
    """Test invalid input handling."""
    response = client.post(
        "/predict",
        json={"features": [0.1, 0.2]}  # Too few features
    )
    assert response.status_code == 422  # Validation error
