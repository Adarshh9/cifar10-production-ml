"""Pytest configuration and fixtures."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.inference.model_loader import ModelLoader


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_features():
    """Sample features for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@pytest.fixture
def sample_batch():
    """Sample batch for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    ]
