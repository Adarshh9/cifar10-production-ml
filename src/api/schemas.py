"""API schemas for CIFAR-10 predictions."""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionResponse(BaseModel):
    """Response for single image prediction."""
    prediction: int = Field(..., description="Predicted class index (0-9)")
    class_name: str = Field(..., description="Human-readable class name")
    probabilities: List[float] = Field(..., description="Probability for each class")
    confidence: float = Field(..., description="Confidence of prediction")
    model_version: str = Field(..., description="Model version used")
    cached: bool = Field(default=False, description="Whether result was cached")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 3,
                "class_name": "cat",
                "probabilities": [0.01, 0.02, 0.03, 0.85, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01],
                "confidence": 0.85,
                "model_version": "1",
                "cached": False
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[int]
    class_names: List[str]
    confidences: List[float]
    batch_size: int
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    redis_connected: bool
    model_name: str = "CIFAR10_ResNet18"
    classes: List[str] = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
