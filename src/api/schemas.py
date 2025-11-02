"""Pydantic schemas for API."""
from pydantic import BaseModel, ConfigDict
from typing import List, Optional

class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    model_config = ConfigDict(protected_namespaces=()) 
    
    prediction: int
    class_name: str
    confidence: float
    probabilities: List[float]
    model_version: str
    cached: bool = False

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    model_config = ConfigDict(protected_namespaces=())
    
    predictions: List[int]
    class_names: List[str]
    confidences: List[float]
    batch_size: int
    model_version: str

class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    redis_connected: bool
    model_name: Optional[str] = "CIFAR10_ResNet18"
    classes: List[str] = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
