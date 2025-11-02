"""API routes for CIFAR-10 predictions."""
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Request
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging

from src.api.schemas import (
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse
)
from src.api.dependencies import get_predictor, get_model_loader, get_cache

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loader = get_model_loader()
    cache = get_cache()
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded() if model_loader else False,
        redis_connected=cache.is_connected() if cache else False
    )

@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    model_loader = get_model_loader()
    if not model_loader or not model_loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return {"status": "ready"}

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Predict class for uploaded image.
    Upload a 32x32 image (will be resized automatically).
    Supported formats: JPEG, PNG
    """
    predictor = get_predictor()
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialized"
        )
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        result = predictor.predict(image)
        
        # Record prediction for monitoring
        if hasattr(request.app.state, 'performance_monitor'):
            request.app.state.performance_monitor.record_prediction(
                confidence=result['confidence'],
                class_id=result['prediction']
            )
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: Request, files: list[UploadFile] = File(...)):
    """
    Predict classes for multiple images.
    Upload multiple images at once.
    Maximum batch size: 32 images.
    """
    predictor = get_predictor()
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialized"
        )
    
    if len(files) > 32:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size too large (max 32 images)"
        )
    
    try:
        images = []
        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
        
        result = predictor.predict_batch(images)
        
        # Record batch predictions
        if hasattr(request.app.state, 'performance_monitor'):
            for conf, class_id in zip(result['confidences'], result['predictions']):
                request.app.state.performance_monitor.record_prediction(
                    confidence=conf,
                    class_id=class_id
                )
        
        return BatchPredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.get("/classes")
async def get_classes():
    """Get list of CIFAR-10 classes."""
    from src.data.data_loader import CIFAR10_CLASSES
    return {
        "classes": CIFAR10_CLASSES,
        "num_classes": len(CIFAR10_CLASSES)
    }

@router.get("/monitoring/metrics")
async def get_monitoring_metrics(request: Request):
    """Get monitoring metrics."""
    if hasattr(request.app.state, 'performance_monitor'):
        metrics = request.app.state.performance_monitor.get_metrics()
        return {
            "performance": metrics,
            "drift_status": "monitoring"  # Will update when drift detector is active
        }
    return {"error": "Monitoring not initialized"}

@router.post("/cache/clear")
async def clear_cache():
    """Clear prediction cache (admin endpoint)."""
    cache = get_cache()
    if cache:
        cache.clear_all()
        return {"message": "Cache cleared"}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache not available"
        )
