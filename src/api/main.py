"""FastAPI application for CIFAR-10 predictions."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from contextlib import asynccontextmanager
import os

from src.api.routes import router
from src.api import dependencies
from src.inference.model_loader import ModelLoader
from src.inference.predictor import CIFAR10Predictor
from src.utils.cache import RedisCache
from src.monitoring.performance_monitor import PerformanceMonitor
# from src.monitoring.drift_detector import DriftDetector  # Will add later with baseline data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global performance monitor
performance_monitor = PerformanceMonitor(window_size=1000)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🚀 Starting CIFAR-10 Prediction API...")
    
    # Load model
    model_loader = ModelLoader(
        model_name="CIFAR10_ResNet18",
        mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    
    # Try loading from MLflow registry, fallback to local
    if not model_loader.load_from_registry(stage="Production"):
        logger.warning("⚠️ MLflow registry not available, falling back to local model")
        model_loader.load_from_path("models/best_model.pth")
    
    # Initialize Redis cache
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    
    cache = RedisCache(host=redis_host, port=redis_port)
    
    # Initialize predictor
    predictor = CIFAR10Predictor(model_loader, use_cache=True)
    predictor.set_cache(cache)
    
    # Set global dependencies
    dependencies.set_predictor(predictor)
    dependencies.set_model_loader(model_loader)
    dependencies.set_cache(cache)
    
    # Store performance monitor globally
    app.state.performance_monitor = performance_monitor
    
    logger.info("✅ API ready!")
    logger.info(f"📊 Model: CIFAR10_ResNet18")
    logger.info(f"🔧 Cache enabled: {cache.is_connected()}")
    logger.info(f"🎯 Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="CIFAR-10 Classification API",
    description="Production ML inference API for CIFAR-10 image classification",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Include routes
app.include_router(router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CIFAR-10 Classification API",
        "version": "1.0.0",
        "model": "ResNet-18",
        "classes": 10,
        "docs": "/docs"
    }
