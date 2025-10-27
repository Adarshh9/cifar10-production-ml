"""FastAPI application for CIFAR-10 predictions."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from contextlib import asynccontextmanager

from src.api.routes import router
from src.api import dependencies
from src.inference.model_loader import ModelLoader
from src.inference.predictor import CIFAR10Predictor
from src.utils.cache import RedisCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting CIFAR-10 Prediction API...")
    
    # Load model
    model_loader = ModelLoader(
        model_name="CIFAR10_ResNet18",
        mlflow_uri="http://mlflow:5000"
    )
    
    # Try loading from registry, fallback to local
    if not model_loader.load_from_registry(stage="Production"):
        logger.warning("⚠️ Falling back to local model")
        model_loader.load_from_path("models/best_model.pth")
    
    # Initialize cache
    cache = RedisCache(host="redis", port=6379)
    
    # Initialize predictor
    predictor = CIFAR10Predictor(model_loader, use_cache=True)
    predictor.set_cache(cache)  # ← KEY FIX: Connect cache to predictor
    
    # Set global dependencies
    dependencies.set_predictor(predictor)
    dependencies.set_model_loader(model_loader)
    dependencies.set_cache(cache)
    
    logger.info("✅ API ready!")
    logger.info("📊 Model: CIFAR10_ResNet18")
    logger.info(f"🔧 Cache enabled: {cache.is_connected()}")
    
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
