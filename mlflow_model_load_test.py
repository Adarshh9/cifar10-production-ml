# test_mlflow_model_load.py
import os
import mlflow
import mlflow.pytorch
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "CIFAR10_ResNet18"
STAGE = "Production"

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    logger.info(f"Loading model from: {model_uri}")

    try:
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        logger.info(f"✅ Loaded and ran forward. Output shape: {y.shape}")
    except Exception as e:
        logger.exception("Failed to load model from MLflow: %s", e)

if __name__ == "__main__":
    main()
