"""Model loader for CIFAR-10 ResNet."""
import torch
import logging
from typing import Optional
import mlflow
import mlflow.pytorch
import os
from pathlib import Path


logger = logging.getLogger(__name__)


class ModelLoader:
    """Load model from MLflow or local path."""
    
    def __init__(self, model_name: str, mlflow_uri: str):
        self.model_name = model_name
        self.mlflow_uri = mlflow_uri
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_version = "local"
        
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
    
    def load_from_registry(self, stage: str = "Production") -> bool:
        """Load model from MLflow registry."""
        try:
            model_uri = "models:/CIFAR10_ResNet18/Production"
            logger.info(f"Loading model from MLflow: {model_uri}")

            # Use mlflow.pytorch.load_model directly; it handles the registry lookup
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_version = stage
            logger.info(f"✅ Loaded model from MLflow registry: {model_uri}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}")
            return False

    
    def load_from_path(self, path: str) -> bool:
        """Load model from local checkpoint."""
        try:
            # Handle both direct file paths and directories
            path = str(path)
            
            # If it's a directory, look for best_model.pth inside
            if os.path.isdir(path):
                model_file = os.path.join(path, "best_model.pth")
                if os.path.exists(model_file):
                    path = model_file
                else:
                    logger.error(f"No model file found in directory: {path}")
                    return False
            
            # Check if file exists
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
            
            logger.info(f"Loading model from: {path}")
            
            # Import the model class
            from src.training.model import create_model
            
            # Create model
            self.model = create_model(num_classes=10, weights=False)
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model = checkpoint
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_version = "mlflow" if "mlflow" in path else "local"
            
            logger.info(f"✅ Loaded model from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from path {path}: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
