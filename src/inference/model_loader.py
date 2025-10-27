"""Model loader for CIFAR-10 ResNet."""
import torch
import logging
from typing import Optional
import mlflow

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load model from MLflow or local path."""
    
    def __init__(self, model_name: str, mlflow_uri: str):
        self.model_name = model_name
        self.mlflow_uri = mlflow_uri
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_version = "local"
        
        mlflow.set_tracking_uri(mlflow_uri)
    
    def load_from_registry(self, stage: str = "Production") -> bool:
        """Load model from MLflow registry."""
        try:
            model_uri = f"models:/{self.model_name}/{stage}"
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_version = stage
            logger.info(f"✅ Loaded model from registry: {model_uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}")
            return False
    
    def load_from_path(self, path: str) -> bool:
        """Load model from local checkpoint."""
        try:
            # Import the model class
            from src.training.model import create_model
            
            # Create model
            self.model = create_model(num_classes=10, pretrained=False)
            
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
            self.model_version = "local"
            logger.info(f"✅ Loaded model from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from path: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
