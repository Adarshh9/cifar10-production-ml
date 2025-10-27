"""MLflow experiment tracking utilities."""
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Wrapper for MLflow tracking operations."""
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "default"
    ):
        """
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.run_id = None
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"Experiment: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        logger.info(f"Started run: {self.run_id}")
        return self.run
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """Log PyTorch model."""
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info(f"Logged model to {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set run tags."""
        mlflow.set_tags(tags)
    
    def end_run(self):
        """End the current run."""
        mlflow.end_run()
        logger.info(f"Ended run: {self.run_id}")


def log_training_run(
    model: torch.nn.Module,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model_name: str,
    tracking_uri: str,
    experiment_name: str
) -> str:
    """
    Log a complete training run to MLflow.
    
    Returns:
        run_id: The MLflow run ID
    """
    tracker = MLflowTracker(tracking_uri, experiment_name)
    
    with tracker.start_run(run_name=f"{model_name}_training"):
        # Log parameters
        tracker.log_params(params)
        
        # Log final metrics
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Set tags
        tracker.set_tags({
            "model_type": "TaxoCapsNet",
            "framework": "pytorch",
            "stage": "training"
        })
        
        run_id = tracker.run_id
    
    return run_id
