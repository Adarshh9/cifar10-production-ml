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
    """MLflow tracking wrapper for experiments."""
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "cifar10_production"
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            raise
        
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.client = MlflowClient(tracking_uri)
        self.run_id = None
    
    def start_run(self, run_name: Optional[str] = None):
        """Start an MLflow run."""
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
        self.run_id = self.active_run.info.run_id
        logger.info(f"Started MLflow run: {self.run_id}")
        return self
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            logger.error(f"Run failed: {exc_val}")
            mlflow.set_tag("status", "failed")
        else:
            mlflow.set_tag("status", "completed")
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags."""
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """
        Log PyTorch model to MLflow.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path where model will be stored
            registered_model_name: Name to register model in registry
        """
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name
            )
            logger.info(f"Logged model to MLflow: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file or directory as an artifact."""
        mlflow.log_artifact(local_path, artifact_path)
