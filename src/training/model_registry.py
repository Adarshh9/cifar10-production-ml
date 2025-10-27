"""MLflow Model Registry operations."""
from mlflow.tracking import MlflowClient
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manage models in MLflow Model Registry."""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.client = MlflowClient(tracking_uri)
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model"
    ) -> str:
        """
        Register a model from a run.
        
        Returns:
            model_version: The version number of the registered model
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = self.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        logger.info(f"Registered model {model_name} version {result.version}")
        return result.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ):
        """
        Transition model to a new stage.
        
        Args:
            model_name: Name of the registered model
            version: Version number
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
    
    def get_latest_model_version(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> str:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            stage: Optional stage filter ('Production', 'Staging', 'None')
        
        Returns:
            version: Latest version number
        """
        if stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
        else:
            versions = self.client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")
        
        latest = max(versions, key=lambda x: int(x.version))
        return latest.version
    
    def load_production_model(self, model_name: str):
        """Load the production version of a model."""
        import mlflow.pytorch
        
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(f"Loaded production model: {model_name}")
        return model
