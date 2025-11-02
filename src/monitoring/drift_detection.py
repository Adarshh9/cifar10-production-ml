"""Model drift detection system."""
import numpy as np
from scipy import stats
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    """Detect data and model drift."""
    
    def __init__(self, reference_data: np.ndarray):
        """Initialize with reference dataset."""
        self.reference_data = reference_data
        self.reference_stats = self._compute_stats(reference_data)
    
    def detect_drift(self, new_data: np.ndarray, threshold: float = 0.05) -> Dict:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            new_data: Recent prediction data
            threshold: P-value threshold (default 0.05)
        
        Returns:
            Drift detection results
        """
        new_stats = self._compute_stats(new_data)
        
        # KS test for distribution shift
        statistic, p_value = stats.ks_2samp(
            self.reference_data.flatten(),
            new_data.flatten()
        )
        
        drift_detected = p_value < threshold
        
        result = {
            "drift_detected": drift_detected,
            "p_value": float(p_value),
            "statistic": float(statistic),
            "reference_mean": float(self.reference_stats['mean']),
            "new_mean": float(new_stats['mean']),
            "reference_std": float(self.reference_stats['std']),
            "new_std": float(new_stats['std'])
        }
        
        if drift_detected:
            logger.warning(f"⚠️ Drift detected! p-value: {p_value:.4f}")
        
        return result
    
    def _compute_stats(self, data: np.ndarray) -> Dict:
        """Compute basic statistics."""
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data)
        }
