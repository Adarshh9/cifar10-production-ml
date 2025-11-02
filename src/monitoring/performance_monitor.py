"""Monitor model performance over time."""
from collections import deque
from datetime import datetime
import numpy as np
from typing import Dict

class PerformanceMonitor:
    """Track model performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def record_prediction(self, confidence: float, class_id: int):
        """Record a prediction."""
        self.predictions.append(class_id)
        self.confidences.append(confidence)
        self.timestamps.append(datetime.now())
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        if not self.confidences:
            return {}
        
        confidences = np.array(self.confidences)
        
        return {
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "low_confidence_rate": float(np.mean(confidences < 0.7)),
            "predictions_count": len(self.predictions),
            "class_distribution": self._get_class_distribution()
        }
    
    def _get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of predictions across classes."""
        from collections import Counter
        return dict(Counter(self.predictions))
