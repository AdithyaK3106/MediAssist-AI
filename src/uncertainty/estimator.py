import numpy as np
from typing import Dict

class UncertaintyEstimator:
    """Safety module to detect out-of-distribution or ambiguous inputs."""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate prediction entropy for confidence measurement."""
        return -np.sum(probabilities * np.log(probabilities + 1e-12))

    def is_reliable(self, probabilities: np.ndarray) -> bool:
        """Check if the model prediction meets safety standards."""
        entropy = self.calculate_entropy(probabilities)
        return entropy < self.threshold
