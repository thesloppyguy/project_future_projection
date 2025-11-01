"""
Base classes and utilities for multiple forecasting models.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Base class for all forecasting models."""
    
    def __init__(self, model_name: str, random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> 'BaseForecaster':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def save(self, path: Path):
        """Save the model."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: Path):
        """Load the model."""
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
            self.is_fitted = True

