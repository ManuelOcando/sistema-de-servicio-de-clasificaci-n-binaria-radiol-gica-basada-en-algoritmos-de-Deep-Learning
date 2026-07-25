from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class IModelLoader(ABC):
    """Interface to load models."""

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """return class, confidence and data visualization."""
        pass


class IExplainer(ABC):
    """Interface to explainability."""

    @abstractmethod
    def generate_heatmap(self, image: np.ndarray, model: Any, target_layer: Any,
                         class_idx: int = None) -> np.ndarray:
        pass


class IImageProcessor(ABC):
    """Interface to p¿image processing."""

    @abstractmethod
    def decode_base64(self, base64_string: str) -> np.ndarray:
        pass

    @abstractmethod
    def encode_base64(self, image: np.ndarray) -> str:
        pass
