from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseWrapper(ABC):
    """Abstract base class for ML Model Wrappers"""

    @abstractmethod
    def predict(self, X: Any, **kwargs) -> Any:
        """Make predictions with the model"""
        pass

    @abstractmethod # Added new
    def predict_proba(self,X:Any, **kwargs) -> Any:
        """Get prediction probability"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk"""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        pass

    @abstractmethod
    def set_params(self, **params) -> None:
        """Set model parameters"""
        pass

    @abstractmethod
    def get_last_conv_layer(self):
        """Return the last convolutional layer for Grad-CAM usage"""
        pass
 
