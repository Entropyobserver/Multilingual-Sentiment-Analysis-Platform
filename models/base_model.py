from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def forward(self, **kwargs):
        """Forward pass"""
        pass
    
    @abstractmethod
    def save_model(self, save_path: str):
        """Save model to specified path"""
        pass
    
    @abstractmethod
    def load_model(self, load_path: str):
        """Load model from specified path"""
        pass
    
    @abstractmethod
    def get_params_info(self) -> Dict[str, Any]:
        """Get model parameters information"""
        pass
    
    @abstractmethod
    def configure_for_training(self):
        """Configure model for training"""
        pass
    
    @abstractmethod
    def configure_for_evaluation(self):
        """Configure model for evaluation"""
        pass