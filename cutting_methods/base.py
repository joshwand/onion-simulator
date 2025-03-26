"""
Abstract base class for cutting methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from core.models import BaseOnion
from models.cut import Cut, CrossCut


class CuttingMethod(ABC):
    """
    Abstract base class for cutting methods.
    Implements the Strategy pattern for interchangeable cutting algorithms.
    """
    
    def __init__(self, onion: BaseOnion):
        """
        Initialize the cutting method.
        
        Args:
            onion: The onion model to apply the cutting method to
        """
        self.onion = onion
    
    @abstractmethod
    def generate_cuts(self, **params) -> List[Cut]:
        """
        Generate 2D cuts based on the method's algorithm.
        
        Args:
            **params: Method-specific parameters
            
        Returns:
            List of Cut objects
        """
        pass
    
    
    
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        """
        Get the default parameters for the cutting method.
        
        Returns:
            Dictionary of default parameters
        """
        return {}
    
    @classmethod
    def from_params(cls, onion: BaseOnion, params: Dict[str, Any]) -> 'CuttingMethod':
        """
        Create a cutting method from a dictionary of parameters.
        
        Args:
            onion: The onion model
            params: Dictionary of parameters
            
        Returns:
            Initialized cutting method
        """
        return cls(onion)


class CuttingMethodFactory:
    """
    Factory class for creating cutting methods.
    Implements the Factory pattern.
    """
    
    _methods = {}
    
    @classmethod
    def register(cls, name: str, method_class: type):
        """
        Register a cutting method.
        
        Args:
            name: The name of the method
            method_class: The cutting method class
        """
        cls._methods[name] = method_class
    
    @classmethod
    def create_method(cls, method_name: str, onion: BaseOnion, **params) -> CuttingMethod:
        """
        Create a cutting method.
        
        Args:
            method_name: The name of the method to create
            onion: The onion model
            **params: Method-specific parameters
            
        Returns:
            Initialized cutting method
            
        Raises:
            ValueError: If the method name is not recognized
        """
        if method_name not in cls._methods:
            raise ValueError(f"Unknown cutting method: {method_name}")
        
        method_class = cls._methods[method_name]
        return method_class(onion) 