"""
Abstract base class for cutting methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

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
    
    def generate_cross_cuts(self, n_cross_cuts: int = 0) -> List[CrossCut]:
        """
        Generate 3D cross-cuts as parallel vertical planes (like slicing bread).
        The cuts are made in the XZ plane, starting from z=0 and moving down to max_height,
        but only in the bottom half of the onion (y≤0).
        
        This implementation is common across all cutting methods as cross-cuts
        are independent of the specific cutting algorithm.
        
        Args:
            n_cross_cuts: Number of cross-cuts to generate
            
        Returns:
            List of CrossCut objects
        """
        cross_cuts = []
        
        if n_cross_cuts > 0:
            # Get the height of the onion
            height = self.onion.max_height
            
            # Calculate spacing between cuts
            spacing = height / (n_cross_cuts + 1)
            
            # Generate evenly spaced vertical cuts from top to bottom
            # but only in the bottom half (y≤0)
            for i in range(n_cross_cuts):
                # Calculate z offset for this cut
                # Start at 0 and move down to max_height
                z_offset = spacing * (i + 1)
                # Create a cut that only exists in the bottom half (y≤0)
                cross_cuts.append(CrossCut.from_z_offset(-z_offset, y_max=0))  # y_max=0 limits to bottom half
        
        return cross_cuts
    
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        """
        Get the default parameters for the cutting method.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'n_cross_cuts': 6,
        }
    
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