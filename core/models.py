"""
Core data models shared between 2D and 3D simulators.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import numpy as np
import scipy.interpolate as interp


class BaseOnion(ABC):
    """Abstract base class for onion models."""
    
    def __init__(self, 
                 diameter: float, 
                 n_layers: int, 
                 start_y: float = 0.5, 
                 end_y: float = 0.8, 
                 p1: Tuple[float, float] = (0.2, 1.2), 
                 p2: Tuple[float, float] = (0.8, 1.0)):
        """
        Initialize the base onion model.
        
        Args:
            diameter: The diameter of the onion in inches
            n_layers: The number of layers in the onion
            start_y: The starting y-value for the thickness curve (center)
            end_y: The ending y-value for the thickness curve (edge)
            p1: The first control point for the spline curve (x, y)
            p2: The second control point for the spline curve (x, y)
        """
        self.radius = diameter / 2
        self.n_layers = n_layers
        self.start_y = start_y
        self.end_y = end_y
        self.p1 = p1
        self.p2 = p2
        self.layer_radii = self.calculate_layer_radii()
    
    def layer_thickness_curve(self, x: float) -> float:
        """
        Calculate the thickness at a normalized radius using a cubic spline.
        
        Args:
            x: Normalized radius (0 to 1)
            
        Returns:
            Relative thickness of the layer
        """
        points = [(0, self.start_y), self.p1, self.p2, (1.0, self.end_y)]
        x_values, y_values = zip(*points)
        spline = interp.CubicSpline(x_values, y_values)
        return spline(x)
    
    @abstractmethod
    def calculate_layer_radii(self) -> np.ndarray:
        """
        Calculate the radius of each layer.
        
        Returns:
            Array of radii for each layer
        """
        pass
    
    @abstractmethod
    def create_layer_boundaries(self) -> List[Any]:
        """
        Create the boundaries between layers.
        
        Returns:
            List of boundary representations (implementation-specific)
        """
        pass


class BaseCut(ABC):
    """Abstract base class for cuts."""
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the cut."""
        pass 