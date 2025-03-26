"""
2D onion model implementation.
"""

from typing import Tuple, List
import numpy as np
from shapely.geometry import Point, Polygon
from core.models import BaseOnion


class HalfOnion(BaseOnion):
    """2D half-onion model (semi-circle with layers)."""
    
    def calculate_layer_radii(self) -> np.ndarray:
        """
        Calculate the radius of each layer.
        
        Returns:
            Array of radii for each layer
        """
        radii = [0]
        total_thickness = sum(self.layer_thickness_curve(i / self.n_layers) for i in range(1, self.n_layers + 1))
        scale_factor = self.radius / total_thickness
        
        current_radius = 0
        for i in range(1, self.n_layers + 1):
            x = i / self.n_layers
            thickness = self.layer_thickness_curve(x) * scale_factor
            current_radius += thickness
            radii.append(current_radius)
        
        return np.array(radii)
    
    def create_layer_boundaries(self) -> List:
        """
        Create the boundaries between layers using Shapely geometries.
        
        Returns:
            List of LineString objects representing the layer boundaries
        """
        # Create a semicircle for each layer
        half_circle = Point(0, 0).buffer(self.radius).intersection(
            Polygon([(-self.radius, 0), (self.radius, 0), (self.radius, self.radius), (-self.radius, self.radius)])
        )
        
        return [
            Point(0, 0).buffer(r).boundary.intersection(half_circle) 
            for r in self.layer_radii[1:]
        ] 