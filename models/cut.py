"""
Cut and piece models for both 2D and 3D onion simulators.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from core.models import BaseCut


@dataclass
class Cut:
    """Represents a 2D cut through the onion."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    
    @property
    def length(self) -> float:
        """Calculate the length of the cut."""
        return np.sqrt((self.end[0] - self.start[0])**2 + (self.end[1] - self.start[1])**2)
    
    def as_shapely(self) -> LineString:
        """Convert the cut to a Shapely LineString."""
        return LineString([self.start, self.end])


@dataclass
class CrossCut(BaseCut):
    """Represents a 3D cross-cut through the onion."""
    normal: Tuple[float, float, float]  # Normal vector of the cutting plane
    point: Tuple[float, float, float]   # Point on the cutting plane
    y_max: Optional[float] = None       # Maximum y value for the cut (None = no limit)
    
    @staticmethod
    def from_z_offset(z_offset: float, y_max: Optional[float] = None) -> 'CrossCut':
        """
        Create a CrossCut from a z-offset value.
        
        Args:
            z_offset: Distance from the center along the z-axis where the cut occurs
            y_max: Maximum y value for the cut (None = no limit, 0 = bottom half only)
            
        Returns:
            CrossCut instance
        """
        # All cuts are vertical (parallel to XZ plane), so normal is always along Z axis
        normal = (0.0, 0.0, 1.0)
        # Point is just the z-offset along the z-axis
        point = (0.0, 0.0, z_offset)
        return CrossCut(normal=normal, point=point, y_max=y_max)
    
    def intersects_point(self, point: Tuple[float, float, float]) -> bool:
        """
        Check if a point is on the positive side of the cutting plane
        and within the y-limit if specified.
        
        Args:
            point: The point to check
            
        Returns:
            True if the point is on the positive side of the plane and within y-limit
        """
        # First check y-limit if specified
        if self.y_max is not None and point[1] > self.y_max:
            return False
            
        # Calculate signed distance from point to plane
        distance = np.dot(
            np.array(point) - np.array(self.point),
            np.array(self.normal)
        )
        return distance > 0
    
    def as_plane(self) -> Tuple[float, float, float, float]:
        """
        Return a representation as a plane for 3D calculations.
        For a vertical cut parallel to XZ plane, the equation is z = z_offset,
        which in plane form is: 0*x + 0*y + 1*z - z_offset = 0
        
        Returns:
            Plane equation coefficients (a, b, c, d) for ax + by + cz + d = 0
        """
        # Calculate d coefficient (negative dot product of normal and point)
        d = -(self.normal[0] * self.point[0] + 
              self.normal[1] * self.point[1] + 
              self.normal[2] * self.point[2])
        return (self.normal[0], self.normal[1], self.normal[2], d)


class OnionPiece3D:
    """Represents a 3D piece of onion after cutting."""
    
    def __init__(self, geometry: Dict[str, Any]):
        """
        Initialize a 3D onion piece.
        
        Args:
            geometry: Dictionary containing piece properties
        """
        self.id = geometry['id']
        self.volume = geometry['volume']
        self.external_area = geometry['external_area']
        self.layer_area = geometry['layer_area']
        self.cut_area = geometry['cut_area']
        self.layer_index = geometry.get('layer_index', 0)
        self.piece_2d = geometry.get('piece_2d')
    
    @property
    def surface_areas(self) -> Dict[str, float]:
        """
        Get the surface areas for different types of faces.
        
        Returns:
            Dictionary with surface areas for different types of faces
        """
        return {
            'external': self.external_area,
            'layer': self.layer_area,
            'cut': self.cut_area,
            'total': self.external_area + self.layer_area + self.cut_area
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the piece to a dictionary.
        
        Returns:
            Dictionary representation of the piece
        """
        return {
            'id': self.id,
            'volume': self.volume,
            'external_area': self.external_area,
            'layer_area': self.layer_area,
            'cut_area': self.cut_area,
            'layer_index': self.layer_index,
            'surface_areas': self.surface_areas
        } 