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
class CrossCut:
    """Represents a 3D cross-cut through the onion."""
    normal: Tuple[float, float, float]  # Normal vector of the cutting plane
    point: Tuple[float, float, float]   # Point on the cutting plane
    
    def intersects_point(self, point: Tuple[float, float, float]) -> bool:
        """
        Check if a point is on the positive side of the cutting plane.
        
        Args:
            point: The point to check
            
        Returns:
            True if the point is on the positive side of the plane
        """
        # Calculate signed distance from point to plane
        distance = np.dot(
            np.array(point) - np.array(self.point),
            np.array(self.normal)
        )
        return distance > 0


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


class Cut(BaseCut):
    """2D cut model representing a straight line cut."""
    
    def __init__(self, start: Tuple[float, float], end: Tuple[float, float]):
        """
        Initialize a 2D cut.
        
        Args:
            start: Starting point (x, y) tuple
            end: Ending point (x, y) tuple
        """
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))

    def as_shapely(self) -> LineString:
        """
        Convert the cut to a Shapely LineString.
        
        Returns:
            Shapely LineString representation of the cut
        """
        return LineString([self.start, self.end])
    
    def __repr__(self) -> str:
        """
        String representation of the cut.
        
        Returns:
            String representation
        """
        return f"Cut(({self.start[0]:.3f}, {self.start[1]:.3f}), ({self.end[0]:.3f}, {self.end[1]:.3f}))"


class CrossCut(BaseCut):
    """3D cross-cut model representing a plane cutting through the onion."""
    
    def __init__(self, angle: float, distance_from_center: float = 0.0):
        """
        Create a cut across the onion in 3D space.
        
        Args:
            angle: Angle in radians from horizontal
            distance_from_center: Distance from center (0 = through center)
        """
        self.angle = angle
        self.distance_from_center = distance_from_center
    
    def as_plane(self) -> Tuple[float, float, float, float]:
        """
        Return a representation as a plane for 3D calculations.
        
        Returns:
            Plane equation coefficients (a, b, c, d) for ax + by + cz + d = 0
        """
        # Calculate the normal vector of the plane
        nx = np.sin(self.angle)
        ny = 0.0
        nz = np.cos(self.angle)
        
        # Calculate the d coefficient (distance from origin)
        d = -self.distance_from_center
        
        return (nx, ny, nz, d)
    
    def __repr__(self) -> str:
        """
        String representation of the cross-cut.
        
        Returns:
            String representation
        """
        return f"CrossCut(angle={self.angle}, distance={self.distance_from_center})" 