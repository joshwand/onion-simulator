import numpy as np
from typing import List, Tuple, Optional
from matplotlib.path import Path
import scipy.special

class SVGProfile:
    """A class representing a 2D profile from an SVG path."""
    
    def __init__(self, path: Path, name: str = ""):
        """
        Initialize an SVG profile.
        
        Args:
            path: Matplotlib Path object containing the SVG path data
            name: Optional name for the profile
        """
        self.path = path
        self.name = name
    
    def get_interpolated_profile(self, resolution: int = 100) -> np.ndarray:
        """
        Get an interpolated version of the profile with proper Bezier curve interpolation.
        
        Args:
            resolution: Number of points in the interpolated profile
            
        Returns:
            Array of (x, y) coordinates for the interpolated profile
        """
        vertices = self.path.vertices
        codes = self.path.codes
        
        if codes is None:
            return vertices
        
        interpolated_points = []
        current_point = None
        
        # Parameter t for Bezier curves
        t = np.linspace(0, 1, resolution)
        
        i = 0
        while i < len(codes):
            code = codes[i]
            
            if code == Path.MOVETO:
                current_point = vertices[i]
                interpolated_points.append(current_point)
                i += 1
            
            elif code == Path.LINETO:
                start = current_point
                end = vertices[i]
                # Linear interpolation
                x = np.linspace(start[0], end[0], resolution)
                y = np.linspace(start[1], end[1], resolution)
                interpolated_points.extend(np.column_stack([x, y]))
                current_point = end
                i += 1
            
            elif code == Path.CURVE3:  # Quadratic Bezier
                p0 = current_point
                p1 = vertices[i]
                p2 = vertices[i + 1]
                
                # Quadratic Bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                basis = np.array([
                    (1 - t) ** 2,
                    2 * (1 - t) * t,
                    t ** 2
                ])
                
                points = np.array([p0, p1, p2])
                curve_points = np.dot(basis.T, points)
                interpolated_points.extend(curve_points)
                
                current_point = p2
                i += 2
            
            elif code == Path.CURVE4:  # Cubic Bezier
                p0 = current_point
                p1 = vertices[i]
                p2 = vertices[i + 1]
                p3 = vertices[i + 2]
                
                # Cubic Bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
                basis = np.array([
                    (1 - t) ** 3,
                    3 * (1 - t) ** 2 * t,
                    3 * (1 - t) * t ** 2,
                    t ** 3
                ])
                
                points = np.array([p0, p1, p2, p3])
                curve_points = np.dot(basis.T, points)
                interpolated_points.extend(curve_points)
                
                current_point = p3
                i += 3
            
            elif code == Path.CLOSEPOLY:
                if not np.array_equal(current_point, interpolated_points[0]):
                    # Connect back to the first point
                    start = current_point
                    end = interpolated_points[0]
                    x = np.linspace(start[0], end[0], resolution)
                    y = np.linspace(start[1], end[1], resolution)
                    interpolated_points.extend(np.column_stack([x, y]))
                break
            
            else:
                i += 1
        
        return np.array(interpolated_points)
    
    def scale(self, scale_factor: float) -> 'SVGProfile':
        """
        Return a new profile scaled by the given factor.
        
        Args:
            scale_factor: Factor to scale the profile by
            
        Returns:
            A new scaled SVGProfile
        """
        scaled_vertices = self.path.vertices * scale_factor
        new_path = Path(scaled_vertices, self.path.codes)
        return SVGProfile(new_path, self.name)
    
    def translate(self, dx: float, dy: float) -> 'SVGProfile':
        """
        Return a new profile translated by the given amounts.
        
        Args:
            dx: Amount to translate in x direction
            dy: Amount to translate in y direction
            
        Returns:
            A new translated SVGProfile
        """
        translated_vertices = self.path.vertices + np.array([dx, dy])
        new_path = Path(translated_vertices, self.path.codes)
        return SVGProfile(new_path, self.name)
    
    def rotate(self, angle_degrees: float) -> 'SVGProfile':
        """
        Return a new profile rotated by the given angle around the origin.
        
        Args:
            angle_degrees: Angle to rotate by in degrees
            
        Returns:
            A new rotated SVGProfile
        """
        angle_rad = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated_vertices = self.path.vertices @ rotation_matrix
        new_path = Path(rotated_vertices, self.path.codes)
        return SVGProfile(new_path, self.name)
    
    def is_closed(self) -> bool:
        """
        Check if the profile forms a closed loop.
        
        Returns:
            True if the profile is closed, False otherwise
        """
        if self.path.codes is not None and Path.CLOSEPOLY in self.path.codes:
            return True
        return np.array_equal(self.path.vertices[0], self.path.vertices[-1])
    
    def get_bounding_box(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get the bounding box of the profile.
        
        Returns:
            ((min_x, min_y), (max_x, max_y))
        """
        vertices = self.path.vertices
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        return (tuple(min_coords), tuple(max_coords)) 