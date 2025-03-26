"""
3D onion model implementation with SVG profile support.
"""

from typing import List, Tuple, Dict, Optional
import os
import numpy as np
import xml.etree.ElementTree as ET
import re
import trimesh
from scipy import interpolate
from core.models import BaseOnion


class SVGPath:
    """Helper class to parse SVG path data."""
    
    @staticmethod
    def parse_path(d: str) -> List[Tuple[float, float]]:
        """
        Parse SVG path data string to extract points.
        
        Args:
            d: SVG path data string (e.g., "M100,100 L200,200...")
            
        Returns:
            List of (x, y) coordinate tuples
        """
        # Extract commands and their coordinates
        command_pattern = r'([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)'
        commands = re.findall(command_pattern, d)
        
        points = []
        current_point = (0, 0)  # Track current point for relative commands
        start_point = None      # Track starting point for Z command
        
        for cmd, coords_str in commands:
            # Extract numbers from coordinate string
            coords = [float(c) for c in re.findall(r'-?\d+\.?\d*', coords_str)]
            
            if cmd == 'M':  # MoveTo absolute
                x, y = coords[0], coords[1]
                points.append((x, y))
                current_point = (x, y)
                start_point = (x, y)  # Update start point
                
                # Handle subsequent pairs as LineTo commands
                for i in range(2, len(coords), 2):
                    x, y = coords[i], coords[i+1]
                    points.append((x, y))
                    current_point = (x, y)
            
            elif cmd == 'm':  # MoveTo relative
                x = current_point[0] + coords[0]
                y = current_point[1] + coords[1]
                points.append((x, y))
                current_point = (x, y)
                start_point = (x, y)  # Update start point
                
                # Handle subsequent pairs as LineTo commands
                for i in range(2, len(coords), 2):
                    x = current_point[0] + coords[i]
                    y = current_point[1] + coords[i+1]
                    points.append((x, y))
                    current_point = (x, y)
            
            elif cmd == 'L':  # LineTo absolute
                for i in range(0, len(coords), 2):
                    x, y = coords[i], coords[i+1]
                    points.append((x, y))
                    current_point = (x, y)
            
            elif cmd == 'l':  # LineTo relative
                for i in range(0, len(coords), 2):
                    x = current_point[0] + coords[i]
                    y = current_point[1] + coords[i+1]
                    points.append((x, y))
                    current_point = (x, y)
            
            elif cmd == 'H':  # Horizontal LineTo absolute
                for x in coords:
                    points.append((x, current_point[1]))
                    current_point = (x, current_point[1])
            
            elif cmd == 'h':  # Horizontal LineTo relative
                for dx in coords:
                    x = current_point[0] + dx
                    points.append((x, current_point[1]))
                    current_point = (x, current_point[1])
            
            elif cmd == 'V':  # Vertical LineTo absolute
                for y in coords:
                    points.append((current_point[0], y))
                    current_point = (current_point[0], y)
            
            elif cmd == 'v':  # Vertical LineTo relative
                for dy in coords:
                    y = current_point[1] + dy
                    points.append((current_point[0], y))
                    current_point = (current_point[0], y)
            
            elif cmd == 'Q':  # Quadratic Bezier absolute
                for i in range(0, len(coords), 4):
                    x1, y1 = coords[i], coords[i+1]     # Control point
                    x2, y2 = coords[i+2], coords[i+3]   # End point
                    # Add both control point and end point
                    points.append((x1, y1))
                    points.append((x2, y2))
                    current_point = (x2, y2)
            
            elif cmd in ('Z', 'z'):  # Close path
                if start_point is not None and not np.allclose(current_point, start_point):
                    points.append(start_point)  # Add the start point to close the path
                    current_point = start_point
        
        return points


class SVGProfile:
    """Represents a 2D profile of an onion layer from SVG."""
    
    def __init__(self, points: List[Tuple[float, float]], layer_index: int):
        """
        Initialize a profile with points.
        
        Args:
            points: List of (x, y) coordinate tuples defining the profile
            layer_index: Index of the layer this profile represents
        """
        self.original_points = points
        self.layer_index = layer_index
        
        # Process the points but preserve the x-coordinates
        self.normalized_points = self._normalize_points()
        
        # Store arrays for interpolation
        self.x_points = np.array([p[0] for p in self.normalized_points])
        self.y_points = np.array([p[1] for p in self.normalized_points])
    
    def _normalize_points(self) -> List[Tuple[float, float]]:
        """
        Normalize points to a 0-1 range while preserving the shape.
        
        Returns:
            Normalized points
        """
        if not self.original_points:
            return []
        
        # Find min and max values
        x_values = [p[0] for p in self.original_points]
        y_values = [p[1] for p in self.original_points]
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)
        
        # Calculate ranges
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        # Avoid division by zero
        if x_range < 1e-10:
            x_range = 1.0
        if y_range < 1e-10:
            y_range = 1.0
        
        # Normalize points to 0-1 range for both dimensions separately
        # This better preserves the real shape from the SVG
        normalized = []
        for x, y in self.original_points:
            norm_x = (x - min_x) / x_range
            norm_y = (y - min_y) / y_range
            normalized.append((norm_x, norm_y))
        
        # Sort by x-value to ensure proper interpolation
        normalized.sort(key=lambda p: p[0])
        
        return normalized
    
    def _parametric_interpolation(self, t_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform parametric interpolation of the profile points.
        
        Args:
            t_values: Parametric values between 0 and 1
            
        Returns:
            Tuple of (x_values, y_values) arrays
        """
        # If we have too few points, use linear interpolation
        if len(self.normalized_points) < 4:
            # Create parameter values for the original points
            t_orig = np.linspace(0, 1, len(self.normalized_points))
            
            # Linear interpolation
            x_interp = np.interp(t_values, t_orig, self.x_points)
            y_interp = np.interp(t_values, t_orig, self.y_points)
            
            return x_interp, y_interp
        
        # Use a parametric approach - treat the points as a path
        # First, create parameter values for the original points based on arc length
        # This gives better distribution of points along the curve
        
        # Calculate cumulative distance along the profile
        dists = [0]
        for i in range(1, len(self.normalized_points)):
            x1, y1 = self.normalized_points[i-1]
            x2, y2 = self.normalized_points[i]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            dists.append(dists[-1] + dist)
        
        # Normalize to 0-1 range
        if dists[-1] > 0:
            t_orig = [d / dists[-1] for d in dists]
        else:
            t_orig = np.linspace(0, 1, len(self.normalized_points))
        
        # Use 1D interpolation on the x and y points separately
        # Use PCHIP interpolator which preserves monotonicity and doesn't overshoot
        try:
            x_interp = interpolate.PchipInterpolator(t_orig, self.x_points)(t_values)
            y_interp = interpolate.PchipInterpolator(t_orig, self.y_points)(t_values)
        except ValueError:
            # Fallback to linear interpolation if PCHIP fails
            x_interp = np.interp(t_values, t_orig, self.x_points)
            y_interp = np.interp(t_values, t_orig, self.y_points)
        
        return x_interp, y_interp
    
    def get_interpolated_profile(self, resolution: int = 50) -> List[Tuple[float, float]]:
        """
        Get smoother profile by interpolating between points.
        
        Args:
            resolution: Number of points to generate
            
        Returns:
            List of interpolated (x, y) coordinate tuples
        """
        # If we have no points or only one point, return simplified result
        if len(self.normalized_points) == 0:
            return []
        elif len(self.normalized_points) == 1:
            return [self.normalized_points[0]] * resolution
        
        # Create parameter values evenly spaced
        t_values = np.linspace(0, 1, resolution)
        
        # Get interpolated x, y values
        x_values, y_values = self._parametric_interpolation(t_values)
        
        # Combine into tuples
        return list(zip(x_values, y_values))


class RealisticOnion(BaseOnion):
    """3D realistic onion model based on SVG profiles."""
    
    def __init__(self,
                 max_diameter: float,
                 profile_name: str,
                 center_axis_offset: float = 0.0):
        """
        Create a 3D onion model based on SVG profiles.
        
        Args:
            max_diameter: Maximum diameter of the widest part (in units)
            profile_name: Name of the selected real-life onion profile
            center_axis_offset: Offset of the center axis from 0 (default: 0.0)
        """
        self.max_diameter = max_diameter
        self.profile_name = profile_name
        self.center_axis_offset = center_axis_offset
        self.svg_profiles = self._load_svg_profiles(profile_name)
        
        # Ensure we have at least one profile
        if not self.svg_profiles:
            raise ValueError(f"No valid profiles found in {profile_name}.svg")
            
        self.n_layers = len(self.svg_profiles)
        self.scale_factor = self._calculate_scale_factor()
        
        # Call BaseOnion __init__ with dummy parameters (will be overridden)
        super().__init__(diameter=max_diameter, n_layers=self.n_layers)
    
    def _load_svg_profiles(self, profile_name: str) -> List[SVGProfile]:
        """
        Load Bezier curve paths from SVG files.
        
        Args:
            profile_name: Name of the SVG file (without extension)
            
        Returns:
            List of SVGProfile objects
        """
        svg_file = os.path.join('assets', f"{profile_name}.svg")
        if not os.path.exists(svg_file):
            raise FileNotFoundError(f"SVG profile not found: {svg_file}")
        
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        # Find all path elements in the SVG
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        
        profiles = []
        for i, path in enumerate(paths):
            d = path.get('d')
            if d:
                points = SVGPath.parse_path(d)
                if len(points) > 2:  # Only add if we have enough points
                    profiles.append(SVGProfile(points, i))
        
        # Sort by layer index - outermost layer (largest) should be first
        # This is assuming that in the SVG, layers are ordered from inside to outside
        # We reverse to have outer layers with higher indices
        profiles.sort(key=lambda p: p.layer_index)
        
        return profiles
    
    def _calculate_scale_factor(self) -> float:
        """
        Calculate the scale factor to achieve the desired max_diameter.
        
        Returns:
            Scale factor
        """
        # The max diameter should be applied to the outermost layer
        if not self.svg_profiles:
            return 1.0
            
        # Get the outermost profile (highest layer index)
        outermost_profile = self.svg_profiles[-1]
        
        # Find the maximum width of the profile
        points = outermost_profile.normalized_points
        x_values = [p[0] for p in points]
        max_width = max(x_values) - min(x_values)
        
        # Calculate scaling factor - avoid division by zero
        # Since we revolve around the Z-axis, the width becomes the diameter
        # So we need to scale to half the desired diameter
        return (self.max_diameter / 2) / max_width if max_width > 0 else 1.0
    
    def calculate_layer_radii(self) -> np.ndarray:
        """
        Calculate the radius of each layer based on SVG profiles.
        
        Returns:
            Array of radii for each layer
        """
        radii = [0]  # Center has radius 0
        
        for profile in self.svg_profiles:
            # Find maximum width in the profile (which represents the diameter)
            interpolated = profile.get_interpolated_profile()
            x_values = [p[0] for p in interpolated]
            max_diameter = max(x_values) - min(x_values)
            radius = (max_diameter / 2) * self.scale_factor
            radii.append(radius)
        
        return np.array(radii)
    
    def create_layer_boundaries(self) -> List[np.ndarray]:
        """
        Create the boundaries between layers using the revolved SVG profiles.
        
        Returns:
            List of boundaries representing each layer
        """
        boundaries = []
        
        for profile in self.svg_profiles:
            interpolated = profile.get_interpolated_profile(resolution=100)
            scaled_profile = [(x * self.scale_factor, y * self.scale_factor) 
                              for x, y in interpolated]
            boundaries.append(np.array(scaled_profile))
        
        return boundaries
    
    def revolve_profile(self, profile: SVGProfile, resolution: int = 36) -> trimesh.Trimesh:
        """
        Revolve a 2D profile around the Z-axis to create a 3D mesh.
        
        Args:
            profile: The SVGProfile to revolve
            resolution: Number of segments for the revolution
            
        Returns:
            3D mesh created by revolving the profile
        """
        # Get interpolated and scaled profile points
        interpolated = profile.get_interpolated_profile(resolution=100)
        scaled_profile = [(x * self.scale_factor, y * self.scale_factor) 
                          for x, y in interpolated]
        
        # Convert to numpy array for faster computation
        profile_points = np.array(scaled_profile)
        
        # Ensure the profile is properly closed
        if not np.allclose(profile_points[0], profile_points[-1]):
            # Add the first point at the end to close the loop
            profile_points = np.vstack([profile_points, profile_points[0]])
        
        # Ensure we have enough points for a valid mesh
        if len(profile_points) < 3:
            raise ValueError("Profile must have at least 3 points to create a valid mesh")
        
        n_profile_points = len(profile_points)
        
        # Create vertices by revolving the profile
        vertices = []
        for i in range(resolution):
            angle = 2 * np.pi * i / resolution
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            for x, y in profile_points:
                # Apply offset if needed
                x_offset = x - self.center_axis_offset
                
                # Revolve point around Z-axis
                px = x_offset * cos_angle
                py = x_offset * sin_angle
                pz = y
                vertices.append([px, py, pz])
        
        vertices = np.array(vertices)
        
        # Create faces by connecting adjacent vertices
        faces = []
        for i in range(resolution):
            next_i = (i + 1) % resolution
            
            for j in range(n_profile_points - 1):
                # Calculate vertex indices for the quad
                v1 = i * n_profile_points + j
                v2 = i * n_profile_points + j + 1
                v3 = next_i * n_profile_points + j + 1
                v4 = next_i * n_profile_points + j
                
                # Create two triangles for the quad
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        # Create end caps
        # Add center points for caps
        top_center = np.mean(vertices[::n_profile_points], axis=0)
        bottom_center = np.mean(vertices[n_profile_points-1::n_profile_points], axis=0)
        
        # Add center points to vertices
        vertices = np.vstack([vertices, top_center, bottom_center])
        top_center_idx = len(vertices) - 2
        bottom_center_idx = len(vertices) - 1
        
        # Add triangles for end caps
        for i in range(resolution):
            next_i = (i + 1) % resolution
            # Top cap
            v1 = i * n_profile_points
            v2 = next_i * n_profile_points
            faces.append([top_center_idx, v1, v2])
            
            # Bottom cap
            v3 = i * n_profile_points + (n_profile_points - 1)
            v4 = next_i * n_profile_points + (n_profile_points - 1)
            faces.append([bottom_center_idx, v4, v3])
        
        faces = np.array(faces)
        
        # Create the mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Fix normals and merge vertices to ensure watertightness
        mesh.fix_normals()
        mesh.merge_vertices()
        
        # Verify watertightness
        if not mesh.is_watertight:
            # Try to fix the mesh
            mesh.fill_holes()
            # Update faces to remove degenerate and duplicate faces
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.update_faces(mesh.unique_faces())
            mesh.merge_vertices()
        
        return mesh
    
    def generate_3d_model(self) -> trimesh.Trimesh:
        """
        Generate a complete 3D model of the onion with all layers.
        
        Returns:
            3D mesh of the complete onion with distinct layer information
        """
        if not self.svg_profiles:
            return trimesh.Trimesh()
        
        # Generate meshes for each layer
        layer_meshes = []
        for i, profile in enumerate(self.svg_profiles):
            mesh = self.revolve_profile(profile)
            
            # Store layer index in mesh metadata
            mesh.metadata = {
                'layer_index': i,
                'is_outer_surface': i == len(self.svg_profiles) - 1
            }
            
            # Set visual properties for the layer
            # Outer layer is green, inner layers are increasingly transparent
            alpha = 0.3 + (0.7 * i / len(self.svg_profiles))
            if i == len(self.svg_profiles) - 1:
                # Outer layer - green
                mesh.visual.face_colors = [0, 255, 0, int(255 * alpha)]
            else:
                # Inner layers - yellow with increasing transparency
                mesh.visual.face_colors = [255, 255, 0, int(255 * alpha)]
            
            layer_meshes.append(mesh)
        
        # Combine all meshes into one, preserving layer information
        combined_mesh = trimesh.util.concatenate(layer_meshes)
        
        # Store the number of layers in metadata
        combined_mesh.metadata = {
            'n_layers': len(self.svg_profiles),
            'layer_vertices': [len(mesh.vertices) for mesh in layer_meshes],
            'layer_faces': [len(mesh.faces) for mesh in layer_meshes]
        }
        
        return combined_mesh 