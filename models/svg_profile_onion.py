"""
SVG Profile-based 3D onion model.
"""

import os
import numpy as np
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET
from models.svg_profile import SVGProfile
from utils.svg_parser import extract_paths_from_svg, normalize_paths

class SvgProfileOnion:
    """3D onion model based on SVG profiles."""
    
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
        # Properly calculate radius based on the provided diameter
        self.radius = max_diameter / 2  # Add radius property
        self.profile_name = profile_name
        self.center_axis_offset = center_axis_offset
        
        # Load and normalize SVG profiles
        self.svg_profiles = self._load_svg_profiles()
        if not self.svg_profiles:
            raise ValueError(f"No valid profiles found in {profile_name}.svg")
            
        # Calculate dimensions and scaling
        self.scale_factor = self._calculate_scale_factor()
        self.max_height, self.max_width = self._calculate_dimensions()
        
        # Add n_layers property
        self.n_layers = len(self.svg_profiles)
    
    def __repr__(self) -> str:
        return f"SvgProfileOnion(max_diameter={self.max_diameter}, profile_name={self.profile_name})"

    def _load_svg_profiles(self) -> List[SVGProfile]:
        """Load and normalize SVG profiles."""
        svg_file = os.path.join('assets', f"{self.profile_name}.svg")
        if not os.path.exists(svg_file):
            raise FileNotFoundError(f"SVG profile not found: {svg_file}")
        
        # Extract paths from SVG
        paths = extract_paths_from_svg(svg_file)
        if not paths:
            raise ValueError(f"No valid paths found in {svg_file}")
        
        # Normalize paths to align them consistently
        normalized_paths = normalize_paths(paths)
        
        # Convert to SVGProfile objects
        profiles = []
        for i, path in enumerate(normalized_paths):
            profiles.append(SVGProfile(path, name=f"Layer_{i}"))
        
        return profiles
    
    def _calculate_scale_factor(self) -> float:
        """Calculate scaling factor to achieve desired max diameter."""
        if not self.svg_profiles:
            return 1.0
            
        # Find maximum width across all profiles- this is the radius of the onion, as 
        # the profiles in the SVG are a *quarter* of the onion (longitudinal cuts)
        max_width = 0
        for profile in self.svg_profiles:
            (min_x, _), (max_x, _) = profile.get_bounding_box()
            width = max_x - min_x
            max_width = max(max_width, width)
        
        # print(f"SvgProfileOnion._calculate_scale_factor: max_width: {max_width}")
        # print(f"SvgProfileOnion._calculate_scale_factor: max_diameter: {self.max_diameter}")
        # Scale to achieve desired diameter, ensuring max_width is interpreted correctly
        # The SVG width becomes the diameter when revolved around the Z-axis
        return self.max_diameter / max_width * 0.5 if max_width > 0 else 1.0
    
    def _calculate_dimensions(self) -> Tuple[float, float]:
        """Calculate maximum height and width after scaling."""
        if not self.svg_profiles:
            return 0.0, 0.0
            
        max_height = 0
        max_width = 0
        
        for profile in self.svg_profiles:
            (min_x, min_y), (max_x, max_y) = profile.get_bounding_box()
            height = max_y - min_y
            width = max_x - min_x
            max_height = max(max_height, height)
            max_width = max(max_width, width)
        
        return max_height * self.scale_factor, max_width * self.scale_factor
    
    def generate_mesh(self, resolution: int = 36) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a 3D mesh by revolving the profiles 180 degrees around the Z-axis.
        
        Args:
            resolution: Number of points around the half-circumference
            
        Returns:
            Tuple of (vertices, faces, normals, layer_indices)
            where layer_indices is an array indicating which layer each vertex belongs to
        """
        if not self.svg_profiles:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Generate points around the half-circumference (180 degrees)
        theta = np.linspace(0, np.pi, resolution)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        vertices = []
        faces = []
        normals = []
        layer_indices = []  # Track which layer each vertex belongs to
        vertex_offset = 0
        
        # Process each layer profile
        for layer_idx, profile in enumerate(self.svg_profiles):
            # Get interpolated profile points with proper Bezier curve interpolation
            profile_points = profile.get_interpolated_profile(resolution=35)  # Higher resolution for smoother curves
            if not isinstance(profile_points, np.ndarray):
                profile_points = np.array(profile_points)
            scaled_points = profile_points * self.scale_factor
            
            # Create vertices by revolving the profile
            layer_vertices = []
            for c, s in zip(cos_theta, sin_theta):
                for x, y in scaled_points:
                    # Apply center axis offset
                    x_offset = x - self.center_axis_offset
                    
                    # Note: SVG Y becomes Z in 3D space
                    vx = x_offset * c
                    vy = x_offset * s  # Only positive Y values for half-onion
                    vz = y  # Y coordinate becomes Z in 3D space
                    layer_vertices.append([vx, vy, vz])
                    layer_indices.append(layer_idx)  # Track layer index for this vertex
            
            layer_vertices = np.array(layer_vertices)
            vertices.extend(layer_vertices)
            
            # Generate faces for this layer
            n_profile_points = len(scaled_points)
            
            # Generate faces for the curved surface
            for i in range(resolution - 1):  # -1 because we don't wrap around for half-onion
                for j in range(n_profile_points - 1):
                    # Calculate vertex indices for the quad
                    v1 = vertex_offset + i * n_profile_points + j
                    v2 = vertex_offset + i * n_profile_points + (j + 1)
                    v3 = vertex_offset + (i + 1) * n_profile_points + (j + 1)
                    v4 = vertex_offset + (i + 1) * n_profile_points + j
                    
                    # Create two triangles for the quad
                    faces.append([v1, v2, v3])  # First triangle
                    faces.append([v1, v3, v4])  # Second triangle
                    
                    # Calculate face normals
                    p1 = vertices[v1]
                    p2 = vertices[v2]
                    p3 = vertices[v3]
                    
                    # Calculate normal using cross product
                    edge1 = p2 - p1
                    edge2 = p3 - p1
                    normal = np.cross(edge1, edge2)
                    if np.any(normal):  # Check if normal is non-zero
                        normal = normal / np.linalg.norm(normal)
                    else:
                        normal = np.array([c, s, 0])  # Fallback normal
                    
                    normals.extend([normal, normal])  # Same normal for both triangles
            
            # Generate faces for the flat surfaces at theta = 0 and theta = pi
            for theta_idx in [0, resolution - 1]:
                base_idx = vertex_offset + theta_idx * n_profile_points
                for j in range(n_profile_points - 1):
                    v1 = base_idx + j
                    v2 = base_idx + j + 1
                    
                    # Create triangles for the flat surface
                    if theta_idx == 0:
                        faces.append([v1, v2, v2 + n_profile_points])
                        faces.append([v1, v2 + n_profile_points, v1 + n_profile_points])
                        normal = np.array([0, -1, 0])  # Normal points in negative Y direction
                    else:
                        faces.append([v1, v2, v2 - n_profile_points])
                        faces.append([v1, v2 - n_profile_points, v1 - n_profile_points])
                        normal = np.array([0, 1, 0])  # Normal points in positive Y direction
                    
                    normals.extend([normal, normal])
            
            vertex_offset += len(layer_vertices)
        
        return np.array(vertices), np.array(faces), np.array(normals), np.array(layer_indices) 