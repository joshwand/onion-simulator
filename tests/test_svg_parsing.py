"""
Tests for SVG parsing functionality.
"""

import unittest
import os
import sys
from typing import List, Tuple
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.svg_parser import parse_path_data, extract_profiles_from_svg, normalize_profile
from models.onion_3d import SVGPath, SVGProfile, RealisticOnion


class TestSVGParsing(unittest.TestCase):
    """Test cases for SVG parsing functionality."""
    
    def test_parse_path_data(self):
        """Test parsing SVG path data."""
        # Simple path with M and L commands
        path_data = "M100,100 L200,200 L300,100"
        points = parse_path_data(path_data)
        
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0], (100.0, 100.0))
        self.assertEqual(points[1], (200.0, 200.0))
        self.assertEqual(points[2], (300.0, 100.0))
        
        # Complex path with bezier curves
        path_data = "M10,10 C30,30 40,40 50,50 Q60,60 70,70"
        points = parse_path_data(path_data)
        
        self.assertEqual(len(points), 6)
        self.assertEqual(points[0], (10.0, 10.0))
        self.assertEqual(points[5], (70.0, 70.0))
        
        # Path with negative values
        path_data = "M-10,-10 L-20,30 L40,-50"
        points = parse_path_data(path_data)
        
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0], (-10.0, -10.0))
        self.assertEqual(points[1], (-20.0, 30.0))
        self.assertEqual(points[2], (40.0, -50.0))
    
    def test_normalize_profile(self):
        """Test normalizing profile points."""
        # Simple profile
        points = [(100.0, 200.0), (200.0, 300.0), (300.0, 100.0)]
        normalized = normalize_profile(points)
        
        self.assertEqual(len(normalized), 3)
        self.assertEqual(normalized[0][0], 0.0)  # Min X = 0
        self.assertEqual(normalized[2][0], 1.0)  # Max X = 1
        
        # Make sure points are sorted by X
        self.assertTrue(all(normalized[i][0] <= normalized[i+1][0] for i in range(len(normalized)-1)))
        
        # Empty profile
        self.assertEqual(normalize_profile([]), [])
        
        # Single point profile
        single_point = [(42.0, 42.0)]
        self.assertEqual(normalize_profile(single_point), [(0.5, 42.0)])
    
    def test_svg_path_class(self):
        """Test the SVGPath class."""
        path_data = "M100,100 L200,200 L300,100"
        points = SVGPath.parse_path(path_data)
        
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0], (100.0, 100.0))
        self.assertEqual(points[1], (200.0, 200.0))
        self.assertEqual(points[2], (300.0, 100.0))
    
    def test_svg_profile_class(self):
        """Test the SVGProfile class."""
        points = [(100.0, 200.0), (200.0, 300.0), (300.0, 100.0)]
        profile = SVGProfile(points, layer_index=1)
        
        self.assertEqual(profile.layer_index, 1)
        self.assertEqual(profile.original_points, points)
        
        normalized = profile.normalized_points
        self.assertEqual(len(normalized), 3)
        self.assertEqual(normalized[0][0], 0.0)  # Min X = 0
        self.assertEqual(normalized[2][0], 1.0)  # Max X = 1
        
        # Test interpolation
        interpolated = profile.get_interpolated_profile(resolution=5)
        self.assertEqual(len(interpolated), 5)
        self.assertEqual(interpolated[0][0], 0.0)  # First point X = 0
        self.assertEqual(interpolated[-1][0], 1.0)  # Last point X = 1
    
    def test_extract_profiles_from_svg(self):
        """Test extracting profiles from an SVG file."""
        # This test requires an actual SVG file
        svg_file = os.path.join('assets', 'onion_debug.svg')
        
        # Skip test if file doesn't exist
        if not os.path.exists(svg_file):
            self.skipTest(f"SVG file not found: {svg_file}")
        
        profiles = extract_profiles_from_svg(svg_file)
        
        # onion_debug.svg should have at least one path
        self.assertGreater(len(profiles), 0)
        self.assertGreater(len(profiles[0]), 0)
    
    def test_realistic_onion_create(self):
        """Test creating a RealisticOnion model."""
        # This test requires an actual SVG file
        profile_name = 'onion1'
        svg_file = os.path.join('assets', f"{profile_name}.svg")
        
        # Skip test if file doesn't exist
        if not os.path.exists(svg_file):
            self.skipTest(f"SVG file not found: {svg_file}")
        
        try:
            onion = RealisticOnion(
                max_diameter=5.0,
                profile_name=profile_name
            )
            
            self.assertEqual(onion.max_diameter, 5.0)
            self.assertEqual(onion.profile_name, profile_name)
            self.assertGreater(len(onion.svg_profiles), 0)
            self.assertGreater(onion.scale_factor, 0)
            
            # Test layer radii calculation
            layer_radii = onion.calculate_layer_radii()
            self.assertEqual(len(layer_radii), len(onion.svg_profiles) + 1)  # +1 for center radius 0
            self.assertEqual(layer_radii[0], 0)  # Center radius should be 0
            
            # Test layer boundaries
            boundaries = onion.create_layer_boundaries()
            self.assertEqual(len(boundaries), len(onion.svg_profiles))
            
        except Exception as e:
            self.fail(f"Failed to create RealisticOnion: {str(e)}")


if __name__ == "__main__":
    unittest.main() 