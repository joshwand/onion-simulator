import unittest
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.onion_3d import RealisticOnion, SVGProfile
from services.visualization_service import VisualizationService
import os

class TestVisualizationService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a test SVG file
        os.makedirs('assets', exist_ok=True)
        with open('assets/test_onion.svg', 'w') as f:
            f.write('''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <path d="M 0,0 L 100,0 L 100,100 L 0,100 Z" />
  <path d="M 25,25 L 75,25 L 75,75 L 25,75 Z" />
</svg>''')
        
        # Create a test onion with the test profile
        self.test_onion = RealisticOnion(
            max_diameter=2.0,
            profile_name='test_onion',
            center_axis_offset=0.0
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove the test SVG file
        try:
            os.remove('assets/test_onion.svg')
        except:
            pass

    def test_create_3d_visualization(self):
        """Test that the 3D visualization is created correctly."""
        fig = VisualizationService.create_3d_visualization(self.test_onion)
        
        # Check that we have a valid figure
        self.assertIsInstance(fig, go.Figure)
        
        # Check subplot titles
        self.assertEqual(fig.layout.annotations[0].text, "3D View")
        self.assertEqual(fig.layout.annotations[1].text, "XY Plane (Top View)")
        self.assertEqual(fig.layout.annotations[2].text, "XZ Plane (Front View)")
        
        # Check that coordinate systems are added
        # This is a basic check - we should see more traces than just our layers
        self.assertTrue(len(fig.data) > 6)

    def test_add_coordinate_system(self):
        """Test that coordinate system indicators are added correctly."""
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        VisualizationService.add_coordinate_system(fig, size=1.0, row=1, col=1)
        
        # Should have 6 traces: 3 axes lines and 3 arrow tips
        self.assertEqual(len(fig.data), 6)
        
        # Check that we have the correct trace types
        for trace in fig.data[:3]:  # First 3 traces should be lines with text
            self.assertEqual(trace.mode, "lines+text")
            self.assertIn(trace.text[1], ["X", "Y", "Z"])
        for trace in fig.data[3:]:  # Last 3 traces should be markers
            self.assertEqual(trace.mode, "markers")

    def test_add_2d_coordinate_system(self):
        """Test that 2D coordinate system indicators are added correctly."""
        fig = make_subplots(rows=1, cols=1)
        VisualizationService.add_2d_coordinate_system(
            fig, size=1.0, row=1, col=1, x_label="X", y_label="Y"
        )
        
        # Should have 4 traces: 2 axes lines and 2 arrow tips
        self.assertEqual(len(fig.data), 4)
        
        # Check that we have the correct trace types
        for trace in fig.data[:2]:  # First 2 traces should be lines with text
            self.assertEqual(trace.mode, "lines+text")
            self.assertIn(trace.text[1], ["X", "Y"])
        for trace in fig.data[2:]:  # Last 2 traces should be markers
            self.assertEqual(trace.mode, "markers")

if __name__ == "__main__":
    unittest.main() 