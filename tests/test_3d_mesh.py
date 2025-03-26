"""
Test the 3D mesh generation functionality.
"""

import os
import unittest
import numpy as np
import trimesh
from models.onion_3d import RealisticOnion, SVGProfile


class Test3DMeshGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test SVG file if it doesn't exist
        self.test_svg_path = os.path.join('assets', 'test_onion.svg')
        if not os.path.exists(self.test_svg_path):
            self._create_test_svg()
        
        # Initialize the onion model
        self.onion = RealisticOnion(
            max_diameter=5.0,
            profile_name='test_onion'
        )
    
    def _create_test_svg(self):
        """Create a simple test SVG file with two layers."""
        os.makedirs('assets', exist_ok=True)
        
        # Simple SVG with two concentric closed paths
        svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
    <!-- Inner layer - closed path -->
    <path d="M 30,50 Q 50,20 70,50 Q 50,80 30,50 Z" />
    <!-- Outer layer - closed path -->
    <path d="M 20,50 Q 50,10 80,50 Q 50,90 20,50 Z" />
</svg>
"""
        with open(self.test_svg_path, 'w') as f:
            f.write(svg_content)
    
    def test_profile_revolving(self):
        """Test that a single profile can be revolved into a 3D mesh."""
        # Get the first profile
        profile = self.onion.svg_profiles[0]
        
        # Revolve the profile
        mesh = self.onion.revolve_profile(profile)
        
        # Verify the mesh properties
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertTrue(len(mesh.vertices) > 0)
        self.assertTrue(len(mesh.faces) > 0)
        self.assertTrue(mesh.is_watertight)  # Mesh should be closed
        self.assertTrue(mesh.is_winding_consistent)  # Face normals should be consistent
    
    def test_complete_3d_model(self):
        """Test generation of the complete 3D model with all layers."""
        # Generate the complete model
        mesh = self.onion.generate_3d_model()
        
        # Verify the mesh properties
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertTrue(len(mesh.vertices) > 0)
        self.assertTrue(len(mesh.faces) > 0)
        
        # Check metadata
        self.assertEqual(mesh.metadata['n_layers'], len(self.onion.svg_profiles))
        self.assertEqual(len(mesh.metadata['layer_vertices']), len(self.onion.svg_profiles))
        self.assertEqual(len(mesh.metadata['layer_faces']), len(self.onion.svg_profiles))
    
    def test_layer_scaling(self):
        """Test that layers are properly scaled according to max_diameter."""
        # Generate the model
        mesh = self.onion.generate_3d_model()
        
        # Calculate the actual maximum diameter
        vertices = mesh.vertices
        max_diameter = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        
        # Should be close to the specified max_diameter
        self.assertAlmostEqual(max_diameter, self.onion.max_diameter, places=2)
    
    def test_visual_properties(self):
        """Test that visual properties are correctly set for each layer."""
        mesh = self.onion.generate_3d_model()
        
        # Check that face colors exist
        self.assertIsNotNone(mesh.visual.face_colors)
        
        # The outer layer should be green (G channel > R,B channels)
        outer_face_start = sum(mesh.metadata['layer_faces'][:-1])
        outer_face_end = outer_face_start + mesh.metadata['layer_faces'][-1]
        outer_faces_colors = mesh.visual.face_colors[outer_face_start:outer_face_end]
        
        # Check that outer faces are green
        self.assertTrue(np.all(outer_faces_colors[:, 1] > outer_faces_colors[:, 0]))  # G > R
        self.assertTrue(np.all(outer_faces_colors[:, 1] > outer_faces_colors[:, 2]))  # G > B
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove test SVG file
        if os.path.exists(self.test_svg_path):
            os.remove(self.test_svg_path)


if __name__ == '__main__':
    unittest.main() 