import unittest
import numpy as np
import trimesh
from typing import List, Dict

from models.cut import OnionPiece3D
from models.plane_slicing import PlaneSlicing
from models.piece_extractor import PieceExtractor


class TestPieceExtractor(unittest.TestCase):
    """Test cases for the PieceExtractor utility class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple cube mesh for testing
        self.cube = trimesh.creation.box(extents=[2, 2, 2])
        
        # Create a sphere mesh
        self.sphere = trimesh.creation.icosphere(radius=1.0)
        
        # Create a more complex mesh (torus) for advanced testing
        self.torus = trimesh.creation.torus(major_radius=2, minor_radius=0.5)
        
        # Define some standard cutting planes
        self.horizontal_plane = (0, 0, 1, 0)  # z=0 plane
        self.vertical_plane = (1, 0, 0, 0)    # x=0 plane
        self.diagonal_plane = (1, 1, 0, 0)    # x+y=0 plane

    def test_extract_single_cut_two_pieces(self):
        """Test extracting two pieces from a single cut."""
        # Cut cube with horizontal plane
        sliced_meshes, face_labels = PlaneSlicing.slice_mesh_with_labels(
            self.cube, self.horizontal_plane
        )
        
        # Extract pieces
        pieces = PieceExtractor.extract_pieces(
            sliced_meshes, 
            face_labels,
            cut_planes=[self.horizontal_plane]
        )
        
        # Should have exactly 2 pieces
        self.assertEqual(len(pieces), 2)
        
        # Both pieces should be valid OnionPiece3D objects
        for piece in pieces:
            self.assertIsInstance(piece, OnionPiece3D)
            self.assertTrue(piece.is_valid())
            self.assertGreater(piece.volume, 0)
            self.assertGreater(piece.surface_areas['total'], 0)

    def test_extract_multiple_cuts(self):
        """Test extracting pieces from multiple intersecting cuts."""
        # First cut the cube with horizontal plane
        sliced_meshes_1, face_labels_1 = PlaneSlicing.slice_mesh_with_labels(
            self.cube, self.horizontal_plane
        )
        
        # Then cut one of the pieces with vertical plane
        if len(sliced_meshes_1) >= 2:
            sliced_meshes_2, face_labels_2 = PlaneSlicing.slice_mesh_with_labels(
                sliced_meshes_1[0], self.vertical_plane
            )
            
            # Extract pieces from the double-cut
            pieces = PieceExtractor.extract_pieces(
                sliced_meshes_2,
                face_labels_2,
                cut_planes=[self.horizontal_plane, self.vertical_plane]
            )
            
            # Should have at least 1 piece from the second cut (might be 1 if cut doesn't split)
            self.assertGreaterEqual(len(pieces), 1)
            
            # All pieces should be valid
            for piece in pieces:
                self.assertIsInstance(piece, OnionPiece3D)
                self.assertTrue(piece.is_valid())

    def test_connected_component_identification(self):
        """Test identifying connected components in complex meshes."""
        # Create two separate cubes (disconnected components)
        cube1 = trimesh.creation.box(extents=[1, 1, 1])
        cube1.vertices += [2, 0, 0]  # Translate first cube
        
        cube2 = trimesh.creation.box(extents=[1, 1, 1])
        cube2.vertices += [-2, 0, 0]  # Translate second cube
        
        # Combine into single mesh with two components
        combined_mesh = trimesh.util.concatenate([cube1, cube2])
        
        # Extract connected components
        components = PieceExtractor.identify_connected_components(combined_mesh)
        
        # Should identify exactly 2 components
        self.assertEqual(len(components), 2)
        
        # Each component should be a valid mesh
        for component in components:
            self.assertIsInstance(component, trimesh.Trimesh)
            self.assertTrue(component.is_watertight)  # Fixed: use is_watertight instead of is_valid
            self.assertGreater(len(component.vertices), 0)
            self.assertGreater(len(component.faces), 0)

    def test_face_classification_preservation(self):
        """Test that face classifications are preserved during extraction."""
        # Cut cube and get labeled faces
        sliced_meshes, face_labels = PlaneSlicing.slice_mesh_with_labels(
            self.cube, self.horizontal_plane
        )
        
        # Extract pieces with face classification
        pieces = PieceExtractor.extract_pieces(
            sliced_meshes,
            face_labels,
            cut_planes=[self.horizontal_plane]
        )
        
        # Check that each piece has proper face classifications
        for piece in pieces:
            face_classes = piece.face_classes
            
            # Should have all three types of faces
            self.assertIn('external', face_classes)
            self.assertIn('layer', face_classes)
            self.assertIn('cut', face_classes)
            
            # Cut faces should exist (from the slicing)
            self.assertGreater(len(face_classes['cut']), 0)

    def test_filter_degenerate_pieces(self):
        """Test filtering out degenerate or extremely small pieces."""
        # Create a mesh that when cut might produce tiny fragments
        # Use a thin sheet that gets cut at an angle
        vertices = np.array([
            [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],  # bottom vertices
            [0, 0, 0.01], [2, 0, 0.01], [2, 2, 0.01], [0, 2, 0.01]  # top vertices (very thin)
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5]   # right
        ])
        thin_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Cut with a nearly parallel plane that might create tiny slivers
        angled_plane = (0, 0, 1, -0.005)  # z=0.005 plane
        
        sliced_meshes, face_labels = PlaneSlicing.slice_mesh_with_labels(
            thin_mesh, angled_plane
        )
        
        # Extract pieces with filtering
        pieces = PieceExtractor.extract_pieces(
            sliced_meshes,
            face_labels,
            cut_planes=[angled_plane],
            min_volume=0.001  # Filter out very small pieces
        )
        
        # All remaining pieces should meet the minimum volume requirement
        for piece in pieces:
            self.assertGreaterEqual(piece.volume, 0.001)

    def test_metadata_preservation(self):
        """Test that metadata is properly preserved during extraction."""
        layer_planes = [(0, 1, 0, 0.5), (0, 1, 0, -0.5)]  # y=0.5 and y=-0.5 planes
        
        # Cut sphere to create pieces
        sliced_meshes, face_labels = PlaneSlicing.slice_mesh_with_labels(
            self.sphere, self.horizontal_plane
        )
        
        # Extract pieces with metadata
        pieces = PieceExtractor.extract_pieces(
            sliced_meshes,
            face_labels,
            cut_planes=[self.horizontal_plane],
            layer_planes=layer_planes,
            layer_index=2
        )
        
        # Check that metadata is preserved
        for piece in pieces:
            # Should have cut planes stored
            self.assertEqual(len(piece.cut_planes), 1)
            self.assertEqual(piece.cut_planes[0], self.horizontal_plane)
            
            # Should have layer planes stored
            self.assertEqual(len(piece.layer_planes), 2)
            self.assertIn((0, 1, 0, 0.5), piece.layer_planes)
            self.assertIn((0, 1, 0, -0.5), piece.layer_planes)
            
            # Should have layer index
            self.assertEqual(piece.layer_index, 2)

    def test_complex_geometry_extraction(self):
        """Test extraction with complex geometries like torus."""
        # Cut torus with vertical plane
        sliced_meshes, face_labels = PlaneSlicing.slice_mesh_with_labels(
            self.torus, self.vertical_plane
        )
        
        # Extract pieces
        pieces = PieceExtractor.extract_pieces(
            sliced_meshes,
            face_labels,
            cut_planes=[self.vertical_plane]
        )
        
        # Should have pieces (exact number depends on torus geometry)
        self.assertGreater(len(pieces), 0)
        
        # All pieces should be valid
        for piece in pieces:
            self.assertIsInstance(piece, OnionPiece3D)
            self.assertTrue(piece.is_valid())

    def test_edge_case_empty_mesh(self):
        """Test handling of edge case with empty mesh."""
        empty_mesh = trimesh.Trimesh()
        
        # Should handle empty mesh gracefully
        pieces = PieceExtractor.extract_pieces(
            [empty_mesh],
            [{'cut': [], 'layer': [], 'external': []}],
            cut_planes=[]
        )
        
        # Should return empty list for empty input
        self.assertEqual(len(pieces), 0)

    def test_edge_case_no_cuts(self):
        """Test handling when no cuts produce disconnected pieces."""
        # Pass an uncut mesh
        pieces = PieceExtractor.extract_pieces(
            [self.cube],
            [{'cut': [], 'layer': [], 'external': list(range(len(self.cube.faces)))}],
            cut_planes=[]
        )
        
        # Should return single piece for uncut mesh
        self.assertEqual(len(pieces), 1)
        self.assertIsInstance(pieces[0], OnionPiece3D)
        self.assertTrue(pieces[0].is_valid())


if __name__ == '__main__':
    unittest.main()