import unittest
import numpy as np
import trimesh
from shapely.geometry import Polygon

from models.cut import OnionPiece3D

class TestOnionPiece3D(unittest.TestCase):
    """Test cases for the enhanced OnionPiece3D class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple cube mesh for testing
        self.cube_vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        self.cube_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom face
            [4, 5, 6], [4, 6, 7],  # top face
            [0, 1, 5], [0, 5, 4],  # front face
            [2, 3, 7], [2, 7, 6],  # back face
            [0, 3, 7], [0, 7, 4],  # left face
            [1, 2, 6], [1, 6, 5]   # right face
        ])
        self.cube_mesh = trimesh.Trimesh(
            vertices=self.cube_vertices,
            faces=self.cube_faces
        )
        
        # Create a simple 2D polygon for compatibility with existing code
        self.polygon_2d = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_init_with_mesh(self):
        """Test initializing with a mesh object."""
        # Create a piece with mesh
        piece = OnionPiece3D(
            geometry={
                'id': 1,
                'layer_index': 2,
                'piece_2d': self.polygon_2d
            },
            mesh=self.cube_mesh
        )
        
        # Verify the mesh was stored
        self.assertIsNotNone(piece.mesh)
        self.assertEqual(piece.mesh.vertices.shape, (8, 3))
        self.assertEqual(piece.mesh.faces.shape, (12, 3))
        
        # Verify metadata
        self.assertEqual(piece.id, 1)
        self.assertEqual(piece.layer_index, 2)
        self.assertIs(piece.piece_2d, self.polygon_2d)
    
    def test_face_classification(self):
        """Test face classification functionality."""
        # Create a piece with mesh and pre-classified faces
        face_classes = {
            'external': [0, 1],
            'layer': [2, 3],
            'cut': [4, 5, 6, 7, 8, 9, 10, 11]
        }
        
        piece = OnionPiece3D(
            geometry={'id': 1, 'layer_index': 2},
            mesh=self.cube_mesh,
            face_classes=face_classes
        )
        
        # Verify face classifications were stored
        self.assertEqual(len(piece.face_classes['external']), 2)
        self.assertEqual(len(piece.face_classes['layer']), 2)
        self.assertEqual(len(piece.face_classes['cut']), 8)
        
        # Verify classification accessors
        external_faces = piece.get_faces_by_class('external')
        self.assertEqual(len(external_faces), 2)
        self.assertTrue(all(idx in face_classes['external'] for idx in external_faces))
    
    def test_volume_calculation(self):
        """Test volume calculation using trimesh."""
        piece = OnionPiece3D(
            geometry={'id': 1, 'layer_index': 2},
            mesh=self.cube_mesh
        )
        
        # The volume of the unit cube should be 1.0
        self.assertAlmostEqual(piece.volume, 1.0, places=5)
    
    def test_surface_area_calculation(self):
        """Test surface area calculations."""
        # Create a piece with mesh and face classifications
        face_classes = {
            'external': [0, 1],          # Bottom face (2 triangles)
            'layer': [2, 3],             # Top face (2 triangles)
            'cut': [4, 5, 6, 7, 8, 9, 10, 11]  # Side faces (8 triangles)
        }
        
        piece = OnionPiece3D(
            geometry={'id': 1, 'layer_index': 2},
            mesh=self.cube_mesh,
            face_classes=face_classes
        )
        
        # Calculate expected areas (each face of the cube has area 1.0)
        # Bottom face (2 triangles) = 1.0
        # Top face (2 triangles) = 1.0
        # Side faces (8 triangles) = 4.0
        expected_areas = {
            'external': 1.0,
            'layer': 1.0,
            'cut': 4.0,
            'total': 6.0
        }
        
        # Get actual areas
        actual_areas = piece.surface_areas
        
        # Verify each area type
        for area_type, expected_area in expected_areas.items():
            self.assertAlmostEqual(
                actual_areas[area_type], 
                expected_area, 
                places=5,
                msg=f"Area type '{area_type}' doesn't match expected value"
            )
    
    def test_serialization(self):
        """Test serialization to dict."""
        # Create a piece with mesh and face classifications
        face_classes = {
            'external': [0, 1],
            'layer': [2, 3],
            'cut': [4, 5, 6, 7, 8, 9, 10, 11]
        }
        
        piece = OnionPiece3D(
            geometry={'id': 1, 'layer_index': 2},
            mesh=self.cube_mesh,
            face_classes=face_classes
        )
        
        # Serialize to dict
        piece_dict = piece.to_dict()
        
        # Verify serialization
        self.assertEqual(piece_dict['id'], 1)
        self.assertEqual(piece_dict['layer_index'], 2)
        self.assertAlmostEqual(piece_dict['volume'], 1.0, places=5)
        
        # Verify surface areas
        self.assertAlmostEqual(piece_dict['surface_areas']['external'], 1.0, places=5)
        self.assertAlmostEqual(piece_dict['surface_areas']['layer'], 1.0, places=5)
        self.assertAlmostEqual(piece_dict['surface_areas']['cut'], 4.0, places=5)
        self.assertAlmostEqual(piece_dict['surface_areas']['total'], 6.0, places=5)
    
    def test_deserialization(self):
        """Test deserialization from dict with mesh."""
        # First serialize a piece
        face_classes = {
            'external': [0, 1],
            'layer': [2, 3],
            'cut': [4, 5, 6, 7, 8, 9, 10, 11]
        }
        
        original_piece = OnionPiece3D(
            geometry={'id': 1, 'layer_index': 2},
            mesh=self.cube_mesh,
            face_classes=face_classes
        )
        
        # Serialize to dict with mesh data
        piece_dict = original_piece.to_dict(include_mesh=True)
        
        # Deserialize to a new piece
        new_piece = OnionPiece3D.from_dict(piece_dict)
        
        # Verify deserialization
        self.assertEqual(new_piece.id, 1)
        self.assertEqual(new_piece.layer_index, 2)
        self.assertAlmostEqual(new_piece.volume, 1.0, places=5)
        
        # Verify mesh was reconstructed
        self.assertIsNotNone(new_piece.mesh)
        self.assertEqual(new_piece.mesh.vertices.shape, (8, 3))
        self.assertEqual(new_piece.mesh.faces.shape, (12, 3))
        
        # Verify face classifications
        external_faces = new_piece.get_faces_by_class('external')
        self.assertEqual(len(external_faces), 2)
        self.assertTrue(all(idx in face_classes['external'] for idx in external_faces))

if __name__ == '__main__':
    unittest.main() 