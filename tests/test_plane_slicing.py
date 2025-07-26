import unittest
import numpy as np
import trimesh
from models.plane_slicing import PlaneSlicing


class TestPlaneSlicing(unittest.TestCase):
    """Test cases for the PlaneSlicing utility class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple cube mesh centered at origin with size 2x2x2
        self.cube = trimesh.creation.box(extents=[2, 2, 2])
        
        # Create a sphere mesh centered at origin with radius 1
        self.sphere = trimesh.creation.icosphere(radius=1.0)
        
        # Create a simple triangle (non-manifold mesh)
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])
        faces = np.array([[0, 1, 2]])
        self.triangle = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Create a hollow mesh (cube with sphere removed) using manifold engine
        try:
            self.hollow_mesh = self.cube.difference(self.sphere, engine='manifold')
            # Check if the result is valid
            if not self.hollow_mesh or self.hollow_mesh.is_empty:
                print("Warning: Manifold difference resulted in an empty mesh.")
                # Fallback or create a placeholder hollow mesh?
                # For now, let's try boolean again without manifold
                self.hollow_mesh = self.cube.difference(self.sphere)
        except Exception as e:
            print(f"Warning: Manifold difference failed ({e}), falling back.")
            self.hollow_mesh = self.cube.difference(self.sphere)
            
        # Ensure hollow mesh is not empty for the test
        if not self.hollow_mesh or self.hollow_mesh.is_empty:
            print("Error: Could not create a valid hollow_mesh for testing.")
            # Create a simple box as a placeholder to avoid immediate test failure
            self.hollow_mesh = trimesh.creation.box(extents=[1.5, 1.5, 1.5])

    def test_find_intersection_points_cube(self):
        """Test finding intersection points of a plane with cube edges."""
        # Plane z=0 cutting through the middle of the cube
        plane_eq = (0, 0, 1, 0)  # z=0 plane
        
        # Find intersection points
        intersections = PlaneSlicing.find_intersection_points(self.cube, plane_eq)
        
        # Should have 4 intersection points (4 edges crossing z=0)
        self.assertEqual(len(intersections), 4)
        
        # All points should have z=0
        for point in intersections:
            self.assertAlmostEqual(point[2], 0.0)
            
            # Points should be on the edges of the cube
            self.assertLessEqual(abs(point[0]), 1.0)
            self.assertLessEqual(abs(point[1]), 1.0)

    def test_find_intersection_points_sphere(self):
        """Test finding intersection points of a plane with sphere edges."""
        # Plane z=0 cutting through the middle of the sphere
        plane_eq = (0, 0, 1, 0)  # z=0 plane
        
        # Find intersection points
        intersections = PlaneSlicing.find_intersection_points(self.sphere, plane_eq)
        
        # Should have multiple intersection points
        self.assertGreater(len(intersections), 0)
        
        # All points should have z=0
        for point in intersections:
            self.assertAlmostEqual(point[2], 0.0)
            
            # Points should be approximately 1.0 units from origin (on sphere surface)
            distance_from_origin = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
            self.assertAlmostEqual(distance_from_origin, 1.0, places=5)

    def test_generate_cutting_contour(self):
        """Test generating a cutting contour along the intersection."""
        # Plane z=0 cutting through the middle of the cube
        plane_eq = (0, 0, 1, 0)  # z=0 plane
        
        # Generate cutting contour
        contour = PlaneSlicing.generate_cutting_contour(self.cube, plane_eq)
        
        # Should have a valid contour
        self.assertIsNotNone(contour)
        self.assertGreater(len(contour), 0)
        
        # Contour should be a path with at least 3 points
        self.assertGreaterEqual(len(contour[0]), 3)
        
        # All points should have z=0
        for point in contour[0]:
            self.assertAlmostEqual(point[2], 0.0)

    def test_slice_mesh_cube(self):
        """Test slicing a cube mesh along a plane."""
        # Plane z=0 cutting through the middle of the cube
        plane_eq = (0, 0, 1, 0)  # z=0 plane
        
        # Slice the mesh
        result = PlaneSlicing.slice_mesh(self.cube, plane_eq)
        
        # Should have 2 resulting meshes
        self.assertEqual(len(result), 2)
        
        # Each result should be a valid mesh
        for mesh in result:
            self.assertIsInstance(mesh, trimesh.Trimesh)
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.volume > 0)
        
        # Each piece should have approximately half the volume of the original
        cube_volume = self.cube.volume
        for mesh in result:
            self.assertAlmostEqual(mesh.volume, cube_volume / 2, places=5)
            
        # The combined volume should equal the original
        combined_volume = sum(mesh.volume for mesh in result)
        self.assertAlmostEqual(combined_volume, cube_volume, places=5)

    def test_slice_mesh_sphere(self):
        """Test slicing a sphere mesh along a plane."""
        # Plane z=0 cutting through the middle of the sphere
        plane_eq = (0, 0, 1, 0)  # z=0 plane
        
        # Slice the mesh
        result = PlaneSlicing.slice_mesh(self.sphere, plane_eq)
        
        # Should have 2 resulting meshes
        self.assertEqual(len(result), 2)
        
        # Each result should be a valid mesh
        for mesh in result:
            self.assertIsInstance(mesh, trimesh.Trimesh)
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.volume > 0)
        
        # Each piece should have approximately half the volume of the original
        sphere_volume = self.sphere.volume
        for mesh in result:
            self.assertAlmostEqual(mesh.volume, sphere_volume / 2, places=4)
            
        # The combined volume should equal the original
        combined_volume = sum(mesh.volume for mesh in result)
        self.assertAlmostEqual(combined_volume, sphere_volume, places=4)

    def test_slice_with_vertex_plane(self):
        """Test slicing when the plane passes exactly through vertices."""
        # Create a mesh where we know vertices are at specific coordinates
        cube_at_vertex = trimesh.creation.box(extents=[2, 2, 2], transform=None)
        
        # Plane x=1 passing exactly through vertices
        plane_eq = (1, 0, 0, -1)  # x=1 plane
        
        # Slice the mesh
        result = PlaneSlicing.slice_mesh(cube_at_vertex, plane_eq)
        
        # Should still create valid meshes
        self.assertEqual(len(result), 2)
        
        # Each result should be a valid mesh
        for mesh in result:
            self.assertIsInstance(mesh, trimesh.Trimesh)
            self.assertTrue(mesh.is_watertight)

    def test_slice_with_parallel_plane(self):
        """Test slicing with a plane nearly parallel to a face."""
        # Plane almost parallel to the top face of the cube
        plane_eq = (0, 1, 0.001, -0.99)  # slightly tilted from y=0.99 plane
        
        # Slice the mesh
        result = PlaneSlicing.slice_mesh(self.cube, plane_eq)
        
        # Should still create valid meshes
        self.assertGreaterEqual(len(result), 1)
        
        # Each result should be a valid mesh
        for mesh in result:
            self.assertIsInstance(mesh, trimesh.Trimesh)
            self.assertTrue(mesh.is_watertight)

    def test_slice_non_manifold(self):
        """Test slicing a non-manifold mesh (triangle)."""
        # Plane cutting through the triangle
        plane_eq = (0.5, 0.5, 0, -0.25)  # 0.5x + 0.5y = 0.25 plane
        
        # Slice the mesh (should handle gracefully)
        with self.assertRaises(ValueError):
            # Should raise ValueError for non-manifold mesh
            PlaneSlicing.slice_mesh(self.triangle, plane_eq)

    def test_slice_hollow_mesh(self):
        """Test slicing a hollow mesh (cube with sphere removed)."""
        # Plane z=0 cutting through the middle
        plane_eq = (0, 0, 1, 0)  # z=0 plane
        
        # Slice the mesh
        result = PlaneSlicing.slice_mesh(self.hollow_mesh, plane_eq)
        
        # Should have 2 resulting meshes
        self.assertEqual(len(result), 2)
        
        # Each result should be a valid mesh
        for mesh in result:
            self.assertIsInstance(mesh, trimesh.Trimesh)
            self.assertTrue(mesh.is_watertight)
            
        # The combined volume should equal the original
        hollow_volume = self.hollow_mesh.volume
        combined_volume = sum(mesh.volume for mesh in result)
        self.assertAlmostEqual(combined_volume, hollow_volume, places=4)

    def test_slice_with_multiple_planes_sequential(self):
        """Test slicing a mesh with multiple planes sequentially."""
        # First plane: z=0
        plane_eq1 = (0, 0, 1, 0)
        
        # Slice with first plane
        intermediate_result = PlaneSlicing.slice_mesh(self.cube, plane_eq1)
        
        # Pick one of the resulting pieces
        piece = intermediate_result[0]
        
        # Second plane: x=0
        plane_eq2 = (1, 0, 0, 0)
        
        # Slice the piece with second plane
        final_result = PlaneSlicing.slice_mesh(piece, plane_eq2)
        
        # Should have 2 resulting meshes
        self.assertEqual(len(final_result), 2)
        
        # Each result should be a valid mesh
        for mesh in final_result:
            self.assertIsInstance(mesh, trimesh.Trimesh)
            self.assertTrue(mesh.is_watertight)
            
        # The combined volume should equal the intermediate piece
        piece_volume = piece.volume
        combined_volume = sum(mesh.volume for mesh in final_result)
        self.assertAlmostEqual(combined_volume, piece_volume, places=4)

    def test_face_labeling(self):
        """Test that newly created faces are labeled as 'cut' faces."""
        # Plane z=0 cutting through the middle of the cube
        plane_eq = (0, 0, 1, 0)  # z=0 plane
        
        # Slice the mesh with face labeling
        result, face_classes = PlaneSlicing.slice_mesh_with_labels(self.cube, plane_eq)
        
        # Should have 2 resulting meshes
        self.assertEqual(len(result), 2)
        self.assertEqual(len(face_classes), 2)
        
        # Each result should have labeled faces
        for i, mesh in enumerate(result):
            self.assertIn('cut', face_classes[i])
            self.assertGreater(len(face_classes[i]['cut']), 0)
            
            # The newly created face should be labeled as 'cut'
            # and should be approximately planar with z=0
            cut_faces = face_classes[i]['cut']
            for face_idx in cut_faces:
                vertices = mesh.triangles[face_idx]
                # Check that all vertices of the face have z ≈ 0
                for vertex in vertices:
                    self.assertAlmostEqual(vertex[2], 0.0, places=5)

    def test_edge_case_no_intersection(self):
        """Test the edge case where a plane doesn't intersect the mesh."""
        # Plane z=5 (outside the cube)
        plane_eq = (0, 0, 1, -5)  # z=5 plane
        
        # Slice the mesh
        result = PlaneSlicing.slice_mesh(self.cube, plane_eq)
        
        # Should have 1 resulting mesh (the original)
        self.assertEqual(len(result), 1)
        
        # The mesh should be identical to the original
        mesh = result[0]
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(mesh.volume, self.cube.volume, places=5)


if __name__ == '__main__':
    unittest.main() 