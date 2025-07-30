import unittest
import numpy as np
import trimesh
from typing import List, Dict

from models.cut import Cut, CrossCut, OnionPiece3D
from models.sequential_cutter import SequentialCutter


class TestSequentialCutter(unittest.TestCase):
    """Test cases for the SequentialCutter class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple cube mesh for testing
        self.cube = trimesh.creation.box(extents=[2, 2, 2])
        
        # Create a sphere mesh
        self.sphere = trimesh.creation.icosphere(radius=1.0)
        
        # Create test cuts using tuple coordinates instead of Point objects
        self.vertical_cut = Cut((-0.5, -1), (-0.5, 1))
        self.horizontal_cut = Cut((-1, 0.5), (1, 0.5))
        self.diagonal_cut = Cut((-0.8, -0.8), (0.8, 0.8))
        
        # Create a cross-cut for testing
        self.cross_cut = CrossCut((0, 0, 1), (0, 0, 0.2))  # z = 0.2 plane
        
        # Create layer planes for testing
        self.layer_planes = [(0, 1, 0, 0.3), (0, 1, 0, -0.3)]  # y = ±0.3 planes
        
        # Initialize sequential cutter
        self.cutter = SequentialCutter(optimize_order=True)

    def test_single_cut_application(self):
        """Test applying a single cut to a mesh."""
        cuts = [self.vertical_cut]
        
        pieces = self.cutter.apply_cuts_sequential(
            self.cube,
            cuts,
            layer_index=0
        )
        
        # Should produce 2 pieces from a single cut
        self.assertGreaterEqual(len(pieces), 1)
        
        # All pieces should be valid OnionPiece3D objects
        for piece in pieces:
            self.assertIsInstance(piece, OnionPiece3D)
            self.assertTrue(piece.is_valid())
            self.assertGreater(piece.volume, 0)

    def test_multiple_cuts_application(self):
        """Test applying multiple cuts sequentially."""
        cuts = [self.vertical_cut, self.horizontal_cut]
        
        pieces = self.cutter.apply_cuts_sequential(
            self.cube,
            cuts,
            layer_index=0
        )
        
        # Multiple cuts should produce multiple pieces
        self.assertGreater(len(pieces), 1)
        
        # All pieces should be valid
        for piece in pieces:
            self.assertIsInstance(piece, OnionPiece3D)
            self.assertTrue(piece.is_valid())
            self.assertGreater(piece.volume, 0)

    def test_cuts_with_cross_cuts(self):
        """Test applying both regular cuts and cross-cuts."""
        cuts = [self.vertical_cut]
        cross_cuts = [self.cross_cut]
        
        pieces = self.cutter.apply_cuts_sequential(
            self.cube,
            cuts,
            cross_cuts=cross_cuts,
            layer_index=0
        )
        
        # Should produce pieces from both regular and cross cuts
        self.assertGreater(len(pieces), 0)
        
        # All pieces should be valid
        for piece in pieces:
            self.assertIsInstance(piece, OnionPiece3D)
            self.assertTrue(piece.is_valid())

    def test_cutting_order_optimization(self):
        """Test that cutting order optimization works."""
        cuts = [self.vertical_cut, self.horizontal_cut, self.diagonal_cut]
        
        # Test with optimization enabled
        cutter_optimized = SequentialCutter(optimize_order=True)
        pieces_optimized = cutter_optimized.apply_cuts_sequential(
            self.cube,
            cuts,
            layer_index=0
        )
        
        # Test with optimization disabled
        cutter_unoptimized = SequentialCutter(optimize_order=False)
        pieces_unoptimized = cutter_unoptimized.apply_cuts_sequential(
            self.cube,
            cuts,
            layer_index=0
        )
        
        # Both should produce valid results
        self.assertGreater(len(pieces_optimized), 0)
        self.assertGreater(len(pieces_unoptimized), 0)
        
        # Check that both produce valid pieces
        for pieces in [pieces_optimized, pieces_unoptimized]:
            for piece in pieces:
                self.assertIsInstance(piece, OnionPiece3D)
                self.assertTrue(piece.is_valid())

    def test_layer_planes_preservation(self):
        """Test that layer planes are preserved in resulting pieces."""
        cuts = [self.vertical_cut]
        
        pieces = self.cutter.apply_cuts_sequential(
            self.cube,
            cuts,
            layer_planes=self.layer_planes,
            layer_index=2
        )
        
        # Check that layer information is preserved
        for piece in pieces:
            self.assertEqual(piece.layer_index, 2)
            self.assertEqual(len(piece.layer_planes), 2)
            self.assertIn((0, 1, 0, 0.3), piece.layer_planes)
            self.assertIn((0, 1, 0, -0.3), piece.layer_planes)

    def test_cutting_statistics(self):
        """Test cutting statistics tracking."""
        cuts = [self.vertical_cut, self.horizontal_cut]
        
        self.cutter.reset_history()
        pieces = self.cutter.apply_cuts_sequential(
            self.cube,
            cuts,
            layer_index=0
        )
        
        stats = self.cutter.get_cutting_stats()
        
        # Should track the number of cuts applied
        self.assertEqual(stats['total_cuts'], 2)
        self.assertGreater(stats['final_pieces'], 0)
        self.assertIn('cut_history', stats)
        self.assertEqual(len(stats['cut_history']), 2)

    def test_apply_cuts_to_existing_pieces(self):
        """Test applying additional cuts to existing pieces."""
        # First, create some initial pieces
        initial_cuts = [self.vertical_cut]
        initial_pieces = self.cutter.apply_cuts_sequential(
            self.cube,
            initial_cuts,
            layer_index=0
        )
        
        # Then apply additional cuts to those pieces
        additional_cuts = [self.horizontal_cut]
        final_pieces = self.cutter.apply_cuts_to_pieces(
            initial_pieces,
            additional_cuts
        )
        
        # Should have at least as many pieces as before
        self.assertGreaterEqual(len(final_pieces), len(initial_pieces))
        
        # All pieces should be valid
        for piece in final_pieces:
            self.assertIsInstance(piece, OnionPiece3D)
            self.assertTrue(piece.is_valid())

    def test_volume_filtering(self):
        """Test filtering out pieces that are too small."""
        cuts = [self.vertical_cut, self.horizontal_cut, self.diagonal_cut]
        
        # Apply cuts with a higher minimum volume threshold
        pieces = self.cutter.apply_cuts_sequential(
            self.cube,
            cuts,
            min_volume=0.1,  # Relatively high threshold
            layer_index=0
        )
        
        # All remaining pieces should meet the volume requirement
        for piece in pieces:
            self.assertGreaterEqual(piece.volume, 0.1)

    def test_edge_case_no_cuts(self):
        """Test handling when no cuts are provided."""
        pieces = self.cutter.apply_cuts_sequential(
            self.cube,
            [],  # No cuts
            layer_index=0
        )
        
        # Should return single piece for uncut mesh
        self.assertEqual(len(pieces), 1)
        self.assertIsInstance(pieces[0], OnionPiece3D)
        self.assertTrue(pieces[0].is_valid())

    def test_edge_case_invalid_cuts(self):
        """Test handling of invalid or degenerate cuts."""
        # Create a degenerate cut (zero length)
        degenerate_cut = Cut((0, 0), (0, 0))
        cuts = [degenerate_cut, self.vertical_cut]
        
        # Should handle gracefully and still produce results from valid cuts
        pieces = self.cutter.apply_cuts_sequential(
            self.cube,
            cuts,
            layer_index=0
        )
        
        # Should still get valid pieces from the valid cut
        self.assertGreater(len(pieces), 0)
        for piece in pieces:
            self.assertIsInstance(piece, OnionPiece3D)


if __name__ == '__main__':
    unittest.main()