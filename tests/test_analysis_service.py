import unittest
import numpy as np
import trimesh
from typing import List, Dict

from models.cut import OnionPiece3D
from services.analysis_service import AnalysisService


class TestAnalysisService(unittest.TestCase):
    """Test cases for the enhanced AnalysisService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test pieces with known properties
        self.test_pieces = []
        
        # Create simple cube meshes with different sizes
        for i, size in enumerate([1.0, 2.0, 1.5, 0.5, 3.0]):
            mesh = trimesh.creation.box(extents=[size, size, size])
            
            geometry = {
                'id': i,
                'layer_index': i % 3,  # Distribute across 3 layers
            }
            
            face_classes = {
                'external': list(range(0, len(mesh.faces) // 3)),
                'layer': list(range(len(mesh.faces) // 3, 2 * len(mesh.faces) // 3)),
                'cut': list(range(2 * len(mesh.faces) // 3, len(mesh.faces)))
            }
            
            piece = OnionPiece3D(
                geometry=geometry,
                mesh=mesh,
                face_classes=face_classes,
                cut_planes=[],
                layer_planes=[]
            )
            
            self.test_pieces.append(piece)

    def test_volume_statistics(self):
        """Test volume statistics calculation."""
        stats = AnalysisService.calculate_volume_statistics(self.test_pieces)
        
        # Check that all expected keys are present
        expected_keys = ['min', 'max', 'mean', 'median', 'std', 'count', 'total', 'q25', 'q75']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check basic properties
        self.assertEqual(stats['count'], 5)
        self.assertGreater(stats['max'], stats['min'])
        self.assertGreater(stats['total'], 0)
        self.assertGreaterEqual(stats['q75'], stats['q25'])

    def test_surface_area_statistics(self):
        """Test surface area statistics calculation."""
        stats = AnalysisService.calculate_surface_area_statistics(self.test_pieces)
        
        # Check that all surface types are present
        expected_types = ['external', 'layer', 'cut', 'total']
        for surface_type in expected_types:
            self.assertIn(surface_type, stats)
            
            # Check that each type has proper statistics
            type_stats = stats[surface_type]
            expected_stats = ['min', 'max', 'mean', 'median', 'std', 'count', 'total']
            for stat_key in expected_stats:
                self.assertIn(stat_key, type_stats)
                self.assertGreaterEqual(type_stats[stat_key], 0)

    def test_surface_to_volume_ratio(self):
        """Test surface-to-volume ratio calculation."""
        ratios = AnalysisService.calculate_surface_to_volume_ratio(self.test_pieces)
        
        expected_keys = ['min', 'max', 'mean', 'median', 'std', 'count']
        for key in expected_keys:
            self.assertIn(key, ratios)
        
        # All ratios should be positive
        self.assertGreater(ratios['max'], 0)
        self.assertGreaterEqual(ratios['min'], 0)
        self.assertEqual(ratios['count'], 5)

    def test_piece_size_distribution(self):
        """Test piece size distribution analysis."""
        distribution = AnalysisService.analyze_piece_size_distribution(self.test_pieces, n_bins=5)
        
        # Check structure
        self.assertIn('bins', distribution)
        self.assertIn('counts', distribution)
        self.assertIn('bin_edges', distribution)
        self.assertIn('metrics', distribution)
        
        # Check bins and counts
        self.assertEqual(len(distribution['bins']), 5)
        self.assertEqual(len(distribution['counts']), 5)
        self.assertEqual(len(distribution['bin_edges']), 6)
        
        # Check metrics
        metrics = distribution['metrics']
        expected_metrics = ['largest_piece_percentage', 'smallest_piece_percentage', 
                          'average_piece_percentage', 'uniformity_coefficient']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertGreaterEqual(metrics[metric], 0)

    def test_cutting_efficiency(self):
        """Test cutting efficiency calculation."""
        total_cut_length = 10.0
        efficiency = AnalysisService.calculate_cutting_efficiency(self.test_pieces, total_cut_length)
        
        expected_keys = ['total_cut_surface_area', 'cut_area_per_unit_length', 
                        'cut_surface_percentage', 'waste_factor']
        for key in expected_keys:
            self.assertIn(key, efficiency)
            self.assertGreaterEqual(efficiency[key], 0)

    def test_layer_distribution(self):
        """Test layer distribution analysis."""
        distribution = AnalysisService.analyze_layer_distribution(self.test_pieces)
        
        # Check structure
        self.assertIn('layer_counts', distribution)
        self.assertIn('layer_volumes', distribution)
        self.assertIn('layer_statistics', distribution)
        
        # Check that we have data for the layers we created (0, 1, 2)
        layer_counts = distribution['layer_counts']
        self.assertGreater(len(layer_counts), 0)
        
        # Total pieces should match
        total_counted = sum(layer_counts.values())
        self.assertEqual(total_counted, 5)

    def test_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        metrics = AnalysisService.calculate_comprehensive_metrics(
            self.test_pieces, 
            total_cut_length=15.0,
            cutting_method="Test Method"
        )
        
        # Check all sections are present
        expected_sections = [
            'cutting_method', 'piece_count', 'volume_statistics', 
            'surface_area_statistics', 'surface_to_volume_ratio',
            'size_distribution', 'cutting_efficiency', 'layer_distribution'
        ]
        for section in expected_sections:
            self.assertIn(section, metrics)
        
        # Check basic properties
        self.assertEqual(metrics['cutting_method'], "Test Method")
        self.assertEqual(metrics['piece_count'], 5)

    def test_compare_cutting_methods(self):
        """Test cutting method comparison."""
        # Create two different sets of pieces
        method1_pieces = self.test_pieces[:3]
        method2_pieces = self.test_pieces[2:]
        
        results = {
            'Method 1': method1_pieces,
            'Method 2': method2_pieces
        }
        
        cut_lengths = {
            'Method 1': 10.0,
            'Method 2': 12.0
        }
        
        comparison = AnalysisService.compare_cutting_methods(results, cut_lengths)
        
        # Check that both methods are analyzed
        self.assertIn('Method 1', comparison)
        self.assertIn('Method 2', comparison)
        self.assertIn('comparative_analysis', comparison)
        
        # Check comparative analysis structure
        comp_analysis = comparison['comparative_analysis']
        expected_comp_keys = ['piece_counts', 'average_volumes', 'efficiency_scores', 'best_method_by_uniformity']
        for key in expected_comp_keys:
            self.assertIn(key, comp_analysis)

    def test_empty_pieces_handling(self):
        """Test handling of empty piece lists."""
        empty_pieces = []
        
        # All functions should handle empty input gracefully
        volume_stats = AnalysisService.calculate_volume_statistics(empty_pieces)
        self.assertEqual(volume_stats['count'], 0)
        
        surface_stats = AnalysisService.calculate_surface_area_statistics(empty_pieces)
        self.assertEqual(surface_stats['total']['count'], 0)
        
        ratio_stats = AnalysisService.calculate_surface_to_volume_ratio(empty_pieces)
        self.assertEqual(ratio_stats['count'], 0)
        
        distribution = AnalysisService.analyze_piece_size_distribution(empty_pieces)
        self.assertEqual(len(distribution['bins']), 0)
        
        efficiency = AnalysisService.calculate_cutting_efficiency(empty_pieces)
        self.assertEqual(efficiency['total_cut_surface_area'], 0)
        
        layer_dist = AnalysisService.analyze_layer_distribution(empty_pieces)
        self.assertEqual(len(layer_dist['layer_counts']), 0)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Single piece
        single_piece = [self.test_pieces[0]]
        
        volume_stats = AnalysisService.calculate_volume_statistics(single_piece)
        self.assertEqual(volume_stats['count'], 1)
        self.assertEqual(volume_stats['min'], volume_stats['max'])
        
        # Zero volume pieces (should be handled gracefully)
        zero_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[])
        zero_geometry = {'id': 0, 'layer_index': 0}
        zero_piece = OnionPiece3D(
            geometry=zero_geometry,
            mesh=zero_mesh,
            face_classes={'external': [], 'layer': [], 'cut': []},
            cut_planes=[],
            layer_planes=[]
        )
        
        zero_pieces = [zero_piece]
        ratio_stats = AnalysisService.calculate_surface_to_volume_ratio(zero_pieces)
        # Should handle zero volume gracefully
        self.assertGreaterEqual(ratio_stats['count'], 0)


if __name__ == '__main__':
    unittest.main()