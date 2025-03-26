"""
Statistical analysis for the onion simulator.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from models.cut import OnionPiece3D


class AnalysisService:
    """Service for statistical analysis of onion cuts."""
    
    @staticmethod
    def calculate_volume_statistics(pieces: List[OnionPiece3D]) -> Dict[str, float]:
        """
        Calculate statistics on piece volumes.
        
        Args:
            pieces: List of OnionPiece3D objects
            
        Returns:
            Dictionary of statistics (min, max, mean, median, std)
        """
        volumes = [piece.volume for piece in pieces]
        
        return {
            'min': np.min(volumes) if volumes else 0,
            'max': np.max(volumes) if volumes else 0,
            'mean': np.mean(volumes) if volumes else 0,
            'median': np.median(volumes) if volumes else 0,
            'std': np.std(volumes) if volumes else 0,
            'count': len(volumes)
        }
    
    @staticmethod
    def calculate_surface_area_statistics(pieces: List[OnionPiece3D]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics on piece surface areas.
        
        Args:
            pieces: List of OnionPiece3D objects
            
        Returns:
            Dictionary of statistics by area type
        """
        if not pieces:
            return {
                'total': {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'count': 0},
                'external': {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'count': 0},
                'layer': {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'count': 0},
                'cut': {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'count': 0}
            }
        
        total_areas = [piece.total_surface_area for piece in pieces]
        external_areas = [piece.surface_areas['external'] for piece in pieces]
        layer_areas = [piece.surface_areas['layer'] for piece in pieces]
        cut_areas = [piece.surface_areas['cut'] for piece in pieces]
        
        return {
            'total': {
                'min': np.min(total_areas),
                'max': np.max(total_areas),
                'mean': np.mean(total_areas),
                'median': np.median(total_areas),
                'std': np.std(total_areas),
                'count': len(total_areas)
            },
            'external': {
                'min': np.min(external_areas),
                'max': np.max(external_areas),
                'mean': np.mean(external_areas),
                'median': np.median(external_areas),
                'std': np.std(external_areas),
                'count': len(external_areas)
            },
            'layer': {
                'min': np.min(layer_areas),
                'max': np.max(layer_areas),
                'mean': np.mean(layer_areas),
                'median': np.median(layer_areas),
                'std': np.std(layer_areas),
                'count': len(layer_areas)
            },
            'cut': {
                'min': np.min(cut_areas),
                'max': np.max(cut_areas),
                'mean': np.mean(cut_areas),
                'median': np.median(cut_areas),
                'std': np.std(cut_areas),
                'count': len(cut_areas)
            }
        }
    
    @staticmethod
    def calculate_area_statistics(areas: List[float]) -> Dict[str, float]:
        """
        Calculate statistics on 2D piece areas.
        
        Args:
            areas: List of areas
            
        Returns:
            Dictionary of statistics
        """
        if not areas:
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'count': 0
            }
        
        return {
            'min': np.min(areas),
            'max': np.max(areas),
            'mean': np.mean(areas),
            'median': np.median(areas),
            'std': np.std(areas),
            'count': len(areas)
        }
    
    @staticmethod
    def calculate_aspect_ratio_statistics(shapes: List[tuple]) -> Dict[str, float]:
        """
        Calculate statistics on piece aspect ratios.
        
        Args:
            shapes: List of shape tuples (width, height, aspect_ratio)
            
        Returns:
            Dictionary of statistics for aspect ratios
        """
        if not shapes:
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'count': 0
            }
        
        aspect_ratios = [shape[2] for shape in shapes]
        
        return {
            'min': np.min(aspect_ratios),
            'max': np.max(aspect_ratios),
            'mean': np.mean(aspect_ratios),
            'median': np.median(aspect_ratios),
            'std': np.std(aspect_ratios),
            'count': len(aspect_ratios)
        }
    
    @staticmethod
    def calculate_piece_count_by_layer(pieces: List[OnionPiece3D], n_layers: int) -> Dict[int, int]:
        """
        Count pieces by layer (in a real implementation).
        
        Args:
            pieces: List of OnionPiece3D objects
            n_layers: Number of layers in the onion
            
        Returns:
            Dictionary mapping layer index to piece count
        """
        # This is a placeholder implementation
        # In a real implementation, this would analyze the 3D geometries to determine layers
        
        layer_counts = {i: 0 for i in range(n_layers)}
        
        # Randomly assign pieces to layers for demonstration
        for _ in pieces:
            layer = np.random.randint(0, n_layers)
            layer_counts[layer] += 1
        
        return layer_counts 