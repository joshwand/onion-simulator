"""
Statistical analysis for the onion simulator.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from models.cut import OnionPiece3D


class AnalysisService:
    """Service for statistical analysis of onion cuts."""
    
    @staticmethod
    def calculate_volume_statistics(pieces: List[OnionPiece3D]) -> Dict[str, float]:
        """
        Calculate comprehensive statistics on piece volumes.
        
        Args:
            pieces: List of OnionPiece3D objects
            
        Returns:
            Dictionary of statistics (min, max, mean, median, std, percentiles)
        """
        if not pieces:
            return {
                'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0,
                'count': 0, 'total': 0, 'q25': 0, 'q75': 0
            }
        
        volumes = [piece.volume for piece in pieces]
        
        return {
            'min': float(np.min(volumes)),
            'max': float(np.max(volumes)),
            'mean': float(np.mean(volumes)),
            'median': float(np.median(volumes)),
            'std': float(np.std(volumes)),
            'count': len(volumes),
            'total': float(np.sum(volumes)),
            'q25': float(np.percentile(volumes, 25)),
            'q75': float(np.percentile(volumes, 75))
        }
    
    @staticmethod
    def calculate_surface_area_statistics(pieces: List[OnionPiece3D]) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive statistics on piece surface areas by type.
        
        Args:
            pieces: List of OnionPiece3D objects
            
        Returns:
            Dictionary of statistics by surface type (external, layer, cut, total)
        """
        if not pieces:
            empty_stats = {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'count': 0, 'total': 0}
            return {
                'external': empty_stats.copy(),
                'layer': empty_stats.copy(),
                'cut': empty_stats.copy(),
                'total': empty_stats.copy()
            }
        
        # Collect surface areas by type
        external_areas = []
        layer_areas = []
        cut_areas = []
        total_areas = []
        
        for piece in pieces:
            surface_areas = piece.surface_areas
            external_areas.append(surface_areas.get('external', 0))
            layer_areas.append(surface_areas.get('layer', 0))
            cut_areas.append(surface_areas.get('cut', 0))
            total_areas.append(surface_areas.get('total', 0))
        
        def calculate_stats(values):
            return {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'count': len(values),
                'total': float(np.sum(values))
            }
        
        return {
            'external': calculate_stats(external_areas),
            'layer': calculate_stats(layer_areas),
            'cut': calculate_stats(cut_areas),
            'total': calculate_stats(total_areas)
        }
    
    @staticmethod
    def calculate_surface_to_volume_ratio(pieces: List[OnionPiece3D]) -> Dict[str, float]:
        """
        Calculate surface area to volume ratio statistics.
        
        Args:
            pieces: List of OnionPiece3D objects
            
        Returns:
            Dictionary of surface-to-volume ratio statistics
        """
        if not pieces:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'count': 0}
        
        ratios = []
        for piece in pieces:
            if piece.volume > 1e-12:  # Avoid division by zero
                total_surface_area = piece.surface_areas.get('total', 0)
                ratio = total_surface_area / piece.volume
                ratios.append(ratio)
        
        if not ratios:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'count': 0}
        
        return {
            'min': float(np.min(ratios)),
            'max': float(np.max(ratios)),
            'mean': float(np.mean(ratios)),
            'median': float(np.median(ratios)),
            'std': float(np.std(ratios)),
            'count': len(ratios)
        }
    
    @staticmethod
    def analyze_piece_size_distribution(pieces: List[OnionPiece3D], n_bins: int = 10) -> Dict[str, Any]:
        """
        Analyze the distribution of piece sizes.
        
        Args:
            pieces: List of OnionPiece3D objects
            n_bins: Number of bins for histogram
            
        Returns:
            Dictionary containing histogram data and distribution metrics
        """
        if not pieces:
            return {'bins': [], 'counts': [], 'bin_edges': [], 'metrics': {}}
        
        volumes = [piece.volume for piece in pieces]
        
        # Create histogram
        counts, bin_edges = np.histogram(volumes, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate distribution metrics
        total_volume = sum(volumes)
        volume_percentages = [(v / total_volume * 100) if total_volume > 0 else 0 for v in volumes]
        
        return {
            'bins': bin_centers.tolist(),
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist(),
            'metrics': {
                'largest_piece_percentage': max(volume_percentages) if volume_percentages else 0,
                'smallest_piece_percentage': min(volume_percentages) if volume_percentages else 0,
                'average_piece_percentage': np.mean(volume_percentages) if volume_percentages else 0,
                'uniformity_coefficient': max(0, 1.0 - (np.std(volumes) / np.mean(volumes))) if volumes and np.mean(volumes) > 0 else 0
            }
        }
    
    @staticmethod
    def calculate_cutting_efficiency(pieces: List[OnionPiece3D], total_cut_length: float = 0) -> Dict[str, float]:
        """
        Calculate metrics related to cutting efficiency.
        
        Args:
            pieces: List of OnionPiece3D objects
            total_cut_length: Total length of cuts applied
            
        Returns:
            Dictionary of cutting efficiency metrics
        """
        if not pieces:
            return {
                'total_cut_surface_area': 0,
                'cut_area_per_unit_length': 0,
                'cut_surface_percentage': 0,
                'waste_factor': 0
            }
        
        total_cut_surface_area = sum(piece.surface_areas.get('cut', 0) for piece in pieces)
        total_surface_area = sum(piece.surface_areas.get('total', 0) for piece in pieces)
        
        cut_area_per_unit_length = (total_cut_surface_area / total_cut_length) if total_cut_length > 0 else 0
        cut_surface_percentage = (total_cut_surface_area / total_surface_area * 100) if total_surface_area > 0 else 0
        
        # Waste factor: ratio of created surface area to original volume
        total_volume = sum(piece.volume for piece in pieces)
        waste_factor = (total_cut_surface_area / total_volume) if total_volume > 0 else 0
        
        return {
            'total_cut_surface_area': float(total_cut_surface_area),
            'cut_area_per_unit_length': float(cut_area_per_unit_length),
            'cut_surface_percentage': float(cut_surface_percentage),
            'waste_factor': float(waste_factor)
        }
    
    @staticmethod
    def analyze_layer_distribution(pieces: List[OnionPiece3D]) -> Dict[str, Any]:
        """
        Analyze how pieces are distributed across layers.
        
        Args:
            pieces: List of OnionPiece3D objects
            
        Returns:
            Dictionary containing layer distribution analysis
        """
        if not pieces:
            return {'layer_counts': {}, 'layer_volumes': {}, 'layer_statistics': {}}
        
        # Group pieces by layer
        layer_pieces = {}
        for piece in pieces:
            layer_idx = piece.layer_index
            if layer_idx not in layer_pieces:
                layer_pieces[layer_idx] = []
            layer_pieces[layer_idx].append(piece)
        
        # Calculate statistics per layer
        layer_counts = {}
        layer_volumes = {}
        layer_statistics = {}
        
        for layer_idx, layer_piece_list in layer_pieces.items():
            layer_counts[layer_idx] = len(layer_piece_list)
            layer_volume = sum(piece.volume for piece in layer_piece_list)
            layer_volumes[layer_idx] = float(layer_volume)
            
            # Calculate layer-specific statistics
            volumes = [piece.volume for piece in layer_piece_list]
            layer_statistics[layer_idx] = {
                'piece_count': len(layer_piece_list),
                'total_volume': float(layer_volume),
                'average_piece_volume': float(np.mean(volumes)),
                'volume_std': float(np.std(volumes)),
                'min_piece_volume': float(np.min(volumes)),
                'max_piece_volume': float(np.max(volumes))
            }
        
        return {
            'layer_counts': layer_counts,
            'layer_volumes': layer_volumes,
            'layer_statistics': layer_statistics
        }
    
    @staticmethod
    def calculate_comprehensive_metrics(
        pieces: List[OnionPiece3D], 
        total_cut_length: float = 0,
        cutting_method: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a complete cutting analysis.
        
        Args:
            pieces: List of OnionPiece3D objects
            total_cut_length: Total length of cuts applied
            cutting_method: Name of the cutting method used
            
        Returns:
            Dictionary containing all analysis metrics
        """
        return {
            'cutting_method': cutting_method,
            'piece_count': len(pieces),
            'volume_statistics': AnalysisService.calculate_volume_statistics(pieces),
            'surface_area_statistics': AnalysisService.calculate_surface_area_statistics(pieces),
            'surface_to_volume_ratio': AnalysisService.calculate_surface_to_volume_ratio(pieces),
            'size_distribution': AnalysisService.analyze_piece_size_distribution(pieces),
            'cutting_efficiency': AnalysisService.calculate_cutting_efficiency(pieces, total_cut_length),
            'layer_distribution': AnalysisService.analyze_layer_distribution(pieces)
        }
    
    @staticmethod
    def compare_cutting_methods(
        results: Dict[str, List[OnionPiece3D]], 
        cut_lengths: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Compare results from different cutting methods.
        
        Args:
            results: Dictionary mapping method names to lists of pieces
            cut_lengths: Optional dictionary mapping method names to total cut lengths
            
        Returns:
            Dictionary containing comparative analysis
        """
        cut_lengths = cut_lengths or {}
        comparison = {}
        
        for method_name, pieces in results.items():
            total_cut_length = cut_lengths.get(method_name, 0)
            comparison[method_name] = AnalysisService.calculate_comprehensive_metrics(
                pieces, total_cut_length, method_name
            )
        
        # Add comparative metrics
        if len(results) > 1:
            # Compare piece counts
            piece_counts = {method: len(pieces) for method, pieces in results.items()}
            
            # Compare average volumes
            avg_volumes = {}
            for method, pieces in results.items():
                if pieces:
                    avg_volumes[method] = np.mean([piece.volume for piece in pieces])
                else:
                    avg_volumes[method] = 0
            
            # Compare efficiency metrics
            efficiency_scores = {}
            for method in results.keys():
                metrics = comparison[method]
                # Simple efficiency score based on uniformity and cut efficiency
                uniformity = metrics['size_distribution']['metrics'].get('uniformity_coefficient', 0)
                cut_efficiency = 1.0 / (1.0 + metrics['cutting_efficiency'].get('waste_factor', 1))
                efficiency_scores[method] = (uniformity + cut_efficiency) / 2
            
            comparison['comparative_analysis'] = {
                'piece_counts': piece_counts,
                'average_volumes': avg_volumes,
                'efficiency_scores': efficiency_scores,
                'best_method_by_uniformity': max(efficiency_scores.keys(), key=lambda k: efficiency_scores[k]) if efficiency_scores else None
            }
        
        return comparison 