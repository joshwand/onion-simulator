"""
Geometric calculations for both 2D and 3D onion simulators.
"""

from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union, polygonize

from core.models import BaseOnion
from models.onion_2d import HalfOnion
from models.svg_profile_onion import SvgProfileOnion
from models.cut import Cut, CrossCut, OnionPiece3D
from models.sequential_cutter import SequentialCutter
import trimesh


class GeometryService:
    """Service for geometric calculations."""
    
    @staticmethod
    def apply_cuts_2d(onion: HalfOnion, cuts: List[Cut]) -> List[Polygon]:
        """
        Apply cuts to a 2D onion model to get pieces.
        
        Args:
            onion: The 2D onion model
            cuts: List of cuts to apply
            
        Returns:
            List of Shapely Polygon objects representing the pieces
        """
        # Filter out any cuts that have length < 0.001
        valid_cuts = [cut for cut in cuts if (cut.end[0] - cut.start[0])**2 + (cut.end[1] - cut.start[1])**2 > 0.001]

        lines = [cut.as_shapely() for cut in valid_cuts]
        lines.extend(onion.create_layer_boundaries())
        
        half_circle = Point(0, 0).buffer(onion.radius * 1.001).intersection(
            Polygon([(-onion.radius, 0), (onion.radius, 0), 
                     (onion.radius, onion.radius), (-onion.radius, onion.radius)])
        )
        
        intersections = unary_union(lines).intersection(half_circle)
        polygons = list(polygonize(intersections))
        
        return [polygon for polygon in polygons if polygon.is_valid and polygon.area > 0]
    
    @staticmethod
    def apply_cuts_3d(
        onion: SvgProfileOnion, 
        cuts: List[Cut], 
        cross_cuts: List[CrossCut] = None,
        use_advanced_cutting: bool = True
    ) -> List[OnionPiece3D]:
        """
        Apply both standard cuts and cross-cuts to a 3D onion model to get pieces.
        
        Args:
            onion: The 3D SVG profile onion model
            cuts: List of 2D cuts
            cross_cuts: Optional list of 3D cross-cuts
            use_advanced_cutting: Whether to use the new SequentialCutter for enhanced 3D cutting
            
        Returns:
            List of OnionPiece3D objects
        """
        if use_advanced_cutting:
            return GeometryService._apply_cuts_3d_advanced(onion, cuts, cross_cuts)
        else:
            return GeometryService._apply_cuts_3d_legacy(onion, cuts, cross_cuts)
    
    @staticmethod
    def _apply_cuts_3d_advanced(
        onion: SvgProfileOnion, 
        cuts: List[Cut], 
        cross_cuts: List[CrossCut] = None
    ) -> List[OnionPiece3D]:
        """
        Apply cuts using the advanced SequentialCutter approach.
        
        Args:
            onion: The 3D SVG profile onion model
            cuts: List of 2D cuts
            cross_cuts: Optional list of 3D cross-cuts
            
        Returns:
            List of OnionPiece3D objects
        """
        cutter = SequentialCutter(optimize_order=True)
        
        # Create a single mesh for the entire onion
        onion_mesh = GeometryService.create_onion_mesh_3d(onion)
        
        if onion_mesh is None or onion_mesh.is_empty:
            return []
            
        # Apply all cuts to the single onion mesh
        all_pieces = cutter.apply_cuts_sequential(
            onion_mesh,
            cuts,
            cross_cuts=cross_cuts,
            min_volume=1e-9  # Small threshold for filtering tiny pieces
        )
        
        return all_pieces
    
    @staticmethod
    def _apply_cuts_3d_legacy(
        onion: SvgProfileOnion, 
        cuts: List[Cut], 
        cross_cuts: List[CrossCut] = None
    ) -> List[OnionPiece3D]:
        """
        Apply cuts using the legacy approach (for backward compatibility).
        
        Args:
            onion: The 3D SVG profile onion model
            cuts: List of 2D cuts
            cross_cuts: Optional list of 3D cross-cuts
            
        Returns:
            List of OnionPiece3D objects
        """
        # First, apply 2D cuts to each layer profile
        layer_pieces = []
        for i, profile in enumerate(onion.svg_profiles):
            # Create a temporary 2D onion for this layer
            temp_onion = HalfOnion(
                diameter=onion.max_diameter * (profile.get_bounding_box()[1][0] - profile.get_bounding_box()[0][0]) / onion.max_width,
                n_layers=1,
                start_y=0.5,
                end_y=0.8
            )
            
            # Apply 2D cuts to this layer
            layer_cuts = GeometryService._transform_cuts_for_layer(cuts, profile, onion)
            pieces = GeometryService.apply_cuts_2d(temp_onion, layer_cuts)
            layer_pieces.append(pieces)
        
        # Then, apply cross-cuts if provided
        if cross_cuts:
            # TODO: Implement legacy cross-cut application
            pass
        
        # Create 3D pieces from the layer pieces
        pieces_3d = []
        for i, pieces in enumerate(layer_pieces):
            for piece in pieces:
                # Calculate piece properties
                volume = piece.area * onion.layer_thickness
                external_area = piece.length
                layer_area = piece.length * onion.layer_thickness
                cut_area = sum(cut.length for cut in cuts if piece.intersects(cut.as_shapely()))
                
                # Create 3D piece
                geometry = {
                    'id': len(pieces_3d),
                    'volume': volume,
                    'external_area': external_area,
                    'layer_area': layer_area,
                    'cut_area': cut_area,
                    'layer_index': i,
                    'piece_2d': piece
                }
                pieces_3d.append(OnionPiece3D(geometry))
        
        return pieces_3d
    
    @staticmethod
    def _create_layer_mesh(
        profile: 'SVGProfile', 
        onion: SvgProfileOnion, 
        layer_idx: int
    ) -> Optional[trimesh.Trimesh]:
        """
        Create a 3D mesh from a layer profile.
        
        Args:
            profile: The SVG profile for this layer
            onion: The parent onion model
            layer_idx: Index of the layer
            
        Returns:
            3D mesh representing the layer, or None if creation fails
        """
        try:
            # Get profile boundary points
            boundary_points = profile.get_boundary_points()
            
            if len(boundary_points) < 3:
                return None
            
            # Convert to 3D points with proper Z coordinates
            layer_thickness = onion.layer_thickness
            z_bottom = layer_idx * layer_thickness
            z_top = (layer_idx + 1) * layer_thickness
            
            # Create bottom and top faces
            bottom_vertices = [(x, y, z_bottom) for x, y in boundary_points]
            top_vertices = [(x, y, z_top) for x, y in boundary_points]
            
            # Combine vertices
            vertices = np.array(bottom_vertices + top_vertices)
            
            # Create faces (triangulation)
            n_points = len(boundary_points)
            faces = []
            
            # Bottom face (triangulate using fan triangulation)
            for i in range(1, n_points - 1):
                faces.append([0, i, i + 1])
            
            # Top face (triangulate using fan triangulation, reversed winding)
            for i in range(1, n_points - 1):
                faces.append([n_points, n_points + i + 1, n_points + i])
            
            # Side faces (connect bottom to top)
            for i in range(n_points):
                next_i = (i + 1) % n_points
                # Two triangles per side face
                faces.append([i, next_i, n_points + i])
                faces.append([next_i, n_points + next_i, n_points + i])
            
            faces = np.array(faces)
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Validate and repair mesh if needed
            if not mesh.is_watertight:
                mesh.fill_holes()
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
            
            return mesh
            
        except Exception as e:
            print(f"Warning: Failed to create layer mesh for layer {layer_idx}: {e}")
            return None
    
    @staticmethod
    def _transform_cuts_for_layer(cuts: List[Cut], profile: 'SVGProfile', onion: SvgProfileOnion) -> List[Cut]:
        """
        Transform cuts to match the scale and position of a specific layer profile.
        
        Args:
            cuts: List of original cuts
            profile: The layer profile to transform for
            onion: The parent onion model
            
        Returns:
            List of transformed cuts
        """
        # Get the bounding box of the profile
        (min_x, min_y), (max_x, max_y) = profile.get_bounding_box()
        profile_width = max_x - min_x
        profile_height = max_y - min_y
        
        # Calculate scaling factors
        width_scale = profile_width / onion.max_width
        height_scale = profile_height / onion.max_height
        
        # Transform each cut
        transformed_cuts = []
        for cut in cuts:
            # For vertical cuts (x is constant), scale height and maintain x position
            if abs(cut.end[0] - cut.start[0]) < 1e-10:
                start_x = cut.start[0] * width_scale
                end_x = start_x  # Keep x constant
                # Ensure Y coordinates are never negative
                start_y = max(0, cut.start[1] * height_scale)
                end_y = max(0, cut.end[1] * height_scale)
            else:
                # For horizontal cuts, scale width and maintain relative y position
                start_x = cut.start[0] * width_scale
                end_x = cut.end[0] * width_scale
                # Scale y position proportionally to maintain relative height
                # and ensure it's never negative
                y_ratio = max(0, cut.start[1] / onion.max_height)
                start_y = max(0, y_ratio * profile_height)
                end_y = start_y  # Keep y constant for horizontal cuts
            
            # Create new cut
            transformed_cuts.append(Cut((start_x, start_y), (end_x, end_y)))
        
        return transformed_cuts
    
    @staticmethod
    def create_onion_mesh_3d(onion: SvgProfileOnion) -> Optional[trimesh.Trimesh]:
        """
        Create a complete 3D mesh of the onion from all layer profiles.
        
        Args:
            onion: The 3D SVG profile onion model
            
        Returns:
            Complete 3D mesh of the onion, or None if creation fails
        """
        try:
            # Generate mesh data for the entire onion
            vertices, faces, _, _ = onion.generate_mesh()
            
            if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
                return None
            
            # Create the Trimesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Validate and repair if necessary
            if not mesh.is_watertight:
                mesh.fill_holes()
            
            if not mesh.is_volume:
                # If it's not a volume, it might be a collection of surfaces.
                # Attempt to stitch it into a single manifold mesh.
                mesh.merge_vertices()
                mesh.remove_duplicate_faces()
            
            return mesh
            
        except Exception as e:
            print(f"Warning: Failed to create complete onion mesh: {e}")
            return None
    
    @staticmethod
    def calculate_volume(piece: OnionPiece3D) -> float:
        """
        Calculate the volume of a 3D piece.
        
        Args:
            piece: The 3D piece
            
        Returns:
            Volume in cubic inches
        """
        return piece.volume
    
    @staticmethod
    def calculate_surface_areas(piece: OnionPiece3D) -> Dict[str, float]:
        """
        Calculate the surface areas for a 3D piece.
        
        Args:
            piece: The 3D piece
            
        Returns:
            Dictionary with surface areas for different types of faces
        """
        return piece.surface_areas
    
    @staticmethod
    def calculate_areas_and_shapes_2d(polygons: List[Polygon]) -> Tuple[List[float], List[Tuple[float, float, float]]]:
        """
        Calculate areas and shapes for 2D pieces.
        
        Args:
            polygons: List of Shapely Polygon objects
            
        Returns:
            Tuple of (areas, shapes) where shapes are (width, height, aspect_ratio)
        """
        areas = []
        shapes = []
        
        for polygon in polygons:
            if polygon.is_valid and polygon.area > 0:
                area = polygon.area
                bounds = polygon.bounds
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                aspect_ratio = min(width, height) / max(width, height)
                
                areas.append(area)
                shapes.append((width, height, aspect_ratio))
        
        return areas, shapes 