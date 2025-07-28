"""
Utility for extracting separate pieces after slicing a mesh with multiple planes.
"""

from typing import List, Dict, Optional, Set, Tuple
import numpy as np
import trimesh
from models.cut import OnionPiece3D
from models.surface_classifier import SurfaceClassifier


class PieceExtractor:
    """
    Utility class for extracting separate pieces after slicing a mesh.
    
    This class provides functionality to:
    - Identify connected components in the mesh after slicing
    - Extract each component as a separate OnionPiece3D object
    - Preserve face classifications and other metadata
    - Filter out degenerate or extremely small pieces
    """
    
    @staticmethod
    def extract_pieces(
        sliced_meshes: List[trimesh.Trimesh],
        face_labels: List[Dict[str, List[int]]],
        cut_planes: List[Tuple[float, float, float, float]] = None,
        layer_planes: List[Tuple[float, float, float, float]] = None,
        layer_index: int = 0,
        min_volume: float = 1e-6,
        min_area: float = 1e-6
    ) -> List[OnionPiece3D]:
        """
        Extract separate pieces from sliced meshes.
        
        Args:
            sliced_meshes: List of mesh pieces after slicing
            face_labels: List of face classification dictionaries for each mesh
            cut_planes: List of cutting plane equations (a, b, c, d)
            layer_planes: List of layer interface plane equations (a, b, c, d)
            layer_index: Layer index for the pieces
            min_volume: Minimum volume threshold for filtering small pieces
            min_area: Minimum surface area threshold for filtering small pieces
            
        Returns:
            List of OnionPiece3D objects
        """
        if not sliced_meshes:
            return []
        
        pieces = []
        cut_planes = cut_planes or []
        layer_planes = layer_planes or []
        
        # Process each sliced mesh
        for mesh_idx, mesh in enumerate(sliced_meshes):
            # Skip empty or invalid meshes
            if not mesh or mesh.is_empty or len(mesh.faces) == 0:
                continue
            
            # Get corresponding face labels
            mesh_face_labels = face_labels[mesh_idx] if mesh_idx < len(face_labels) else {
                'cut': [], 'layer': [], 'external': []
            }
            
            # Identify connected components in this mesh
            components = PieceExtractor.identify_connected_components(mesh)
            
            # Extract each component as a separate piece
            for component_idx, component in enumerate(components):
                # Filter out degenerate pieces
                if not PieceExtractor._is_valid_piece(component, min_volume, min_area):
                    continue
                
                # Map face labels from original mesh to component
                component_face_labels = PieceExtractor._map_face_labels(
                    mesh, component, mesh_face_labels
                )
                
                # Create piece geometry data
                geometry = {
                    'id': len(pieces),
                    'layer_index': layer_index,
                }
                
                # Create OnionPiece3D object
                piece = OnionPiece3D(
                    geometry=geometry,
                    mesh=component,
                    face_classes=component_face_labels,
                    cut_planes=cut_planes,
                    layer_planes=layer_planes
                )
                
                pieces.append(piece)
        
        return pieces
    
    @staticmethod
    def identify_connected_components(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
        """
        Identify connected components in a mesh.
        
        Args:
            mesh: The mesh to analyze
            
        Returns:
            List of trimesh objects, one for each connected component
        """
        if not mesh or mesh.is_empty or len(mesh.faces) == 0:
            return []
        
        try:
            # Use trimesh's built-in connected components functionality
            components = mesh.split(only_watertight=False)
            
            # If split returns nothing, return the original mesh as single component
            if not components:
                return [mesh]
            
            # Filter out empty components
            valid_components = []
            for component in components:
                if (component is not None and 
                    not component.is_empty and 
                    len(component.faces) > 0):
                    valid_components.append(component)
            
            # If no valid components found, return original mesh
            if not valid_components:
                return [mesh]
            
            return valid_components
            
        except Exception as e:
            # If splitting fails, return the original mesh as single component
            print(f"Warning: Connected component analysis failed: {e}")
            return [mesh]
    
    @staticmethod
    def _is_valid_piece(mesh: trimesh.Trimesh, min_volume: float, min_area: float) -> bool:
        """
        Check if a mesh piece meets the minimum size requirements.
        
        Args:
            mesh: The mesh to check
            min_volume: Minimum volume threshold
            min_area: Minimum surface area threshold
            
        Returns:
            True if the piece is valid
        """
        if not mesh or mesh.is_empty or len(mesh.faces) == 0:
            return False
        
        try:
            # Check volume if mesh is watertight
            if mesh.is_watertight:
                volume = abs(mesh.volume)
                if volume < min_volume:
                    return False
            else:
                # For non-watertight meshes, use bounding box volume as approximation
                if hasattr(mesh, 'extents'):
                    bbox_volume = np.prod(mesh.extents)
                    if bbox_volume < min_volume * 10:  # More lenient for non-watertight
                        return False
            
            # Check surface area
            try:
                area = mesh.area
                if area < min_area:
                    return False
            except Exception:
                # If area calculation fails, use face count as proxy
                if len(mesh.faces) < 3:  # Need at least 3 faces for a valid 3D piece
                    return False
            
            return True
            
        except Exception:
            # If any calculation fails, be conservative and keep the piece
            return len(mesh.faces) >= 3
    
    @staticmethod
    def _map_face_labels(
        original_mesh: trimesh.Trimesh,
        component_mesh: trimesh.Trimesh,
        original_labels: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        """
        Map face labels from the original mesh to a component mesh.
        
        Args:
            original_mesh: The original mesh before component extraction
            component_mesh: The extracted component mesh
            original_labels: Face labels for the original mesh
            
        Returns:
            Face labels mapped to the component mesh
        """
        component_labels = {'cut': [], 'layer': [], 'external': []}
        
        if not original_labels or not original_mesh or not component_mesh:
            return component_labels
        
        try:
            # Get centroids of both meshes for matching
            orig_centroids = original_mesh.triangles_center
            comp_centroids = component_mesh.triangles_center
            
            # For each face in the component, find the closest face in the original
            for comp_face_idx, comp_centroid in enumerate(comp_centroids):
                # Find closest face in original mesh
                distances = np.linalg.norm(orig_centroids - comp_centroid, axis=1)
                closest_orig_face = np.argmin(distances)
                
                # Only map if the distance is very small (same face)
                if distances[closest_orig_face] < 1e-6:
                    # Find which label this face had in the original
                    for label_type, face_list in original_labels.items():
                        if closest_orig_face in face_list:
                            component_labels[label_type].append(comp_face_idx)
                            break
                    else:
                        # If no label found, default to external
                        component_labels['external'].append(comp_face_idx)
                else:
                    # If no close match, classify based on position and normal
                    component_labels['external'].append(comp_face_idx)
            
            return component_labels
            
        except Exception as e:
            print(f"Warning: Face label mapping failed: {e}")
            # Fallback: classify all faces as external
            component_labels['external'] = list(range(len(component_mesh.faces)))
            return component_labels
    
    @staticmethod
    def extract_pieces_sequential(
        original_mesh: trimesh.Trimesh,
        cut_sequence: List[Tuple[float, float, float, float]],
        layer_planes: List[Tuple[float, float, float, float]] = None,
        layer_index: int = 0,
        min_volume: float = 1e-6
    ) -> List[OnionPiece3D]:
        """
        Extract pieces by applying cuts sequentially.
        
        Args:
            original_mesh: The original mesh to cut
            cut_sequence: Sequence of cutting planes to apply
            layer_planes: List of layer interface plane equations
            layer_index: Layer index for the pieces
            min_volume: Minimum volume threshold for filtering
            
        Returns:
            List of OnionPiece3D objects after all cuts are applied
        """
        if not cut_sequence:
            # No cuts, return original mesh as single piece
            geometry = {'id': 0, 'layer_index': layer_index}
            face_labels = {'cut': [], 'layer': [], 'external': list(range(len(original_mesh.faces)))}
            piece = OnionPiece3D(
                geometry=geometry,
                mesh=original_mesh,
                face_classes=face_labels,
                cut_planes=[],
                layer_planes=layer_planes or []
            )
            return [piece]
        
        # Apply cuts sequentially
        current_meshes = [original_mesh]
        current_labels = [{'cut': [], 'layer': [], 'external': list(range(len(original_mesh.faces)))}]
        applied_cuts = []
        
        for cut_plane in cut_sequence:
            applied_cuts.append(cut_plane)
            new_meshes = []
            new_labels = []
            
            # Apply current cut to all existing pieces
            for mesh, labels in zip(current_meshes, current_labels):
                try:
                    from models.plane_slicing import PlaneSlicing
                    sliced_meshes, face_labels = PlaneSlicing.slice_mesh_with_labels(
                        mesh, cut_plane
                    )
                    
                    new_meshes.extend(sliced_meshes)
                    new_labels.extend(face_labels)
                    
                except Exception as e:
                    print(f"Warning: Slicing failed for cut {cut_plane}: {e}")
                    # Keep original mesh if slicing fails
                    new_meshes.append(mesh)
                    new_labels.append(labels)
            
            current_meshes = new_meshes
            current_labels = new_labels
        
        # Extract final pieces
        return PieceExtractor.extract_pieces(
            current_meshes,
            current_labels,
            cut_planes=applied_cuts,
            layer_planes=layer_planes,
            layer_index=layer_index,
            min_volume=min_volume
        )