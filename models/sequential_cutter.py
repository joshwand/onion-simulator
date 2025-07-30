"""
Sequential cutting implementation for applying multiple cuts to a 3D mesh.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import trimesh
from models.cut import Cut, CrossCut, OnionPiece3D
from models.plane_slicing import PlaneSlicing
from models.piece_extractor import PieceExtractor


class SequentialCutter:
    """
    Class for applying multiple cuts (both regular cuts and cross-cuts) to a 3D mesh sequentially.
    
    This class provides functionality to:
    - Apply a list of Cut objects to a 3D mesh in sequence
    - Apply CrossCut objects correctly
    - Generate all resulting pieces as OnionPiece3D objects
    - Maintain face classifications through multiple cuts
    - Optimize cutting order for efficiency
    """
    
    def __init__(self, optimize_order: bool = True):
        """
        Initialize the sequential cutter.
        
        Args:
            optimize_order: Whether to optimize the cutting order for efficiency
        """
        self.optimize_order = optimize_order
        self.cut_history = []
        
    def apply_cuts_sequential(
        self,
        original_mesh: trimesh.Trimesh,
        cuts: List[Cut],
        cross_cuts: List[CrossCut] = None,
        layer_planes: List[Tuple[float, float, float, float]] = None,
        layer_index: int = 0,
        min_volume: float = 1e-6
    ) -> List[OnionPiece3D]:
        """
        Apply cuts sequentially to a mesh and return resulting pieces.
        
        Args:
            original_mesh: The original mesh to cut
            cuts: List of 2D cuts to apply
            cross_cuts: Optional list of 3D cross-cuts
            layer_planes: List of layer interface plane equations
            layer_index: Layer index for the pieces
            min_volume: Minimum volume threshold for filtering
            
        Returns:
            List of OnionPiece3D objects after all cuts are applied
        """
        if not cuts and not cross_cuts:
            # No cuts, return original mesh as single piece
            return self._create_single_piece(original_mesh, layer_planes, layer_index)
        
        # Convert 2D cuts to 3D cutting planes
        cut_planes = self._convert_cuts_to_planes(cuts, layer_index)
        
        # Add cross-cuts as planes
        if cross_cuts:
            cut_planes.extend(self._convert_cross_cuts_to_planes(cross_cuts))
        
        # Optimize cutting order if requested
        if self.optimize_order:
            cut_planes = self._optimize_cutting_order(original_mesh, cut_planes)
        
        # Apply cuts sequentially
        return self._apply_planes_sequential(
            original_mesh, cut_planes, layer_planes, layer_index, min_volume
        )
    
    def apply_cuts_to_pieces(
        self,
        pieces: List[OnionPiece3D],
        cuts: List[Cut],
        cross_cuts: List[CrossCut] = None,
        min_volume: float = 1e-6
    ) -> List[OnionPiece3D]:
        """
        Apply additional cuts to existing pieces.
        
        Args:
            pieces: Existing pieces to cut further
            cuts: List of 2D cuts to apply
            cross_cuts: Optional list of 3D cross-cuts
            min_volume: Minimum volume threshold for filtering
            
        Returns:
            List of OnionPiece3D objects after applying additional cuts
        """
        all_pieces = []
        
        for piece in pieces:
            # Apply cuts to this piece
            piece_cuts = self.apply_cuts_sequential(
                piece.mesh,
                cuts,
                cross_cuts,
                piece.layer_planes,
                piece.layer_index,
                min_volume
            )
            all_pieces.extend(piece_cuts)
        
        return all_pieces
    
    def _convert_cuts_to_planes(self, cuts: List[Cut], layer_index: int = 0) -> List[Tuple[float, float, float, float]]:
        """
        Convert 2D cuts to 3D cutting planes.
        
        Args:
            cuts: List of 2D cuts
            layer_index: Layer index for Z-coordinate handling
            
        Returns:
            List of plane equations (a, b, c, d) where ax + by + cz + d = 0
        """
        planes = []
        
        for cut in cuts:
            # For 2D cuts, we need to convert to 3D planes
            # Assuming the cut is in the XY plane and extends through Z
            start_point = np.array([cut.start[0], cut.start[1], 0])  # Fixed: use tuple indexing
            end_point = np.array([cut.end[0], cut.end[1], 0])        # Fixed: use tuple indexing
            
            # Calculate the direction vector of the cut
            direction = end_point - start_point
            direction_norm = np.linalg.norm(direction[:2])
            
            if direction_norm > 1e-10:  # Avoid division by zero
                # Normal vector perpendicular to the cut in XY plane
                normal = np.array([-direction[1], direction[0], 0])
                normal = normal / np.linalg.norm(normal)
                
                # Calculate plane equation: normal · (point - start_point) = 0
                # Which becomes: a*x + b*y + c*z + d = 0
                d = -np.dot(normal, start_point)
                plane = (normal[0], normal[1], normal[2], d)
                planes.append(plane)
        
        return planes
    
    def _convert_cross_cuts_to_planes(self, cross_cuts: List[CrossCut]) -> List[Tuple[float, float, float, float]]:
        """
        Convert 3D cross-cuts to cutting planes.
        
        Args:
            cross_cuts: List of 3D cross-cuts
            
        Returns:
            List of plane equations (a, b, c, d)
        """
        planes = []
        
        for cross_cut in cross_cuts:
            # CrossCut should have a plane or points defining the cutting plane
            if hasattr(cross_cut, 'plane') and cross_cut.plane:
                planes.append(cross_cut.plane)
            elif hasattr(cross_cut, 'normal') and hasattr(cross_cut, 'point'):
                # Calculate plane from normal and point
                normal = np.array(cross_cut.normal)
                point = np.array(cross_cut.point)
                normal = normal / np.linalg.norm(normal)  # Normalize
                d = -np.dot(normal, point)
                planes.append((normal[0], normal[1], normal[2], d))
            elif hasattr(cross_cut, 'points') and len(cross_cut.points) >= 3:
                # Calculate plane from three points
                p1, p2, p3 = cross_cut.points[:3]
                v1 = np.array(p2) - np.array(p1)
                v2 = np.array(p3) - np.array(p1)
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                d = -np.dot(normal, np.array(p1))
                planes.append((normal[0], normal[1], normal[2], d))
        
        return planes
    
    def _optimize_cutting_order(
        self,
        mesh: trimesh.Trimesh,
        planes: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[float, float, float, float]]:
        """
        Optimize the order of cutting planes for efficiency.
        
        Args:
            mesh: The mesh to be cut
            planes: List of cutting planes
            
        Returns:
            Optimized list of cutting planes
        """
        if len(planes) <= 1:
            return planes
        
        # Simple optimization: sort by how much of the mesh each plane cuts
        # More sophisticated optimizations could consider plane intersections
        plane_scores = []
        
        for plane in planes:
            try:
                # Estimate how much this plane divides the mesh
                # by checking how many vertices are on each side
                a, b, c, d = plane
                distances = mesh.vertices @ np.array([a, b, c]) + d
                
                # Count vertices on each side
                positive = np.sum(distances > 1e-10)
                negative = np.sum(distances < -1e-10)
                
                # Score based on how evenly the plane divides the mesh
                # Planes that split more evenly get higher priority
                total = len(distances)
                if total > 0:
                    balance = 1.0 - abs(positive - negative) / total
                    score = balance * min(positive, negative)
                else:
                    score = 0
                
                plane_scores.append((score, plane))
            except Exception:
                # If calculation fails, give low priority
                plane_scores.append((0, plane))
        
        # Sort by score (descending) and return the planes
        plane_scores.sort(key=lambda x: x[0], reverse=True)
        return [plane for score, plane in plane_scores]
    
    def _apply_planes_sequential(
        self,
        original_mesh: trimesh.Trimesh,
        planes: List[Tuple[float, float, float, float]],
        layer_planes: List[Tuple[float, float, float, float]] = None,
        layer_index: int = 0,
        min_volume: float = 1e-6
    ) -> List[OnionPiece3D]:
        """
        Apply cutting planes sequentially to a mesh.
        
        Args:
            original_mesh: The original mesh to cut
            planes: List of cutting planes to apply
            layer_planes: List of layer interface plane equations
            layer_index: Layer index for the pieces
            min_volume: Minimum volume threshold for filtering
            
        Returns:
            List of OnionPiece3D objects after all cuts are applied
        """
        # Start with the original mesh
        current_meshes = [original_mesh]
        current_labels = [{'cut': [], 'layer': [], 'external': list(range(len(original_mesh.faces)))}]
        applied_planes = []
        
        # Apply each plane sequentially
        for plane in planes:
            applied_planes.append(plane)
            new_meshes = []
            new_labels = []
            
            # Apply current plane to all existing pieces
            for mesh, labels in zip(current_meshes, current_labels):
                try:
                    sliced_meshes, face_labels = PlaneSlicing.slice_mesh_with_labels(
                        mesh, plane
                    )
                    
                    # Add sliced meshes and their labels
                    new_meshes.extend(sliced_meshes)
                    new_labels.extend(face_labels)
                    
                except Exception as e:
                    print(f"Warning: Slicing failed for plane {plane}: {e}")
                    # Keep original mesh if slicing fails
                    new_meshes.append(mesh)
                    new_labels.append(labels)
            
            current_meshes = new_meshes
            current_labels = new_labels
            
            # Store cutting progress in history
            self.cut_history.append({
                'plane': plane,
                'pieces_count': len(current_meshes),
                'applied_planes': applied_planes.copy()
            })
        
        # Extract final pieces using PieceExtractor
        return PieceExtractor.extract_pieces(
            current_meshes,
            current_labels,
            cut_planes=applied_planes,
            layer_planes=layer_planes or [],
            layer_index=layer_index,
            min_volume=min_volume
        )
    
    def _create_single_piece(
        self,
        mesh: trimesh.Trimesh,
        layer_planes: List[Tuple[float, float, float, float]] = None,
        layer_index: int = 0
    ) -> List[OnionPiece3D]:
        """
        Create a single piece from an uncut mesh.
        
        Args:
            mesh: The mesh to create a piece from
            layer_planes: List of layer interface plane equations
            layer_index: Layer index for the piece
            
        Returns:
            List containing a single OnionPiece3D object
        """
        geometry = {'id': 0, 'layer_index': layer_index}
        face_labels = {'cut': [], 'layer': [], 'external': list(range(len(mesh.faces)))}
        
        piece = OnionPiece3D(
            geometry=geometry,
            mesh=mesh,
            face_classes=face_labels,
            cut_planes=[],
            layer_planes=layer_planes or []
        )
        
        return [piece]
    
    def get_cutting_stats(self) -> Dict:
        """
        Get statistics about the cutting process.
        
        Returns:
            Dictionary with cutting statistics
        """
        if not self.cut_history:
            return {'total_cuts': 0, 'final_pieces': 0}
        
        return {
            'total_cuts': len(self.cut_history),
            'final_pieces': self.cut_history[-1]['pieces_count'] if self.cut_history else 0,
            'cut_history': self.cut_history
        }
    
    def reset_history(self):
        """Reset the cutting history."""
        self.cut_history = []