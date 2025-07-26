"""
Utility for classifying mesh faces in 3D onion pieces.
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import trimesh


class SurfaceClassifier:
    """
    Utility class for classifying mesh faces based on geometric properties.
    
    This classifier can identify:
    - External faces: Faces on the outer surface of the original onion
    - Layer faces: Faces on the interfaces between onion layers
    - Cut faces: Faces created by the cutting planes
    """
    
    @staticmethod
    def classify_by_normal(
        mesh: trimesh.Trimesh,
        normal_direction: Dict[str, Tuple[float, float, float]],
        tolerance: float = 0.1
    ) -> Dict[str, List[int]]:
        """
        Classify faces based on their normal vectors.
        
        Args:
            mesh: The mesh to classify
            normal_direction: Dictionary mapping face type to normal vector direction
            tolerance: Angle tolerance in cosine similarity (0 to 1)
            
        Returns:
            Dictionary mapping face types to lists of face indices
        """
        face_classes = {face_type: [] for face_type in normal_direction}
        
        # Get face normals
        face_normals = mesh.face_normals
        
        # For each face, check its normal against the provided directions
        for face_idx, normal in enumerate(face_normals):
            normal_unit = normal / np.linalg.norm(normal)
            
            # Check against each classification direction
            for face_type, direction in normal_direction.items():
                direction_unit = np.array(direction) / np.linalg.norm(direction)
                cosine_similarity = np.dot(normal_unit, direction_unit)
                
                # If the normal is aligned with the direction (within tolerance)
                if cosine_similarity > (1 - tolerance):
                    face_classes[face_type].append(face_idx)
                    break
        
        return face_classes
    
    @staticmethod
    def classify_by_position(
        mesh: trimesh.Trimesh,
        plane_equations: Dict[str, List[Tuple[float, float, float, float]]],
        tolerance: float = 0.01
    ) -> Dict[str, List[int]]:
        """
        Classify faces based on their position relative to planes.
        
        Args:
            mesh: The mesh to classify
            plane_equations: Dictionary mapping face type to list of plane equations (a, b, c, d)
                            where ax + by + cz + d = 0 defines the plane
            tolerance: Distance tolerance for considering a face on a plane
            
        Returns:
            Dictionary mapping face types to lists of face indices
        """
        face_classes = {face_type: [] for face_type in plane_equations}
        
        # Get face centroids
        face_centroids = mesh.triangles_center
        
        # For each face, check its position against the provided planes
        for face_idx, centroid in enumerate(face_centroids):
            # Check against each classification plane
            classified = False
            for face_type, planes in plane_equations.items():
                for plane in planes:
                    a, b, c, d = plane
                    distance = abs(a * centroid[0] + b * centroid[1] + c * centroid[2] + d)
                    norm = np.sqrt(a**2 + b**2 + c**2)
                    if norm > 0:
                        distance /= norm
                    
                    # If the face center is on the plane (within tolerance)
                    if distance < tolerance:
                        face_classes[face_type].append(face_idx)
                        classified = True
                        break
                
                if classified:
                    break
        
        return face_classes
    
    @staticmethod
    def classify_by_region(
        mesh: trimesh.Trimesh,
        regions: Dict[str, List[Tuple[str, Tuple[float, float, float, float], bool]]],
        tolerance: float = 0.01
    ) -> Dict[str, List[int]]:
        """
        Classify faces based on their position in 3D regions defined by planes.
        
        Args:
            mesh: The mesh to classify
            regions: Dictionary mapping face type to list of region definitions.
                   Each region is defined as (op, plane, inside) where:
                   - op is 'AND' or 'OR' (combining with previous regions)
                   - plane is (a, b, c, d) defining ax + by + cz + d = 0
                   - inside is True if the region is the inside of the plane
            tolerance: Distance tolerance for plane calculations
            
        Returns:
            Dictionary mapping face types to lists of face indices
        """
        face_classes = {face_type: [] for face_type in regions}
        
        # Get face centroids
        face_centroids = mesh.triangles_center
        
        # Track which faces have been classified
        classified_faces = set()
        
        # For each face type, apply the region tests
        for face_type, region_tests in regions.items():
            # For each face, check if it's in the region
            for face_idx, centroid in enumerate(face_centroids):
                if face_idx in classified_faces:
                    continue
                    
                # Apply the region tests
                in_region = False
                for i, (op, plane, inside) in enumerate(region_tests):
                    a, b, c, d = plane
                    distance = a * centroid[0] + b * centroid[1] + c * centroid[2] + d
                    norm = np.sqrt(a**2 + b**2 + c**2)
                    if norm > 0:
                        distance /= norm
                    
                    # Determine if point is inside the plane
                    point_inside = (distance < 0) if inside else (distance > 0)
                    
                    # For the first test or if operation is 'AND'
                    if i == 0 or op == 'AND':
                        in_region = point_inside
                    # If operation is 'OR'
                    elif op == 'OR':
                        in_region = in_region or point_inside
                    
                    # Early exit if we know it's not in the region (for AND)
                    if op == 'AND' and not in_region:
                        break
                
                # If the face is in the region for this face type
                if in_region:
                    face_classes[face_type].append(face_idx)
                    classified_faces.add(face_idx)
        
        return face_classes
    
    @staticmethod
    def classify_onion_piece(
        mesh: trimesh.Trimesh,
        cut_planes: List[Tuple[float, float, float, float]],
        layer_planes: List[Tuple[float, float, float, float]] = None,
        external_direction: Tuple[float, float, float] = (0, -1, 0)
    ) -> Dict[str, List[int]]:
        """
        Classify faces for an onion piece.
        
        Args:
            mesh: The mesh to classify
            cut_planes: List of cutting plane equations (a, b, c, d)
            layer_planes: List of layer interface plane equations (a, b, c, d)
            external_direction: Normal direction for external faces
            
        Returns:
            Dictionary mapping face types to lists of face indices
        """
        face_classes = {'external': [], 'layer': [], 'cut': []}
        
        # 1. First, try to classify by position for cut and layer faces
        if cut_planes:
            cut_classification = SurfaceClassifier.classify_by_position(
                mesh, {'cut': cut_planes}, tolerance=0.01
            )
            face_classes['cut'] = cut_classification['cut']
        
        if layer_planes:
            layer_classification = SurfaceClassifier.classify_by_position(
                mesh, {'layer': layer_planes}, tolerance=0.01
            )
            face_classes['layer'] = layer_classification['layer']
        
        # 2. Then, classify remaining faces by normal direction for external faces
        # Get all already classified faces
        classified_faces = set(face_classes['cut']).union(set(face_classes['layer']))
        
        # Get all faces
        all_faces = set(range(len(mesh.faces)))
        
        # Get remaining faces
        remaining_faces = all_faces - classified_faces
        
        # Get face normals for remaining faces
        face_normals = mesh.face_normals
        
        # Convert external direction to unit vector
        ext_dir = np.array(external_direction)
        ext_dir = ext_dir / np.linalg.norm(ext_dir)
        
        # For each remaining face, check if it's external based on normal
        for face_idx in remaining_faces:
            normal = face_normals[face_idx]
            normal_unit = normal / np.linalg.norm(normal)
            
            # Calculate cosine similarity with external direction
            cosine_similarity = np.dot(normal_unit, ext_dir)
            
            # If the normal is aligned with the external direction
            if cosine_similarity > 0.7:  # Use a threshold of cos(45°) ≈ 0.7
                face_classes['external'].append(face_idx)
            else:
                # If not classified as external, default to layer
                face_classes['layer'].append(face_idx)
        
        return face_classes 