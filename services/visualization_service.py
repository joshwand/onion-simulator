"""
Visualization service for the onion simulator.
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh
from models.svg_profile_onion import SvgProfileOnion
from models.cut import CrossCut, Cut, OnionPiece3D
import plotly.express as px
from services.geometry_service import GeometryService


class VisualizationService:
    """Service for creating visualizations of onion models and cuts."""
    
    @staticmethod
    def visualize_profile(profile: SvgProfileOnion, scale_factor: float = 1.0) -> go.Figure:
        """
        Visualize a 2D profile.
        
        Args:
            profile: The SvgProfileOnion to visualize
            scale_factor: Scale factor to apply to the profile
            
        Returns:
            Plotly figure with the profile visualization
        """
        # Get interpolated points
        interpolated = profile.get_interpolated_profile(resolution=100)
        
        # Scale points
        x_values = [x * scale_factor for x, _ in interpolated]
        y_values = [y * scale_factor for _, y in interpolated]
        
        # Create figure
        fig = go.Figure()
        
        # Add profile line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            name=f'Layer {profile.layer_index}',
            line=dict(color='green', width=2)
        ))
        
        # Add points
        original_x = [x * scale_factor for x, _ in profile.original_points]
        original_y = [y * scale_factor for _, y in profile.original_points]
        
        fig.add_trace(go.Scatter(
            x=original_x,
            y=original_y,
            mode='markers',
            name='Control Points',
            marker=dict(color='red', size=8)
        ))
        
        # Set layout
        fig.update_layout(
            title='Onion Profile',
            xaxis_title='X',
            yaxis_title='Y',
            # showlegend=True,
            height=500
        )
        
        # Ensure equal scaling
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig
    
    @staticmethod
    def visualize_all_profiles(onion: SvgProfileOnion) -> go.Figure:
        """
        Create a 2D visualization of all layer profiles.
        
        Args:
            onion: The onion model to visualize
            
        Returns:
            Plotly figure with the layer profiles
        """
        fig = go.Figure()
        
        # Plot each profile with a different color
        colors = px.colors.qualitative.Set3
        for i, profile in enumerate(onion.svg_profiles):
            # Get points from the profile
            points = profile.get_interpolated_profile(resolution=200)
            scaled_points = points * onion.scale_factor
            
            # Add the profile line
            fig.add_trace(go.Scatter(
                x=scaled_points[:, 0],
                y=scaled_points[:, 1],
                mode='lines',
                name=f'Layer {i+1}',
                line=dict(
                    color=colors[i % len(colors)],
                    width=2
                )
            ))
        
        # Update layout
        fig.update_layout(
            title="Layer Profiles",
            xaxis_title="X",
            yaxis_title="Z",  # Note: Y in SVG becomes Z in 3D
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Make the plot aspect ratio equal and add grid
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            zeroline=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.5)'
        )
        
        fig.update_xaxes(
            showgrid=True,
            zeroline=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.5)'
        )
        
        return fig
    
    @staticmethod
    def visualize_pieces(
        pieces: List[OnionPiece3D],
        title: str = "Onion Pieces"
    ) -> go.Figure:
        """
        Create a 3D visualization of onion pieces.

        Args:
            pieces: List of OnionPiece3D objects to visualize.
            title: The title for the plot.

        Returns:
            A Plotly figure containing the 3D visualization of the pieces.
        """
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, piece in enumerate(pieces):
            if piece.mesh is None or piece.mesh.is_empty:
                continue

            mesh = piece.mesh
            color = colors[i % len(colors)]

            fig.add_trace(go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color=color,
                opacity=0.9,
                name=f'Piece {i+1}'
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.5, y=-1.5, z=1.5)
                )
            ),
            showlegend=True
        )

        return fig
    
    @staticmethod
    def create_3d_visualization(
        onion: SvgProfileOnion,
        cuts: Optional[List[Cut]] = None,
        cross_cuts: Optional[List[CrossCut]] = None,
        pieces: Optional[List[OnionPiece3D]] = None
    ) -> go.Figure:
        """
        Create a 3D visualization of the onion model with optional cuts and pieces.
        
        Args:
            onion: The 3D onion model
            cuts: Optional list of 2D cuts
            cross_cuts: Optional list of 3D cross-cuts
            pieces: Optional list of 3D pieces
            
        Returns:
            Plotly figure with the 3D visualization
        """
        # Generate mesh data
        vertices, faces, normals, layer_indices = onion.generate_mesh()
        
        # Create figure
        fig = go.Figure()
        
        # Add the onion mesh
        if pieces is None:
            # If no pieces, show the whole onion with layer colors
            colors = px.colors.qualitative.Set3[:len(onion.svg_profiles)]
            vertex_colors = [colors[i] for i in layer_indices]
            
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=vertex_colors,
                name='Onion'
            ))
        else:
            # this is the old way, which is wrong. The new way is to call visualize_pieces
            pass
        
        # Add cuts if provided
        if cuts:
            for i, cut in enumerate(cuts):
                # Create a vertical plane for each cut
                plane_points = VisualizationService._create_cut_plane(cut, onion)
                
                # Ensure we have valid points before attempting to create a mesh
                if len(plane_points) == 4:  # Ensure we have exactly 4 points for a quad
                    # Create two triangles for the quad (0,1,2) and (0,2,3)
                    fig.add_trace(go.Mesh3d(
                        x=plane_points[:, 0],
                        y=plane_points[:, 1],
                        z=plane_points[:, 2],
                        i=[0, 0],
                        j=[1, 2],
                        k=[2, 3],
                        color='red',
                        opacity=0.5,
                        name=f'Cut {i+1}'
                    ))
        
        # Add cross-cuts if provided
        if cross_cuts:
            for cut in cross_cuts:
                # Create a plane to visualize the cross-cut
                plane_points = VisualizationService._create_plane_points(cut)
                fig.add_trace(go.Mesh3d(
                    x=plane_points[:, 0],
                    y=plane_points[:, 1],
                    z=plane_points[:, 2],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color='red',
                    opacity=0.5,
                    name='Cross-cut'
                ))
        
        # Update layout with camera positioned at -z, -y, -x
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.5, y=-1.5, z=-1.5)
                )
            ),
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def visualize_cross_section(
        onion: SvgProfileOnion,
        cuts: Optional[List[Cut]] = None
    ) -> go.Figure:
        """
        Create a cross-section visualization of the onion model in the XY plane.
        
        Args:
            onion: The onion model to visualize
            cuts: Optional list of cuts to show
            
        Returns:
            Plotly figure with the cross-section visualization
        """
        # print(f"VisualizationService.visualize_cross_section: onion: {onion}")
        fig = go.Figure()
        
        # Create a circle for the cross-section
        theta = np.linspace(0, np.pi, 100)  # Half circle for half-onion
        x = onion.radius * np.cos(theta)
        y = onion.radius * np.sin(theta)
        
        # Plot the outer circle
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Outer Layer',
            line=dict(color='black', width=2)
        ))
        
        # Plot each layer's circle with a different color
        colors = px.colors.qualitative.Set3
        for i in range(onion.n_layers):
            # Calculate radius for this layer using the actual profile data
            if i < len(onion.svg_profiles):
                profile = onion.svg_profiles[i]
                (min_x, _), (max_x, _) = profile.get_bounding_box()
                profile_width = max_x - min_x
                
                # Apply the same scaling that's used in the 3D model generation
                # The scale_factor ensures that the maximum width profile matches the specified diameter
                layer_radius = profile_width * onion.scale_factor  # Don't divide by 2, the profile is already half of the onion
            else:
                # raise an error
                raise ValueError(f"Profile not found for layer {i+1}")
            
            x = layer_radius * np.cos(theta)
            y = layer_radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f'Layer {i+1}',
                line=dict(
                    color=colors[i % len(colors)],
                    width=2
                )
            ))
        
        # Add cuts if provided
        if cuts:
            # print(f"VisualizationService.visualize_cross_section: cuts: {cuts}")

            for i, cut in enumerate(cuts):
                # For vertical cuts (x is constant)
                # if abs(cut.end[0] - cut.start[0]) < 1e-10:  # Vertical cut
                    x_pos = cut.start[0]
                    # Calculate y range based on onion radius
                    y_range = np.sqrt(onion.radius**2 - x_pos**2)
                    if abs(x_pos) <= onion.radius:  # Only show if cut intersects onion
                        fig.add_trace(go.Scatter(
                            x=[cut.start[0], cut.end[0]],
                            y=[cut.start[1], cut.end[1]],
                            mode='lines',
                            line=dict(color='red', width=2),
                            name=f'Cut {i+1}'
                        ))
        
        # Update layout
        fig.update_layout(
            title="Cross-Section View (XY Plane)",
            xaxis_title="X",
            yaxis_title="Y",
            height=600,
            showlegend=False,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Make the plot aspect ratio equal and add grid
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            zeroline=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.5)'
        )
        
        fig.update_xaxes(
            showgrid=True,
            zeroline=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.5)'
        )
        
        return fig
    
    @staticmethod
    def visualize_top_view(
        onion: SvgProfileOnion,
        cuts: Optional[List[Cut]] = None
    ) -> go.Figure:
        """
        Create a top view visualization of the onion model. The root end of the onion is at the 
        top of the plot, at Z=0, and the tip is in negative Z direction
        
        Args:
            onion: The onion model to visualize
            cuts: Optional list of cuts to show
            
        Returns:
            Plotly figure with the top view visualization
        """
        fig = go.Figure()
        
        # Add layers using actual profile data
        colors = px.colors.qualitative.Set3
        for i, profile in enumerate(onion.svg_profiles):
            # Get the actual profile points with proper interpolation
            points = profile.get_interpolated_profile(resolution=35)
            
            # Apply scaling
            scaled_points = points * onion.scale_factor
            
            # The X coordinate stays as X, and Y coordinate becomes Z (Y in the plot)
            # In the top view, we're looking at the X-Z plane (X-Y in the plot)
            x_values = scaled_points[:, 0]
            z_values = scaled_points[:, 1]  # Y in the original profile becomes Z


            
            # make a copy on the other side of the Z axis to show both quarters of the onion
            x_values_copy = -x_values
               
            # append the copy to the original points
            x_values = np.append(x_values, x_values_copy)
            z_values = np.append(z_values, z_values)
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=z_values,
                mode='lines',
                name=f'Layer {i+1}',
                line=dict(
                    color=colors[i % len(colors)],
                    width=2
                )
            ))

        # Add cuts if provided
        if cuts:
            for i, cut in enumerate(cuts):
                # For vertical cuts (x is constant)
                if abs(cut.end[0] - cut.start[0]) < 1e-10:  # Vertical cut
                    x_pos = cut.start[0]
                    # Calculate z range based on profile height
                    z_range = np.sqrt(onion.max_height**2 - x_pos**2)
                    fig.add_trace(go.Scatter(
                        x=[x_pos, x_pos],
                        y=[0, -z_range],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'Vertical Cut {i+1}'
                    ))
                
                # For horizontal cuts (y is constant)
                elif abs(cut.end[1] - cut.start[1]) < 1e-10:  # Horizontal cut
                    y_pos = cut.start[1]
                    # Only show if the cut is visible from this view
                    if y_pos == 0:
                        x_left = min(cut.start[0], cut.end[0])
                        x_right = max(cut.start[0], cut.end[0])
                        fig.add_trace(go.Scatter(
                            x=[x_left, x_right],
                            y=[0, 0],
                            mode='lines',
                            line=dict(color='red', width=2, dash='dot'),
                            name=f'Horizontal Cut {i+1}'
                        ))
        
        # Update layout
        fig.update_layout(
            title="Top View (X-Z Plane)", 
            yaxis_range=[-onion.max_height, 0],
            xaxis_title="X",
            yaxis_title="Z",
            height=600,
            showlegend=False,
        )
        
        # Make the plot aspect ratio equal and add grid
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            zeroline=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.5)'
        )
        
        fig.update_xaxes(
            showgrid=True,
            zeroline=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.5)'
        )
        
        return fig
    
    @staticmethod
    def _get_piece_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        layer_indices: np.ndarray,
        piece: OnionPiece3D
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the mesh data for a specific piece.
        
        Args:
            vertices: All vertices
            faces: All faces
            layer_indices: Layer indices for each vertex
            piece: The piece to get mesh data for
            
        Returns:
            Tuple of (piece_vertices, piece_faces)
        """
        # Get vertices belonging to this piece
        piece_vertex_indices = np.where(layer_indices == piece.layer_index)[0]
        
        # Create mapping from original to new vertex indices
        vertex_map = {old: new for new, old in enumerate(piece_vertex_indices)}
        
        # Get faces that belong to this piece
        piece_faces = []
        for face in faces:
            if all(v in piece_vertex_indices for v in face):
                piece_faces.append([vertex_map[v] for v in face])
        
        return vertices[piece_vertex_indices], np.array(piece_faces)
    
    @staticmethod
    def _create_cut_plane(cut: Cut, onion: SvgProfileOnion) -> np.ndarray:
        """
        Create points for visualizing a cut plane.
        
        Args:
            cut: The cut to visualize (defined in 2D XY plane)
            onion: The onion model to use for geometry calculations
            
        Returns:
            Array of points defining the plane in 3D space
        """
        # print(f"VisualizationService._create_cut_plane: cut: {cut}")
        radius = onion.radius
        # We want the planes to extend beyond the onion
        plane_size_xy = radius # Width in XY plane
        plane_size_z = onion.max_height
        
        # For vertical cuts (x is constant)
        if abs(cut.end[0] - cut.start[0]) < 1e-10:
            # Scale x coordinate by radius (without clamping)
            x_scaled = cut.start[0] # * radius
            
            # Create a vertical plane
            points = np.array([
                [x_scaled, 0, 0],                # Bottom front
                [x_scaled, -plane_size_xy, 0],   # Top front
                [x_scaled, -plane_size_xy, -plane_size_z],  # Top back
                [x_scaled, 0, -plane_size_z]     # Bottom back
            ])
        else:
            # For horizontal cuts - scale coordinates by radius without clamping
            # Apply the same scaling used for vertical cuts
            x1_scaled = cut.start[0] # * radius
            x2_scaled = cut.end[0] # * radius
            
            # Scale y coordinate by radius
            y_height = -cut.end[1] # * radius  # Negative to invert Y axis
            # print(f"VisualizationService._create_cut_plane: y_height: {y_height:.3f}")
            
            # Make sure horizontal cuts aren't being clamped
            # Use the full plane width for visualization
            # This matches how vertical cuts are handled
            points = np.array([
                [x1_scaled, y_height, 0],             # Front left
                [x1_scaled, y_height, -plane_size_z], # Back left
                [x2_scaled, y_height, -plane_size_z], # Back right
                [x2_scaled, y_height, 0]              # Front right
            ])
        
        return points
    
    @staticmethod
    def _create_plane_points(cut: CrossCut) -> np.ndarray:
        """
        Create points for visualizing a cross-cut plane.
        The plane will only exist in the y≤0 region.
        
        Args:
            cut: The cross-cut to visualize
            
        Returns:
            Array of points defining the plane
        """
        # Create a plane perpendicular to the normal vector
        normal = np.array(cut.normal)
        point = np.array(cut.point)
        
        # Find two vectors perpendicular to the normal
        v1 = np.cross(normal, [1, 0, 0])
        if np.allclose(v1, 0):
            v1 = np.cross(normal, [0, 1, 0])
        v1 = v1 / np.linalg.norm(v1)
        
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Create four points to define the plane, but only in y≤0 region
        scale = 2.0  # Adjust this to change the size of the plane
        
        # Create initial points
        points = np.array([
            point + scale * (v1 + v2),
            point + scale * (v1 - v2),
            point + scale * (-v1 - v2),
            point + scale * (-v1 + v2)
        ])
        
        # Clamp y coordinates to be ≤0
        points[:, 1] = np.minimum(points[:, 1], 0)
        
        return points 