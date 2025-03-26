"""
2D Onion Simulator page.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import traceback

from core.config import (
    DEFAULT_2D_ONION_DIAMETER, DEFAULT_2D_ONION_LAYERS, 
    DEFAULT_2D_START_Y, DEFAULT_2D_END_Y, DEFAULT_2D_P1, DEFAULT_2D_P2
)
from models.onion_2d import HalfOnion
from models.cut import Cut
from cutting_methods.classic import ClassicCuttingMethod
from cutting_methods.kenji import KenjiCuttingMethod
from cutting_methods.josh import JoshCuttingMethod
from services.geometry_service import GeometryService
from services.analysis_service import AnalysisService
from services.persistence_service import PersistenceService
from cutting_methods.base import CuttingMethodFactory


# Setup logging
logger = logging.getLogger(__name__)


def visualize_onion_and_cuts(onion, cuts):
    """
    Visualize the onion with cuts using Plotly.
    
    Args:
        onion: HalfOnion object
        cuts: List of Cut objects
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Draw onion outline
    theta = np.linspace(0, np.pi, 100)
    x = onion.radius * np.cos(theta)
    y = onion.radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), mode='lines', name='Onion outline', 
                            line=dict(color='black'), hoverinfo='skip'))

    # Draw layers
    for r in onion.layer_radii[1:]:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        fig.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), 
                                mode='lines', 
                                line=dict(dash='dot', color='gray'), 
                                name=f'Layer (r={r:.2f})', 
                                hoverinfo='skip'))

    # Draw cuts
    for i, cut in enumerate(cuts):
        fig.add_shape(
            type="line",
            x0=cut.start[0], y0=cut.start[1],
            x1=cut.end[0], y1=cut.end[1],
            line=dict(color="red", width=2),
            editable=True,            
        )

    # Calculate the range for both axes
    xrange = [-onion.radius, onion.radius+0.5]
    yrange = [0, onion.radius+0.5]

    # Update layout
    fig.update_layout(
        title='Onion with Cuts (Interactive)',
        autosize=True,
        xaxis_title='X',
        yaxis_title='Y',
        showlegend=False,
        height=300,
        modebar_remove=['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 
                        'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 
                        'hoverCompareCartesian', 'toggleSpikelines', 'download'],
        modebar_add=['drawline','eraseshape'],
        newshape=dict(line=dict(color='red', width=2)),
        newselection=dict(mode='immediate'),
        margin=dict(l=0, r=0, b=0, t=30),
        xaxis=dict(
            range=xrange,
            scaleanchor="y",
            scaleratio=1,
            visible=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=yrange,
            visible=False,
            fixedrange=True
        ),
    )
    
    return fig


def visualize_pieces(polygons, onion):
    """
    Visualize the resulting pieces using Plotly.
    
    Args:
        polygons: List of Shapely Polygon objects
        onion: HalfOnion object
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for i, polygon in enumerate(polygons):
        x, y = polygon.exterior.xy
        color = plt.cm.tab10(i % 10)
        fig.add_trace(go.Scatter(
            x=list(x), y=list(y), 
            fill='toself', 
            fillcolor=f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.6)',
            line=dict(color='white', width=1),
            mode='lines',
        ))

    fig.update_layout(
        height=350,
        width=500,
        showlegend=False,
        title_text="Resulting Pieces",
        yaxis=dict(range=[0, onion.radius+0.5]),
        margin=dict(l=0, r=0, b=0, t=30),        
    )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig


def visualize_piece_shapes(polygons, title):
    """
    Visualize piece shapes in a grid using Plotly.
    
    Args:
        polygons: List of Shapely Polygon objects
        title: Title for the plot
        
    Returns:
        Plotly figure or error message
    """
    if not polygons:
        return "No pieces to visualize."

    sorted_polygons = sorted(polygons, key=lambda p: p.area, reverse=True)
    n_pieces = len(sorted_polygons)
    n_cols = min(15, n_pieces)
    n_rows = (n_pieces + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, horizontal_spacing=0.001, vertical_spacing=0.001)

    # Find the maximum dimension across all pieces
    max_dimension = max(max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]) for p in sorted_polygons)

    for i, polygon in enumerate(sorted_polygons):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Scale and center the polygon
        bounds = polygon.bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        
        # Create a new polygon shifted to center (can't use affine_transform directly)
        shifted_x = [x - center_x for x in polygon.exterior.xy[0]]
        shifted_y = [y - center_y for y in polygon.exterior.xy[1]]
        
        # Scale the shifted polygon
        scale_factor = 0.98 / max_dimension
        scaled_x = [x * scale_factor for x in shifted_x]
        scaled_y = [y * scale_factor for y in shifted_y]

        color = plt.cm.tab10(i % 10)
        fig.add_trace(go.Scatter(
            x=scaled_x, y=scaled_y,
            fill='toself',
            fillcolor=f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.6)',
            line=dict(color='white', width=1),
            mode='lines',
        ), row=row, col=col)

        fig.update_xaxes(range=[-0.5, 0.5], row=row, col=col, showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(range=[-0.5, 0.5], row=row, col=col, showticklabels=False, showgrid=False, zeroline=False)
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=row, col=col)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=row, col=col)

    fig.update_layout(
        height=200*n_rows,
        width=1000,
        showlegend=False,
        title_text=f"{title} - Piece Shapes (Largest to Smallest)",
        margin=dict(l=0, r=0, b=0, t=30),
    )

    return fig


def display_piece_area_distribution(areas):
    """
    Display a histogram of piece areas.
    
    Args:
        areas: List of piece areas
    """
    st.subheader("Piece Area Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    
    ax_hist.hist(areas, bins=20)
    ax_hist.set_title("Distribution of Piece Areas")
    ax_hist.set_xlabel('Area (sq inches)')
    ax_hist.set_ylabel('Frequency')
    median_area = np.median(areas)
    std_dev_area = np.std(areas)
    plt.tight_layout()
    st.pyplot(fig_hist)
    st.caption(f"Median: {median_area:.4f} sq inches, Std Dev: {std_dev_area:.4f} sq inches")


def display_aspect_ratio_distribution(shapes):
    """
    Display a histogram of piece aspect ratios.
    
    Args:
        shapes: List of shape tuples (width, height, aspect_ratio)
    """
    st.subheader("Piece Aspect Ratio Distribution")
    fig_square, ax_square = plt.subplots(figsize=(8, 5))
    ax_square.hist([s[2] for s in shapes], bins=20)
    ax_square.set_title("Distribution of Piece Aspect Ratio")
    ax_square.set_xlabel('Aspect Ratio (1 is perfect square)')
    ax_square.set_ylabel('Frequency')
    st.pyplot(fig_square)


def display_piece_statistics(areas, onion):
    """
    Display statistics about the pieces.
    
    Args:
        areas: List of piece areas
        onion: HalfOnion object
    """
    st.subheader("Piece Statistics")
    
    st.markdown(f"""
    Number of pieces: {len(areas)}  
    Average piece area: {np.mean(areas):.4f} sq inches  
    Std dev of areas: {np.std(areas):.4f} sq inches  
    Normalized piece std dev: {np.std([area for area in areas if area > 0.01]) / (np.pi * onion.radius**2):.4f} %
    """)


def select_cutting_method():
    """
    Display UI for selecting a cutting method.
    
    Returns:
        Selected cutting method name
    """
    st.sidebar.header("Cutting Method")
    cutting_method = st.sidebar.selectbox(
        "Select Cutting Method", 
        ["Josh's Method", "Classic", "Kenji", "Custom"],
        index=0
    )
    return cutting_method


def generate_cuts_menu(onion, cutting_method):
    """
    Generate cuts based on the selected method and parameters.
    
    Args:
        onion: HalfOnion object
        cutting_method: Selected cutting method name
        
    Returns:
        List of Cut objects
    """
    cuts = []
    
    if cutting_method == "Josh's Method":
        n_horizontal = st.sidebar.slider("Number of Horizontal Cuts", 2, 10, 3)
        n_vertical = st.sidebar.slider("Number of Vertical Cuts", 3, 20, 10)
        horizontal_depth = st.sidebar.slider("Horizontal Cut Depth (fraction of radius)", 0.1, 1.0, 0.85)
        vertical_height = st.sidebar.slider("Vertical Cut Depth (fraction of radius)", 0.1, 1.0, 0.17)
        
        josh_method = JoshCuttingMethod(onion)
        cuts = josh_method.generate_cuts(
            n_horizontal=n_horizontal,
            n_vertical=n_vertical,
            horizontal_depth=horizontal_depth,
            vertical_height=vertical_height
        )
    
    elif cutting_method == "Classic":
        n_vertical = st.sidebar.slider("Number of Vertical Cuts", 4, 16, 10)
        n_horizontal = st.sidebar.slider("Number of Horizontal Cuts", 0, 10, 2)
        
        classic_method = ClassicCuttingMethod(onion)
        cuts = classic_method.generate_cuts(
            n_vertical=n_vertical,
            n_horizontal=n_horizontal
        )
    
    elif cutting_method == "Kenji":
        n_cuts = st.sidebar.slider("Number of Cuts", 3, 20, 10)
        pct_below = st.sidebar.slider("Target Point (fraction of radius below center)", 0.1, 0.9, 0.6)
        
        kenji_method = KenjiCuttingMethod(onion)
        cuts = kenji_method.generate_cuts(
            n_cuts=n_cuts,
            pct_below=pct_below
        )
    
    elif cutting_method == "Custom" and "cuts" in st.session_state:
        cuts = st.session_state.cuts if st.session_state.cuts else []
        
    logger.info(f"Generated {len(cuts)} cuts using {cutting_method} method")
    return cuts


def main():
    """Main function for the 2D simulator page."""
    st.set_page_config(
        page_title="2D Onion Simulator",
        page_icon="🧅",
        layout="wide"
    )
    
    st.title("2D Onion Simulator")
    
    # Initialize services and state
    PersistenceService.initialize_session_state()
    
    # Instructions
    with st.expander("Instructions"):
        st.caption("This app simulates cutting an onion into pieces. Make adjustments in the sidebar on the left.")
        st.caption("I believe I've found a novel method (Josh's Method) for cutting onions. It's statistically better than any other method I've heard of. You can compare to Kenji's method or the classic method in the sidebar.")
        st.caption("Drag the red lines to adjust cuts. Stats will update automatically.")
        st.caption("Unfortunately there's no way to delete cuts yet; just drag the line outside the onion.")
        st.caption("As you make changes, the URL will be updated. You can share the URL with others to share your preferred onion cuts.")
    
    # Try to load settings from URL
    try:
        onion, cuts, cutting_method = PersistenceService.decode_settings_from_url()
        
        if onion is None:
            # Use default settings
            onion = HalfOnion(
                DEFAULT_2D_ONION_DIAMETER,
                DEFAULT_2D_ONION_LAYERS,
                DEFAULT_2D_START_Y,
                DEFAULT_2D_END_Y,
                DEFAULT_2D_P1,
                DEFAULT_2D_P2
            )
        
        if cuts is None:
            cuts = []
        
        if cutting_method is None:
            cutting_method = "Josh's Method"
        
        # Store in session state
        st.session_state.onion_2d = onion
        st.session_state.cuts = cuts
        st.session_state.cutting_method = cutting_method
    
    except Exception as e:
        logger.error(f"Error loading settings from URL: {str(e)}")
        st.error(f"Error loading settings from URL: {str(e)}")
        
        # Use default settings
        if 'onion_2d' not in st.session_state or st.session_state.onion_2d is None:
            st.session_state.onion_2d = HalfOnion(
                DEFAULT_2D_ONION_DIAMETER,
                DEFAULT_2D_ONION_LAYERS,
                DEFAULT_2D_START_Y,
                DEFAULT_2D_END_Y,
                DEFAULT_2D_P1,
                DEFAULT_2D_P2
            )
        
        if 'cuts' not in st.session_state or st.session_state.cuts is None:
            st.session_state.cuts = []
        
        if 'cutting_method' not in st.session_state or st.session_state.cutting_method is None:
            st.session_state.cutting_method = "Josh's Method"
    
    # Sidebar: Onion parameters
    with st.sidebar:
        st.sidebar.header("Onion Parameters")
        onion_diameter = st.sidebar.slider(
            "Onion Diameter (inches)", 
            1.0, 10.0, 
            st.session_state.onion_2d.radius * 2, 
            0.1
        )
        
        n_layers = st.sidebar.slider(
            "Number of Layers", 
            3, 20, 
            st.session_state.onion_2d.n_layers, 
            1
        )
        
        with st.expander("Advanced: Onion Layer Thickness"):
            show_layer_curve = st.checkbox("Show Layer Thickness Curve Helper", value=False)
            start_y = st.slider("Start y-value", 0.1, 1.5, st.session_state.onion_2d.start_y, 0.05)
            end_y = st.slider("End y-value", 0.5, 1.5, st.session_state.onion_2d.end_y, 0.05)
            p1_x = st.slider("P1 x-position", 0.05, 0.5, st.session_state.onion_2d.p1[0], 0.05)
            p1_y = st.slider("P1 y-value", 0.5, 1.5, st.session_state.onion_2d.p1[1], 0.05)
            p2_x = st.slider("P2 x-position", 0.5, 0.9, st.session_state.onion_2d.p2[0], 0.05)
            p2_y = st.slider("P2 y-value", 0.8, 1.2, st.session_state.onion_2d.p2[1], 0.05)
    
    # Update onion model if parameters changed
    onion = HalfOnion(
        onion_diameter,
        n_layers,
        start_y,
        end_y,
        (p1_x, p1_y),
        (p2_x, p2_y)
    )
    
    if onion != st.session_state.onion_2d:
        st.session_state.onion_2d = onion
    
    # Display layer thickness curve if requested
    if show_layer_curve:
        st.subheader("Layer Thickness Curve")
        x = np.linspace(0, 1, 100)
        y = [onion.layer_thickness_curve(xi) for xi in x]
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y)
        # Plot control points
        ax.scatter(
            x=[0, onion.p1[0], onion.p2[0], 1], 
            y=[onion.start_y, onion.p1[1], onion.p2[1], onion.end_y], 
            color='red', 
            marker='o'
        )
        
        # Add labels for control points
        ax.text(0, onion.start_y - 0.05, 'start_y', color='red', ha='center', va='top')
        ax.text(onion.p1[0], onion.p1[1] - 0.05, 'p1', color='red', ha='center', va='top')
        ax.text(onion.p2[0], onion.p2[1] - 0.05, 'p2', color='red', ha='center', va='top')
        ax.text(1, onion.end_y - 0.05, 'end_y', color='red', ha='center', va='top')
        
        ax.set_xlabel('Normalized radius')
        ax.set_ylabel('Relative layer thickness')
        ax.set_title('Layer Thickness Curve')
        ax.grid(True)
        ax.set_ylim(0, max(y) + 0.1)
        ax.set_xlim(0, 1)
        
        st.pyplot(fig)
    
    # Update cutting method and generate cuts
    cutting_method = select_cutting_method()
    
    if cutting_method != st.session_state.cutting_method or not st.session_state.cuts:
        st.session_state.cutting_method = cutting_method
        st.session_state.cuts = generate_cuts_menu(onion, cutting_method)
    
    # Display interactive visualization
    st.header("Interactive Onion Cuts and Piece Size Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Onion with Cuts (Interactive)")
        fig_cuts = visualize_onion_and_cuts(onion, st.session_state.cuts)
        st.plotly_chart(fig_cuts, use_container_width=True)
    
    # Apply cuts and calculate results
    try:
        polygons = GeometryService.apply_cuts_2d(onion, st.session_state.cuts)
        areas, shapes = GeometryService.calculate_areas_and_shapes_2d(polygons)
        
        with col2:
            display_piece_area_distribution(areas)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Resulting Pieces")
            fig_pieces = visualize_pieces(polygons, onion)
            st.plotly_chart(fig_pieces, use_container_width=True)
        
        with col4:
            display_piece_statistics(areas, onion)
        
        st.header("Piece Cross-Sections (Largest to Smallest)")
        piece_shape_plot = visualize_piece_shapes(polygons, cutting_method)
        st.plotly_chart(piece_shape_plot, use_container_width=True)
        
        # Update URL with current settings
        PersistenceService.update_url(onion, st.session_state.cuts, None, cutting_method)
    
    except Exception as e:
        logger.error(f"Error processing cuts: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred while processing the cuts: {str(e)}")
        st.write("Please check the console for more detailed error information.")


if __name__ == "__main__":
    main() 