"""
3D Onion Simulator page.
"""

import streamlit as st
import os
from models.svg_profile_onion import SvgProfileOnion
from services.visualization_service import VisualizationService
from services.geometry_service import GeometryService
from models.cut import Cut, CrossCut
from cutting_methods.classic import ClassicCuttingMethod
from cutting_methods.kenji import KenjiCuttingMethod
from cutting_methods.josh import JoshCuttingMethod

from wfork_streamlit_profiler import Profiler
 
def select_cutting_method():
    """Display UI for selecting a cutting method."""
    st.sidebar.header("Cutting Method")
    cutting_method = st.sidebar.selectbox(
        "Select Cutting Method", 
        ["Josh's Method", "Classic", "Kenji"],
        index=0
    )
    return cutting_method

def generate_cuts_menu(onion, cutting_method):
    """Generate cuts based on the selected method and parameters."""
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
    
    return cuts

def generate_cross_cuts_menu(onion, cutting_method):
    """Generate cross-cuts based on parameters."""
    st.sidebar.header("Cross-Cuts")
    
    # Common control for all cutting methods
    n_cross_cuts = st.sidebar.slider("Number of Cross-Cuts", 3, 12, 6, 1)
    
    # Get the appropriate cutting method instance
    method = None
    if cutting_method == "Josh's Method":
        method = JoshCuttingMethod(onion)
    elif cutting_method == "Classic":
        method = ClassicCuttingMethod(onion)
    elif cutting_method == "Kenji":
        method = KenjiCuttingMethod(onion)
    
    # Generate cross-cuts
    if method:
        cross_cuts = method.generate_cross_cuts(n_cross_cuts=n_cross_cuts)
        return cross_cuts
    
    return []

def main():
    st.title("3D Onion Simulator")
    st.write("Visualize onion profiles in 3D with interactive controls")

    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Profile selection
    profile_files = [f.replace('.svg', '') for f in os.listdir('assets') if f.endswith('.svg')]
    if not profile_files:
        st.error("No SVG profiles found in assets directory. Please add some profiles first.")
        return
        
    selected_profile = st.sidebar.selectbox(
        "Select Onion Profile",
        profile_files,
        index=0
    )
    
    # Size control
    max_diameter = st.sidebar.slider(
        "Maximum Diameter (units)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )
    
    # Center axis offset
    center_offset = st.sidebar.slider(
        "Center Axis Offset",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.1
    )
    
    try:
        # Create the onion model
        onion = SvgProfileOnion(
            max_diameter=max_diameter,
            profile_name=selected_profile,
            center_axis_offset=center_offset
        )
        
        # Initialize session state for cuts if needed
        if 'cuts' not in st.session_state:
            st.session_state.cuts = []
        if 'cross_cuts' not in st.session_state:
            st.session_state.cross_cuts = []
        if 'cutting_method' not in st.session_state:
            st.session_state.cutting_method = "Josh's Method"
        
        # Update cutting method and generate cuts
        cutting_method = select_cutting_method()
        if cutting_method != st.session_state.cutting_method:
            st.session_state.cutting_method = cutting_method
        st.session_state.cuts = generate_cuts_menu(onion, cutting_method)
        
        # Generate cross-cuts
        st.session_state.cross_cuts = generate_cross_cuts_menu(onion, cutting_method)
        
        # Get pieces from geometry service
        pieces = GeometryService.apply_cuts_to_onion(onion, st.session_state.cuts, st.session_state.cross_cuts)

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["3D View", "Pieces", "Cross-Section", "Top View"])
        
        # 3D visualization
        with tab1:           
            fig = VisualizationService.create_3d_visualization(
                onion,
                cuts=st.session_state.cuts,
                cross_cuts=st.session_state.cross_cuts
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **View Controls:**
            - 🔄 Rotate: Click and drag
            - 🔍 Zoom: Scroll or pinch
            - ✋ Pan: Right-click and drag
            - ↺ Reset: Double-click
            """)
        
        with tab2:
            st.header("3D Pieces")
            if pieces:
                fig_pieces = VisualizationService.visualize_pieces(pieces)
                st.plotly_chart(fig_pieces, use_container_width=True)
            else:
                st.write("No pieces generated.")

        # Cross-section view
        with tab3:
            st.header("Cross-Section View")
            fig_cross = VisualizationService.visualize_cross_section(
                onion,
                cuts=st.session_state.cuts
            )
            st.plotly_chart(fig_cross, use_container_width=True)
        
        # Top view
        with tab4:
            st.header("Top View")
            fig_top = VisualizationService.visualize_top_view(
                onion,
                cuts=st.session_state.cuts
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        # Layer Profiles
        st.header("Layer Profiles")
        profile_fig = VisualizationService.visualize_all_profiles(onion)
        st.plotly_chart(profile_fig, use_container_width=True)
        
        # Profile Information
        st.markdown("""
        ### Profile Information
        
        | Metric | Value |
        |--------|-------|
        | Number of layers | {} |
        | Scale factor | {:.2f} |
        | Total height | {:.2f} units |
        | Maximum width | {:.2f} units |
        """.format(
            len(onion.svg_profiles),
            onion.scale_factor,
            onion.max_height,
            onion.max_width
        ))
            
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error("Please check that the selected profile exists and is valid.")

if __name__ == "__main__":
    with Profiler():
        main() 