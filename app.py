"""
Main application entry point for the Onion Simulator.
"""

import streamlit as st
import logging

from services.persistence_service import PersistenceService


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Onion Simulator",
        page_icon="🧅",
        layout="wide"
    )
    
    # Initialize session state
    PersistenceService.initialize_session_state()
    
    # Main page content
    st.title("🧅 Onion Simulator")
    st.write("Welcome to the Onion Simulator! Select a page from the sidebar to get started.")
    
    st.markdown("""
    ## Available Features:
    1. **3D Visualization**
       - View onion models in 3D with interactive controls
       - Multiple view angles and coordinate systems
       - Layer visualization with transparency
    
    2. **Profile Management**
       - Load SVG profiles from the assets directory
       - Visualize layer profiles in 2D
       - Adjust onion size and center offset
    
    ## Getting Started
    1. Add your SVG profiles to the `assets` directory
    2. Navigate to the 3D Simulator using the sidebar
    3. Select a profile and adjust parameters
    4. Interact with the 3D visualization
    
    ## Controls
    - **3D View**:
      - Rotate: Click and drag
      - Zoom: Scroll or pinch
      - Pan: Right-click and drag
      - Reset: Double-click
    """)


if __name__ == "__main__":
    main() 