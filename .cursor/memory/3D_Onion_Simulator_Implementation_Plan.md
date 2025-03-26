# Incremental Implementation Plan for 3D Onion Simulator

## Phase 1: SVG Profile Loading & Basic Visualization ✅

1. **Create SVG Profile Structure** ✅
   - Define SVG format specification for onion profiles ✅
   - Create sample SVG files with Bezier curves for different onion varieties ✅
   - Implement a simple viewer to validate SVG content ✅

2. **Build SVG Parser** ✅
   - Implement SVG path parsing to extract Bezier curves ✅
   - Convert SVG coordinates to our coordinate system ✅
   - Create unit tests to verify parser accuracy ✅
   - Implement layer normalization algorithm ✅
   - Add visualization tools for debugging normalization ✅

3. **Basic Profile Visualization** ✅
   - Display extracted Bezier curves in 2D ✅
   - Implement profile scaling based on max_diameter ✅
   - Add simple UI for loading and selecting different profiles ✅
   - Visualize layer normalization with bounding boxes and alignment points ✅

4. **Profile Revolving Implementation** ✅
   - Implement algorithm to revolve 2D profiles around Z-axis ✅
   - Generate 3D mesh of the onion surface from revolved profiles ✅
   - Visualize the 3D onion model with basic lighting ✅
   - Add layer-specific coloring for better visual distinction ✅

## Phase 2: 3D Model Enhancement 🔄

5. **Layer Structure Implementation** ✅
   - Extend the parser to handle multiple layers from SVG ✅
   - Create internal layer boundaries from multiple profiles ✅
   - Implement proper scaling across all layers ✅
   - Implement layer normalization with corner detection ✅
   - TODO: Add secondary vertical alignment for layers

6. **3D Rendering Improvements** 🔄
   - Implement proper materials and transparency to see layers ✅
   - Add cross-section view capability ✅
   - Implement coordinate system visualization (axes indicators) ✅
   - Add layer-specific coloring with qualitative color palette ✅
   - TODO: Add layer visibility toggles
   - TODO: Add layer opacity controls
   - TODO: Add layer legend

7. **Test Multiple Onion Varieties** 🔄
   - Create a set of real-life onion profiles (3-5 varieties) ✅
   - Implement profile selection UI with previews ✅
   - Test rendering across different profile selections 🔄

## Phase 3: Cutting Implementation ⏱️

8. **Port 2D Cutting Logic**
   - Adapt existing 2D cutting algorithms to work with the new model
   - Implement the interface for editing cuts
   - add root_end_offset slider
   - Ensure visual feedback during cut editing

9. **Implement 3D Cross-Cuts**
   - Create the CrossCut class implementation
   - Develop geometric algorithms for 3D plane cutting
   - Visualize cross-cuts in the 3D view

10. **Cutting Method Integration**
    - Extend the three cutting methods (Classic, Kenji, Josh)
    - Implement editable parameters for each method
    - Create UI controls for modifying cutting parameters

## Phase 4: Analysis and Completion ⏱️

11. **Piece Geometry Implementation**
    - Implement algorithms to generate pieces from cuts
    - Calculate volumes and surface areas
    - Create unit tests for geometric calculations

12. **Analysis Visualization**
    - Implement statistical visualizations for piece metrics
    - Create comparison views between different cutting methods
    - Add interactive elements to explore piece properties

13. **UI Integration and Polish**
    - Integrate all components into cohesive UI
    - Implement synchronized views (cross-section, top view)
    - Add intuitive navigation controls for 3D exploration

14. **Complete Testing**
    - Test with various onion profiles and cutting strategies
    - Verify performance with complex cuts
    - Ensure UI responsiveness with large models

## Recent Progress Summary

1. Successfully implemented SVG path parsing and visualization.
2. Completed the normalization algorithm for aligning SVG profile layers:
   - Algorithm now identifies the most consistent corner across all layers
   - Aligns all paths using bounding box corners
   - Visualizes the normalization process with debug information
   - Includes TODO for future vertical alignment enhancement

3. Enhanced 3D visualization with layer-specific coloring:
   - Added vertex-level layer tracking in mesh generation
   - Implemented distinct colors for each layer using Plotly's Set3 palette
   - Improved visual distinction between layers in 3D view
   - Maintained consistency with 2D layer profile visualization

4. Tested the visualization with sample SVG profiles and confirmed proper rendering.

## Next Steps

1. Add layer visualization enhancements:
   - Implement layer visibility toggles
   - Add layer opacity controls
   - Create a layer legend
   - Consider adding layer labels

2. Begin work on cutting implementation after 3D model is complete

## Testing Approach for Each Step

For each implementation step, we can create specific tests:

1. **Unit Tests**: For parsers, geometric calculations, and algorithms
2. **Visual Tests**: For rendering and visualization components
3. **Integration Tests**: For component interaction
4. **User Scenario Tests**: For complete workflows

## Notes on Implementation Strategy

- Focus on modularity to allow for independent testing of components
- Implement and test one feature at a time
- Use version control to track progress and allow for feature branches
- Maintain compatibility with the existing 2D simulator throughout development
- Consider performance implications when working with complex 3D meshes
- Document design decisions and implementation details as we progress 