# Onion Simulator: Active Context

## Current Task: 3D Onion Cutting Implementation

### Main Focus
- Calculating the resulting onion pieces bounded by the cuts and the layer surfaces
- Adding cross-cuts visualization to the top view


## Final Step-by-Step Implementation Plan

1. Update OnionPiece3D to store and manage trimesh objects
2. Implement basic volume calculation using trimesh
3. Add face classification for different surface types
4. Create area calculation for each surface type
5. Implement plane-mesh intersection algorithm
6. Develop mesh splitting along a plane
7. Add mesh validation and repair functionality
8. Implement connected component analysis for piece extraction
9. Add face origin labeling during cutting
10. Create sequential cutting algorithm
11. Implement CrossCut processing
12. Add integration with existing cutting methods
13. Develop piece generation service
14. Create visualization enhancements for pieces
15. Implement statistical analysis of pieces
16. Add UI components for piece exploration and analysis

# Prompt Series for TDD Implementation

## Prompt 1: Enhancing OnionPiece3D Class

```
Implement an enhanced version of the OnionPiece3D class that stores a complete 3D mesh representation using trimesh. The class should have the following capabilities:

1. Store a trimesh mesh object representing the 3D geometry
2. Track face classifications (external, layer, cut surfaces)
3. Add required metadata (layer information, id, etc.)
4. Provide basic serialization/deserialization methods

Approach this implementation using test-driven development:
1. First, create a test file with meaningful test cases for the OnionPiece3D class
2. Implement the class to pass these tests
3. Make sure you use trimesh properly for mesh operations

Keep in mind that this class will need to work with the existing code base, particularly with the geometry_service.py file, which already includes methods for applying cuts and analyzing pieces.
```

## Prompt 2: Volume and Surface Area Calculations

```
Implement volume and surface area calculations for the OnionPiece3D class following test-driven development principles. The implementation should:

1. Add a method to calculate the volume of 3D pieces accurately using trimesh
2. Implement algorithms to classify mesh faces into different types:
   - External faces (from the original onion surface)
   - Layer faces (interfaces between onion layers)
   - Cut faces (created by cut planes)
3. Calculate surface areas for each type of face
4. Add properties to access these calculations efficiently

For testing:
1. Create test cases with simple shapes with known volumes and surface areas
2. Test face classification with various mesh configurations
3. Verify calculations match expected results for test cases

Ensure your implementation is robust against edge cases:
- Degenerate meshes (very small or flat)
- Meshes with holes or non-manifold elements
- Complex, non-convex shapes
```

## Prompt 3: Implementing Mesh Slicing

```
Implement a robust mesh slicing algorithm that will form the foundation of the piece generation feature. This implementation should:

1. Create a PlaneSlicing utility class with the following capabilities:
   - Find all intersection points of a plane with the mesh edges
   - Generate a cutting contour along the intersection
   - Split the mesh along the cutting plane
   - Label the newly created faces as "cut" faces
2. Handle edge cases correctly:
   - Planes that pass through vertices or edges
   - Nearly parallel planes
   - Multiple intersection points
3. Ensure resulting meshes are watertight

Test-driven approach:
1. Create tests with simple geometric shapes (cubes, spheres) that have known slicing results
2. Test edge cases thoroughly
3. Verify topological correctness of the resulting meshes
4. Check that face labeling is consistent

The slicing implementation should work with the trimesh library and be optimized for performance with complex meshes.
```

## Prompt 4: Piece Extraction from Sliced Meshes

```
Implement a functionality to extract separate pieces after slicing a mesh with multiple planes. The implementation should:

1. Create a PieceExtractor utility class with the following capabilities:
   - Identify connected components in the mesh after slicing
   - Extract each component as a separate OnionPiece3D object
   - Preserve face classifications and other metadata
   - Filter out degenerate or extremely small pieces
2. Handle the case of multiple sequential cuts correctly
3. Maintain proper face classification through the extraction process

Test-driven approach:
1. Create tests for simple cases (one cut creating two pieces)
2. Test complex scenarios with multiple intersecting cuts
3. Verify that all pieces are correctly extracted
4. Check that face classifications are preserved
5. Test edge cases like cuts that produce very small fragments

The implementation should integrate with the previously developed mesh slicing functionality and prepare the foundation for the full piece generation algorithm.
```

## Prompt 5: Implementing Sequential Cutting

```
Implement a sequential cutting algorithm that can apply multiple cuts (both regular cuts and cross-cuts) to an onion mesh. This should:

1. Create a SequentialCutter class that can:
   - Apply a list of Cut objects to a 3D mesh in sequence
   - Apply CrossCut objects correctly
   - Generate all resulting pieces as OnionPiece3D objects
   - Maintain face classifications through multiple cuts
2. Optimize the cutting order for efficiency
3. Handle complex interactions between multiple cuts

Test-driven approach:
1. Create tests for various cutting patterns (horizontal, vertical, angled cuts)
2. Test combinations of regular cuts and cross-cuts
3. Verify that all pieces are correctly generated
4. Check that face classifications remain consistent
5. Test with complex real-world cutting strategies

The implementation should build on the previously developed components and should integrate smoothly with the existing cutting method classes in the project.
```

## Prompt 6: Integrating with Existing Cutting Methods

```
Integrate the piece generation functionality with the existing cutting methods in the project. The implementation should:

1. Modify the GeometryService class to use the new piece generation algorithm
2. Update the apply_cuts_3d method to properly handle both normal cuts and cross-cuts
3. Ensure compatibility with all cutting methods (Classic, Kenji, Josh)
4. Optimize the implementation for performance with complex cuts

Test-driven approach:
1. Create tests that apply each cutting method and verify piece generation
2. Test with multiple layers and complex cutting patterns
3. Verify that results are consistent with the expected behavior of each method
4. Benchmark performance with varying complexity levels

The implementation should maintain backward compatibility with existing code while adding the new piece geometry functionality.
```

## Prompt 7: Implementing Analysis Features

```
Enhance the AnalysisService class to provide comprehensive analysis of 3D pieces. The implementation should:

1. Update calculate_volume_statistics to use the accurate volume calculations
2. Enhance calculate_surface_area_statistics to analyze different surface types
3. Add new analysis methods specific to 3D pieces:
   - Distribution of piece sizes across layers
   - Surface-to-volume ratio analysis
   - Cut efficiency metrics (how much cut surface is created)
4. Implement visualization data preparation for these metrics

Test-driven approach:
1. Create tests for each analysis method
2. Verify statistical calculations against known results
3. Test with various piece distributions and cutting patterns
4. Ensure compatibility with the UI visualization components

The analysis implementation should provide valuable insights into the effectiveness of different cutting methods and support the main goals of the onion simulator.
```

## Prompt 8: Adding Visualization Components

```
Implement visualization enhancements for the piece geometry feature. The implementation should:

1. Add methods to visualize individual pieces in 3D
2. Create color coding for different surface types (external, layer, cut)
3. Implement interactive piece selection
4. Add visualization of piece metrics and statistics
5. Create comparison views between different cutting methods

Test-driven approach:
1. Create tests for the visualization components
2. Verify correct rendering of pieces with different properties
3. Test interactive behavior
4. Ensure performance with complex scenes

The visualization implementation should integrate with the existing UI structure and enhance the user experience by providing clear visual feedback about the cutting results.
```

## Prompt 9: Final Integration and Performance Optimization

```
Complete the piece geometry feature by integrating all components and optimizing for performance. The implementation should:

1. Create a unified PieceGeometryService that combines all functionality:
   - Piece generation from cuts
   - Geometric calculations
   - Analysis and visualization preparation
2. Implement caching mechanisms for expensive calculations
3. Add parallel processing for performance-critical operations
4. Ensure robust error handling and graceful degradation

Test-driven approach:
1. Create integration tests for the complete workflow
2. Benchmark performance with various complexity levels
3. Test error handling with problematic inputs
4. Verify end-to-end functionality

This final implementation should complete the piece geometry feature and provide a robust foundation for further enhancements to the 3D onion simulator.
```

# Implementation Steps Summary

The implementation plan follows these key principles:

1. **Incremental Development**: Each prompt builds on the previous one, creating a clear progression.
2. **Test-Driven Approach**: Every component starts with test development before implementation.
3. **Integration Focus**: Components are designed to integrate smoothly with existing code.
4. **Performance Consideration**: Performance optimization is built into the design from the beginning.
5. **Robustness**: Edge cases and error handling are emphasized throughout.

This plan provides a comprehensive roadmap for implementing the piece geometry feature, with well-defined steps that build on each other and integrate into the existing codebase.

---
Implementation Steps Summary
The implementation plan follows these key principles:
Incremental Development: Each prompt builds on the previous one, creating a clear progression.
Test-Driven Approach: Every component starts with test development before implementation.
Integration Focus: Components are designed to integrate smoothly with existing code.
Performance Consideration: Performance optimization is built into the design from the beginning.
Robustness: Edge cases and error handling are emphasized throughout.
This plan provides a comprehensive roadmap for implementing the piece geometry feature, with well-defined steps that build on each other and integrate into the existing codebase.