"""
SVG parsing utilities for the 3D Onion Simulator.
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from matplotlib.path import Path
import matplotlib.patches as patches


logger = logging.getLogger(__name__)


def parse_svg_path_commands(d: str) -> List[Tuple[str, List[float]]]:
    """
    Parse SVG path data string to extract commands and coordinates.
    
    Args:
        d: SVG path data string (e.g., "M100,100 C30,30 40,40 50,50 L200,200...")
        
    Returns:
        List of tuples with (command, [coordinates])
    """
    # Match commands (letters) followed by numbers
    command_pattern = r'([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)'
    commands = re.findall(command_pattern, d)
    
    parsed_commands = []
    for cmd, coords_str in commands:
        # Extract numbers from coordinate string
        coords = re.findall(r'-?\d+\.?\d*', coords_str)
        # Convert to float
        coords = [float(c) for c in coords]
        parsed_commands.append((cmd, coords))
    
    return parsed_commands


def commands_to_matplotlib_path(commands: List[Tuple[str, List[float]]]) -> Path:
    """
    Convert SVG path commands to a matplotlib Path object.
    
    Args:
        commands: List of (command, [coordinates]) tuples
        
    Returns:
        Matplotlib Path object
    """
    vertices = []
    codes = []
    
    current_point = (0, 0)
    
    for cmd, coords in commands:
        if cmd == 'M':  # MoveTo absolute
            vertices.append((coords[0], coords[1]))
            codes.append(Path.MOVETO)
            current_point = (coords[0], coords[1])
        
        elif cmd == 'm':  # MoveTo relative
            x, y = current_point
            vertices.append((x + coords[0], y + coords[1]))
            codes.append(Path.MOVETO)
            current_point = (x + coords[0], y + coords[1])
        
        elif cmd == 'L':  # LineTo absolute
            for i in range(0, len(coords), 2):
                vertices.append((coords[i], coords[i+1]))
                codes.append(Path.LINETO)
                current_point = (coords[i], coords[i+1])
        
        elif cmd == 'l':  # LineTo relative
            x, y = current_point
            for i in range(0, len(coords), 2):
                vertices.append((x + coords[i], y + coords[i+1]))
                codes.append(Path.LINETO)
                current_point = (x + coords[i], y + coords[i+1])
        
        elif cmd == 'H':  # Horizontal LineTo absolute
            for i in range(len(coords)):
                x = coords[i]
                y = current_point[1]
                vertices.append((x, y))
                codes.append(Path.LINETO)
                current_point = (x, y)
        
        elif cmd == 'h':  # Horizontal LineTo relative
            for i in range(len(coords)):
                x = current_point[0] + coords[i]
                y = current_point[1]
                vertices.append((x, y))
                codes.append(Path.LINETO)
                current_point = (x, y)
        
        elif cmd == 'V':  # Vertical LineTo absolute
            for i in range(len(coords)):
                x = current_point[0]
                y = coords[i]
                vertices.append((x, y))
                codes.append(Path.LINETO)
                current_point = (x, y)
        
        elif cmd == 'v':  # Vertical LineTo relative
            for i in range(len(coords)):
                x = current_point[0]
                y = current_point[1] + coords[i]
                vertices.append((x, y))
                codes.append(Path.LINETO)
                current_point = (x, y)
        
        elif cmd == 'C':  # Cubic Bezier absolute
            for i in range(0, len(coords), 6):
                x1, y1 = coords[i], coords[i+1]
                x2, y2 = coords[i+2], coords[i+3]
                x3, y3 = coords[i+4], coords[i+5]
                
                vertices.extend([(x1, y1), (x2, y2), (x3, y3)])
                codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
                current_point = (x3, y3)
        
        elif cmd == 'c':  # Cubic Bezier relative
            x, y = current_point
            for i in range(0, len(coords), 6):
                x1, y1 = x + coords[i], y + coords[i+1]
                x2, y2 = x + coords[i+2], y + coords[i+3]
                x3, y3 = x + coords[i+4], y + coords[i+5]
                
                vertices.extend([(x1, y1), (x2, y2), (x3, y3)])
                codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
                current_point = (x3, y3)
        
        elif cmd == 'Q':  # Quadratic Bezier absolute
            for i in range(0, len(coords), 4):
                x1, y1 = coords[i], coords[i+1]
                x2, y2 = coords[i+2], coords[i+3]
                
                vertices.extend([(x1, y1), (x2, y2)])
                codes.extend([Path.CURVE3, Path.CURVE3])
                current_point = (x2, y2)
        
        elif cmd == 'q':  # Quadratic Bezier relative
            x, y = current_point
            for i in range(0, len(coords), 4):
                x1, y1 = x + coords[i], y + coords[i+1]
                x2, y2 = x + coords[i+2], y + coords[i+3]
                
                vertices.extend([(x1, y1), (x2, y2)])
                codes.extend([Path.CURVE3, Path.CURVE3])
                current_point = (x2, y2)
        
        elif cmd in ('Z', 'z'):  # Close path
            vertices.append(vertices[0])  # Close the path
            codes.append(Path.CLOSEPOLY)
    
    return Path(vertices, codes)


def extract_paths_from_svg(file_path: str) -> List[Path]:
    """
    Extract path objects from an SVG file.
    
    Args:
        file_path: Path to the SVG file
        
    Returns:
        List of matplotlib Path objects
    """
    if not os.path.exists(file_path):
        logger.error(f"SVG file not found: {file_path}")
        return []
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Find all path elements in the SVG
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        
        matplotlib_paths = []
        for path in paths:
            d = path.get('d')
            if d:
                commands = parse_svg_path_commands(d)
                path_obj = commands_to_matplotlib_path(commands)
                matplotlib_paths.append(path_obj)
        
        return matplotlib_paths
    except Exception as e:
        logger.error(f"Error parsing SVG file: {e}")
        logger.error(f"Details: {str(e)}")
        return []


def sample_path_points(path: Path, n_points: int = 100) -> np.ndarray:
    """
    Sample points along a path at regular intervals.
    
    Args:
        path: Matplotlib Path object
        n_points: Number of points to sample
        
    Returns:
        Array of (x, y) coordinates
    """
    # Generate parameter values from 0 to 1
    t_points = np.linspace(0, 1, n_points)
    
    # Interpolate points along the path
    points = path.interpolated(n_points).vertices
    
    return points


def normalize_paths(paths: List[Path]) -> List[Path]:
    """
    Normalize paths by finding the most consistent corner across all paths and aligning them.
    
    The algorithm:
    1. Draw a rectilinear bounding box around each path
    2. For each path and each corner direction (NE, NW, SW, SE), find the closest point
    3. Identify which corner has the most consistent cluster of closest points
    4. Use that corner to align all paths
    
    Args:
        paths: List of matplotlib Path objects
        
    Returns:
        List of normalized matplotlib Path objects
        
    TODO: Add secondary vertical alignment to ensure both X and Y axes are properly aligned.
    This would involve finding another reference point (such as the bottom edge midpoint) 
    and aligning these points vertically after the corner alignment is complete.
    """
    if not paths:
        return []
    
    # print("\n----- Normalization Debug Info -----")
    
    # Define corners: Northeast, Northwest, Southwest, Southeast
    CORNERS = {
        'NE': (1, 1),   # Top right
        'NW': (-1, 1),  # Top left
        'SW': (-1, -1), # Bottom left
        'SE': (1, -1)   # Bottom right
    }
    
    # Step 1: Find bounding box for each path
    bboxes = []
    for i, path in enumerate(paths):
        vertices = path.vertices
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)
        bbox = {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'corners': {
                'NE': (max_x, max_y),
                'NW': (min_x, max_y),
                'SW': (min_x, min_y),
                'SE': (max_x, min_y)
            }
        }
        bboxes.append(bbox)
        # print(f"Path {i+1} bounding box: ({min_x:.2f}, {min_y:.2f}) to ({max_x:.2f}, {max_y:.2f})")
    
    # Step 2: Find the closest point on each path to each corner direction
    corner_closest_points = {corner: [] for corner in CORNERS.keys()}
    
    for i, (path, bbox) in enumerate(zip(paths, bboxes)):
        vertices = path.vertices
        
        for corner_name, corner_pos in bbox['corners'].items():
            # Calculate distances from all vertices to the corner
            distances = np.sqrt(
                (vertices[:, 0] - corner_pos[0])**2 + 
                (vertices[:, 1] - corner_pos[1])**2
            )
            
            # Find the closest vertex
            closest_idx = np.argmin(distances)
            closest_point = vertices[closest_idx]
            corner_closest_points[corner_name].append({
                'path_idx': i,
                'point': closest_point,
                'distance': distances[closest_idx],
                'bbox_corner': corner_pos  # Store the actual bbox corner
            })
            
            # print(f"Path {i+1} closest to {corner_name}: ({closest_point[0]:.2f}, {closest_point[1]:.2f}), distance: {distances[closest_idx]:.2f}")
    
    # Step 3: Determine which corner has the most consistent cluster (lowest average distance)
    corner_avg_distances = {}
    for corner_name, corner_points in corner_closest_points.items():
        # For each corner, calculate centroid of the closest points
        points = np.array([p['point'] for p in corner_points])
        centroid = np.mean(points, axis=0)
        
        # Calculate average distance from each path's closest point to the centroid
        distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
        avg_distance = np.mean(distances)
        
        corner_avg_distances[corner_name] = {
            'avg_distance': avg_distance,
            'centroid': centroid,
            'points': points,
            'distances': distances
        }
        
        # print(f"{corner_name} corner - avg distance to centroid: {avg_distance:.2f}, centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
    
    # Step 4: Choose the corner with the lowest average distance (most consistent)
    origin_corner = min(corner_avg_distances.items(), key=lambda x: x[1]['avg_distance'])[0]
    origin_data = corner_avg_distances[origin_corner]
    
    # print(f"\nChosen origin corner: {origin_corner} with avg distance: {origin_data['avg_distance']:.2f}")
    # print(f"Origin centroid: ({origin_data['centroid'][0]:.2f}, {origin_data['centroid'][1]:.2f})")
    
    # Step 5-7: Translate each path based on vector from its bbox corner to origin
    normalized_paths = []
    for i, path in enumerate(paths):
        vertices = path.vertices.copy()
        
        # Get the actual bounding box corner for this path (not the closest point)
        bbox_corner = corner_closest_points[origin_corner][i]['bbox_corner']
        
        # Calculate the translation vector
        x_translate = -bbox_corner[0]
        y_translate = -bbox_corner[1]
        
        # print(f"Path {i+1} translation vector: ({x_translate:.2f}, {y_translate:.2f})")
        
        # Apply translation
        vertices[:, 0] += x_translate
        vertices[:, 1] += y_translate
        
        # Create a new path with the translated vertices
        new_path = Path(vertices=vertices, codes=path.codes)
        normalized_paths.append(new_path)
    
    # print("----- End Normalization Debug Info -----\n")
    
    return normalized_paths


def visualize_svg_paths(svg_file: str, normalize: bool = False) -> None:
    """
    Visualize paths from an SVG file.
    
    Args:
        svg_file: Path to the SVG file
        normalize: Whether to normalize the paths
    """
    paths = extract_paths_from_svg(svg_file)
    
    if not paths:
        print(f"No paths found in {svg_file}")
        return
    
    print(f"Found {len(paths)} paths in SVG file.")
    
    # Create a figure with two subplots if normalizing
    if normalize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        plt.subplots_adjust(wspace=0.3)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 10))
    
    # Plot each path with a different color
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
    
    # Calculate bounding boxes for visualization
    bboxes = []
    for path in paths:
        vertices = path.vertices
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)
        bboxes.append((min_x, min_y, max_x, max_y))
    
    # Plot original paths
    for i, path in enumerate(paths):
        color_idx = i % len(colors)
        color = colors[color_idx]
        
        # Plot the path
        patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=2, label=f'Layer {i+1}')
        ax1.add_patch(patch)
        
        # Plot the bounding box
        bbox = bboxes[i]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1], 
            linewidth=1, 
            edgecolor=color, 
            facecolor='none', 
            linestyle='--'
        )
        ax1.add_patch(rect)
        
        # Add corner markers
        corners = [
            (bbox[0], bbox[3]), # NW
            (bbox[2], bbox[3]), # NE
            (bbox[0], bbox[1]), # SW
            (bbox[2], bbox[1])  # SE
        ]
        corner_labels = ['NW', 'NE', 'SW', 'SE']
        
        # Find closest points to corners
        vertices = path.vertices
        for j, (corner, label) in enumerate(zip(corners, corner_labels)):
            # Mark the corner of the bounding box
            ax1.plot(corner[0], corner[1], 'o', color=color, markersize=4)
            
            # Calculate distances from all vertices to this corner
            distances = np.sqrt(
                (vertices[:, 0] - corner[0])**2 + 
                (vertices[:, 1] - corner[1])**2
            )
            
            # Find the closest vertex
            closest_idx = np.argmin(distances)
            closest_point = vertices[closest_idx]
            
            # Plot the closest point and a line to the corner
            ax1.plot(closest_point[0], closest_point[1], '*', color=color, markersize=8)
            ax1.plot([corner[0], closest_point[0]], [corner[1], closest_point[1]], 
                    color=color, linestyle=':', linewidth=1)
    
    ax1.set_aspect('equal')
    ax1.autoscale_view()
    ax1.set_title("Original SVG Paths")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.grid(True)
    
    # If normalizing, show normalized paths in second subplot
    if normalize:
        # First find the corner that will be used for normalization
        corner_info = find_normalization_corner(paths)
        best_corner = corner_info['corner']
        
        # Highlight the best corner in the original plot
        for i, bbox in enumerate(bboxes):
            corner_coords = None
            if best_corner == 'NE':
                corner_coords = (bbox[2], bbox[3])
                ax1.annotate("NE", (bbox[2], bbox[3]), fontsize=12, color='red', fontweight='bold')
            elif best_corner == 'NW':
                corner_coords = (bbox[0], bbox[3])
                ax1.annotate("NW", (bbox[0], bbox[3]), fontsize=12, color='red', fontweight='bold')
            elif best_corner == 'SW':
                corner_coords = (bbox[0], bbox[1])
                ax1.annotate("SW", (bbox[0], bbox[1]), fontsize=12, color='red', fontweight='bold')
            elif best_corner == 'SE':
                corner_coords = (bbox[2], bbox[1])
                ax1.annotate("SE", (bbox[2], bbox[1]), fontsize=12, color='red', fontweight='bold')
            
            # Highlight the chosen corner with a larger marker
            if corner_coords:
                ax1.plot(corner_coords[0], corner_coords[1], 'rx', markersize=12, markeredgewidth=2)
        
        # Now normalize and plot
        normalized_paths = normalize_paths(paths)
        
        for i, path in enumerate(normalized_paths):
            color_idx = i % len(colors)
            color = colors[color_idx]
            
            # Plot the normalized path
            patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=2, label=f'Layer {i+1}')
            ax2.add_patch(patch)
            
            # Calculate the normalized bounding box
            vertices = path.vertices
            min_x, min_y = np.min(vertices, axis=0)
            max_x, max_y = np.max(vertices, axis=0)
            
            # Plot the normalized bounding box
            rect = patches.Rectangle(
                (min_x, min_y), 
                max_x - min_x, 
                max_y - min_y, 
                linewidth=1, 
                edgecolor=color, 
                facecolor='none', 
                linestyle='--'
            )
            ax2.add_patch(rect)
            
            # Highlight the origin-aligned corner
            corner_coords = None
            if best_corner == 'NE':
                corner_coords = (max_x, max_y)
            elif best_corner == 'NW':
                corner_coords = (min_x, max_y)
            elif best_corner == 'SW':
                corner_coords = (min_x, min_y)
            elif best_corner == 'SE':
                corner_coords = (max_x, min_y)
            
            if corner_coords:
                ax2.plot(corner_coords[0], corner_coords[1], 'rx', markersize=12, markeredgewidth=2)
        
        # Add origin marker for reference
        ax2.plot(0, 0, 'ko', markersize=10)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        ax2.set_aspect('equal')
        ax2.autoscale_view()
        ax2.set_title(f"Normalized Paths (Aligned by {best_corner} Corner)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def find_normalization_corner(paths):
    """
    Find the best corner for normalization without actually normalizing.
    For visualization purposes.
    """
    if not paths:
        return {'corner': 'NE', 'avg_distance': 0.0}
    
    # Define corners: Northeast, Northwest, Southwest, Southeast
    CORNERS = {
        'NE': (1, 1),   # Top right
        'NW': (-1, 1),  # Top left
        'SW': (-1, -1), # Bottom left
        'SE': (1, -1)   # Bottom right
    }
    
    # Find bounding box for each path
    bboxes = []
    for path in paths:
        vertices = path.vertices
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)
        bbox = {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'corners': {
                'NE': (max_x, max_y),
                'NW': (min_x, max_y),
                'SW': (min_x, min_y),
                'SE': (max_x, min_y)
            }
        }
        bboxes.append(bbox)
    
    # Find the closest point on each path to each corner direction
    corner_closest_points = {corner: [] for corner in CORNERS.keys()}
    
    for i, (path, bbox) in enumerate(zip(paths, bboxes)):
        vertices = path.vertices
        
        for corner_name, corner_pos in bbox['corners'].items():
            # Calculate distances from all vertices to the corner
            distances = np.sqrt(
                (vertices[:, 0] - corner_pos[0])**2 + 
                (vertices[:, 1] - corner_pos[1])**2
            )
            
            # Find the closest vertex
            closest_idx = np.argmin(distances)
            closest_point = vertices[closest_idx]
            corner_closest_points[corner_name].append({
                'path_idx': i,
                'point': closest_point,
                'distance': distances[closest_idx]
            })
    
    # Determine which corner has the most consistent cluster
    corner_avg_distances = {}
    for corner_name, corner_points in corner_closest_points.items():
        # For each corner, calculate centroid of the closest points
        points = np.array([p['point'] for p in corner_points])
        centroid = np.mean(points, axis=0)
        
        # Calculate average distance from each path's closest point to the centroid
        distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
        avg_distance = np.mean(distances)
        
        corner_avg_distances[corner_name] = avg_distance
    
    # Choose the corner with the lowest average distance
    corner = min(corner_avg_distances.items(), key=lambda x: x[1])[0]
    
    return {
        'corner': corner,
        'avg_distance': corner_avg_distances[corner]
    }


if __name__ == "__main__":
    """Test the SVG parser with a sample file."""
    import sys
    
    if len(sys.argv) > 1:
        svg_file = sys.argv[1]
        normalize = len(sys.argv) > 2 and sys.argv[2].lower() == 'normalize'
    else:
        svg_file = os.path.join('assets', 'onion1.svg')
        normalize = True  # Default to normalizing
    
    if not os.path.exists(svg_file):
        print(f"File not found: {svg_file}")
        sys.exit(1)
    
    visualize_svg_paths(svg_file, normalize=normalize) 