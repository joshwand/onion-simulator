import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union, polygonize
from shapely.affinity import affine_transform, scale

class Cut:
    def __init__(self, start, end):
        self.start = (float(start[0]), float(start[1]))  # (x, y) tuple
        self.end = (float(end[0]), float(end[1]))        # (x, y) tuple

    def as_shapely(self):
        return LineString([self.start, self.end])

class HalfOnion:
    def __init__(self, diameter, n_layers):
        self.radius = diameter / 2
        self.n_layers = n_layers
        self.layer_radii = np.linspace(0, self.radius, n_layers + 1)

    def create_layer_boundaries(self):
        return [Point(0, 0).buffer(r).boundary.intersection(Point(0, 0).buffer(self.radius).intersection(Polygon([(-self.radius, 0), (self.radius, 0), (self.radius, self.radius), (-self.radius, self.radius)]))) for r in self.layer_radii[1:]]

def apply_cuts(onion, cuts):
    lines = [cut.as_shapely() for cut in cuts]
    lines.extend(onion.create_layer_boundaries())
    half_circle = Point(0, 0).buffer(onion.radius * 1.001).intersection(Polygon([(-onion.radius, 0), (onion.radius, 0), (onion.radius, onion.radius), (-onion.radius, onion.radius)]))
    intersections = unary_union(lines).intersection(half_circle)
    polygons = list(polygonize(intersections))
    return [polygon for polygon in polygons if polygon.is_valid and polygon.area > 0]

def calculate_areas_and_shapes(polygons):
    areas = []
    shapes = []
    for polygon in polygons:
        if polygon.is_valid and polygon.area > 0:
            area = polygon.area
            bounds = polygon.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            aspect_ratio = min(width, height) / max(width, height)
            areas.append(area)
            shapes.append((width, height, aspect_ratio))
    return areas, shapes

def visualize_piece_shapes(polygons, title):
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
        scaled_polygon = affine_transform(polygon, [1, 0, 0, 1, -center_x, -center_y])
        scale_factor = 0.98 / max_dimension  # Increased from 0.95 to 0.98 to reduce space between pieces
        scaled_polygon = scale(scaled_polygon, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))

        x, y = scaled_polygon.exterior.xy
        color = plt.cm.tab10(i % 10)
        fig.add_trace(go.Scatter(
            x=list(x), y=list(y),
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
        title_text=f"{title} - Piece Shapes (Largest to Smallest)"
    )

    return fig

def classic_cuts(onion, n_vertical, n_horizontal):
    cuts = []
    
    # Vertical cuts
    for i in range(1, n_vertical):
        x = -onion.radius + i * onion.radius * 2 / n_vertical
        cuts.append(Cut((x, 0), (x, onion.radius)))
    
    # Horizontal cuts
    for i in range(1, n_horizontal):
        y = i * onion.radius / n_horizontal
        cuts.append(Cut((-onion.radius, y), (onion.radius, y)))
    
    # Add static bottom cut
    cuts.append(Cut((-onion.radius, 0), (onion.radius, 0)))
    
    return cuts

def kenji_cuts(onion, n_cuts, pct_below):
    cuts = []
    target_point = (0, -pct_below * onion.radius)  # pct_below% of the radius below the center
    
    for i in range(n_cuts):
        angle = np.pi * i / (n_cuts - 1)
        start_x = onion.radius * np.cos(angle)
        start_y = onion.radius * np.sin(angle)
        
        # Calculate the intersection with the circle
        dx = start_x - target_point[0]
        dy = start_y - target_point[1]
        a = dx**2 + dy**2
        b = 2 * (dx * target_point[0] + dy * target_point[1])
        c = target_point[0]**2 + target_point[1]**2 - onion.radius**2
        t = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
        
        end_x = target_point[0] + t * dx
        end_y = target_point[1] + t * dy
        
        cuts.append(Cut((start_x, start_y), (end_x, end_y)))
    
    # Add static bottom cut
    cuts.append(Cut((-onion.radius, 0), (onion.radius, 0)))
    
    return cuts

def custom_cuts(onion, n_horizontal, vertical_height, horizontal_depth, n_vertical):
    cuts = []
    layer_width = onion.radius / onion.n_layers

    # Calculate vertical cut positions
    vertical_positions = [-onion.radius + i * onion.radius * 2 / (n_vertical - 1) for i in range(n_vertical)]

    # Find the vertical cuts closest to the desired horizontal depth
    left_cut_index = np.argmin(np.abs(np.array(vertical_positions) + onion.radius * horizontal_depth))
    right_cut_index = np.argmin(np.abs(np.array(vertical_positions) - onion.radius * horizontal_depth))

    # Horizontal cuts
    horizontal_cuts = []
    for i in range(n_horizontal):
        y = onion.radius * (i + 1) / (n_horizontal + 1) * vertical_height
        start_x_left = -np.sqrt(onion.radius**2 - y**2)
        end_x_left = vertical_positions[left_cut_index]
        start_x_right = np.sqrt(onion.radius**2 - y**2)
        end_x_right = vertical_positions[right_cut_index]
        horizontal_cuts.append((start_x_left, end_x_left, y))
        horizontal_cuts.append((end_x_right, start_x_right, y))

    # Add horizontal cuts to the cuts list
    for start_x, end_x, y in horizontal_cuts:
        cuts.append(Cut((start_x, y), (end_x, y)))

    # Function to check if a vertical cut is too close to the edge
    def is_cut_too_narrow(x, y):
        edge_x = np.sqrt(onion.radius**2 - y**2)
        return abs(abs(x) - edge_x) < layer_width

    # Vertical cuts with width checking
    topmost_y = max(cut[2] for cut in horizontal_cuts)
    for i, x in enumerate(vertical_positions):
        y_start = np.sqrt(onion.radius**2 - x**2)  # Start from edge
        if i == left_cut_index or i == right_cut_index:
            # Leftmost and rightmost intersecting cuts always extend to the bottom
            y_end = 0
        else:
            y_end = 0  # Default to extending to the bottom
            # Check for intersections with horizontal cuts
            for start_x, end_x, y in horizontal_cuts:
                if start_x <= x <= end_x:
                    y_end = max(y_end, y)  # Stop at the highest intersecting horizontal cut
        
        # Check if this is an outermost cut and if it's too narrow at the topmost intersection
        if (i == 0 or i == len(vertical_positions) - 1) and is_cut_too_narrow(x, topmost_y):
            continue  # Skip this cut if it's too narrow
        
        # Add the vertical cut
        cuts.append(Cut((x, y_start), (x, y_end)))

    # Add static bottom cut
    cuts.append(Cut((-onion.radius, 0), (onion.radius, 0)))

    return cuts

def visualize_onion_and_cuts(onion, cuts):
    fig = go.Figure()

    # Draw onion outline
    theta = np.linspace(0, np.pi, 100)
    x = onion.radius * np.cos(theta)
    y = onion.radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), mode='lines', name='Onion outline'))

    # Draw layers
    for r in onion.layer_radii[1:]:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        fig.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), mode='lines', line=dict(dash='dot'), name=f'Layer (r={r:.2f})'))

    # Draw cuts
    for i, cut in enumerate(cuts):
        fig.add_trace(go.Scatter(x=[cut.start[0], cut.end[0]], 
                                 y=[cut.start[1], cut.end[1]], 
                                 mode='lines+markers',
                                 name=f'Cut {i+1}',
                                 line=dict(color='red'),
                                 marker=dict(size=8, symbol='square')))

    fig.update_layout(
        title='Onion with Cuts',
        xaxis_title='X',
        yaxis_title='Y',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        height=600,
        width=600
    )

    return fig


def visualize_pieces(polygons, title):
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
            # name=f'Piece {i+1}'
        ))

    fig.update_layout(
        height=500,
        width=500,
        showlegend=False,
        title_text=f"{title} - Resulting Pieces"
    )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False)

    return fig

def visualize_pieces(polygons, title):
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
            # name=f'Piece {i+1}'
        ))

    fig.update_layout(
        height=500,
        width=500,
        showlegend=False,
        title_text=f"{title} - Resulting Pieces"
    )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False)

    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Onion Cutting Simulator")

    # Sidebar for inputs
    st.sidebar.header("Onion Parameters")
    onion_diameter = st.sidebar.slider("Onion Diameter (inches)", 1.0, 10.0, 5.0)
    n_layers = st.sidebar.slider("Number of Layers", 3, 20, 11)

    # Create the onion object
    onion = HalfOnion(onion_diameter, n_layers)

    # Add method selector
    st.sidebar.header("Cutting Method")
    cutting_method = st.sidebar.selectbox("Select Cutting Method", ["Josh's Method", "Classic", "Kenji"])

    # Add method-specific parameters
    st.sidebar.header("Cut Parameters")
    if cutting_method == "Josh's Method":
        n_horizontal = st.sidebar.slider("Number of Horizontal Cuts", 2, 10, 3)
        vertical_height = st.sidebar.slider("Vertical Cut Height (fraction of radius)", 0.1, 1.0, 0.8)
        horizontal_depth = st.sidebar.slider("Horizontal Cut Depth (fraction of radius)", 0.1, 1.0, 0.3)
        n_vertical = st.sidebar.slider("Number of Vertical Cuts", 3, 20, 10)
        cuts = custom_cuts(onion, n_horizontal, vertical_height, horizontal_depth, n_vertical)
    elif cutting_method == "Classic":
        n_vertical = st.sidebar.slider("Number of Vertical Cuts", 4, 16, 10)
        n_horizontal = st.sidebar.slider("Number of Horizontal Cuts", 1, 10, 2)
        cuts = classic_cuts(onion, n_vertical, n_horizontal)
    else:  # Kenji method
        n_cuts = st.sidebar.slider("Number of Cuts", 3, 20, 10)
        pct_below = st.sidebar.slider("Target Point (fraction of radius below center)", 0.1, 0.9, 0.5)
        cuts = kenji_cuts(onion, n_cuts, pct_below)

    # Apply cuts and calculate polygons
    polygons = apply_cuts(onion, cuts)

    # First row: Onion with Cuts and Piece Area Distribution
    st.header("Onion Cuts and Piece Size Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Onion with Cuts")
        fig_cuts = visualize_onion_and_cuts(onion, cuts)
        st.plotly_chart(fig_cuts, use_container_width=True)

    with col2:
        st.subheader("Piece Area Distribution")
        areas, shapes = calculate_areas_and_shapes(polygons)
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

    # Second row: Resulting Pieces and Piece Aspect Ratio Distribution
    st.header("Resulting Pieces and Aspect Ratio Distribution")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Resulting Pieces")
        fig_pieces = visualize_pieces(polygons, cutting_method)
        st.plotly_chart(fig_pieces, use_container_width=True)

    with col4:
        st.subheader("Piece Aspect Ratio Distribution")
        fig_square, ax_square = plt.subplots(figsize=(8, 5))
        ax_square.hist([s[2] for s in shapes], bins=20)
        ax_square.set_title("Distribution of Piece Aspect Ratio")
        ax_square.set_xlabel('Aspect Ratio (1 is perfect square)')
        ax_square.set_ylabel('Frequency')
        st.pyplot(fig_square)

    # Display the piece statistics
    st.header("Piece Statistics")
    st.write(f"Number of pieces: {len(areas)}")
    st.write(f"Average piece area: {np.mean(areas):.4f} sq inches")
    st.write(f"Std dev of areas: {np.std(areas):.4f} sq inches")
    st.write(f"Average aspect ratio (1 is perfect): {np.mean([s[2] for s in shapes]):.4f}")

    # Add the piece shape visualization
    st.header("Piece Cross-Sections (Largest to Smallest)")
    try:
        piece_shape_plot = visualize_piece_shapes(polygons, cutting_method)
        st.plotly_chart(piece_shape_plot, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred while visualizing piece shapes: {str(e)}")
        st.write("Please try adjusting the cut parameters.")

if __name__ == "__main__":
    main()