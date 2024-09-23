import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union, polygonize
from shapely.affinity import affine_transform, scale
import logging
import traceback
import json
import urllib.parse


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Cut:
    def __init__(self, start, end):
        self.start = (float(start[0]), float(start[1]))  # (x, y) tuple
        self.end = (float(end[0]), float(end[1]))        # (x, y) tuple

    def as_shapely(self):
        return LineString([self.start, self.end])
    
    def __repr__(self) -> str:
        return f"Cut({self.start}, {self.end})"

class HalfOnion:
    def __init__(self, diameter, n_layers):
        self.radius = diameter / 2
        self.n_layers = n_layers
        self.layer_radii = np.linspace(0, self.radius, n_layers + 1)

    def create_layer_boundaries(self):
        return [Point(0, 0).buffer(r).boundary.intersection(Point(0, 0).buffer(self.radius).intersection(Polygon([(-self.radius, 0), (self.radius, 0), (self.radius, self.radius), (-self.radius, self.radius)]))) for r in self.layer_radii[1:]]

def apply_cuts(onion, cuts):

    # remove any cuts that have length < 0.001
    cuts = [cut for cut in cuts if (cut.end[0] - cut.start[0])**2 + (cut.end[1] - cut.start[1])**2 > 0.001]

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
    
    # Horizontal cuts (excluding the bottom static cut)
    for i in range(1, n_horizontal + 1):
        y = i * onion.radius / (n_horizontal + 1)
        cuts.append(Cut((-onion.radius, y), (onion.radius, y)))
    
    # Add static bottom cut (always included)
    cuts.append(Cut((-onion.radius, 0), (onion.radius, 0)))
    
    return cuts

def kenji_cuts(onion, n_cuts, pct_below):
    cuts = []
    target_point = (0, -pct_below * onion.radius)  # pct_below% of the radius below the center
    
    n_cuts += 2
    for i in range(n_cuts):
        angle = np.pi * i / (n_cuts - 1)
        start_x = onion.radius * np.cos(angle)
        start_y = onion.radius * np.sin(angle)
        
        # Skip cuts at (-r, 0) and (r, 0)
        if abs(start_x) == onion.radius and start_y == 0:
            continue
        
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
    left_cut_index = np.argmin(np.abs(np.array(vertical_positions) + onion.radius * (1 -horizontal_depth)))
    right_cut_index = np.argmin(np.abs(np.array(vertical_positions) - onion.radius * (1 -horizontal_depth)))

    # Horizontal cuts
    horizontal_cuts = []
    for i in range(n_horizontal):
        y = onion.radius * (i + 1) / (n_horizontal + 1) * (1 - vertical_height)
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

import plotly.graph_objects as go
import numpy as np

import plotly.graph_objects as go
import numpy as np

def visualize_onion_and_cuts(onion, cuts):
    fig = go.Figure()

    # Draw onion outline
    theta = np.linspace(0, np.pi, 100)
    x = onion.radius * np.cos(theta)
    y = onion.radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), mode='lines', name='Onion outline', line=dict(color='black'), hoverinfo='skip'))

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
        # fig.add_trace(go.Scatter(
        #     x=[cut.start[0], cut.end[0]], 
        #     y=[cut.start[1], cut.end[1]], 
        #     mode='lines+markers', 
        #     line=dict(color='red', width=2),
        #     marker=dict(color='red', size=10),
        #     hoverinfo='skip',
            
        # ))

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
        
        height=200,
        
        # width=300,
        modebar_remove=['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines', 'download'],
        modebar_add=['drawline','eraseshape'],
        newshape=dict(line=dict(color='red', width=2)),
        newselection=dict(mode='immediate'),
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(
            range=xrange,
            scaleanchor="y",
            scaleratio=1,
            visible=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=yrange,
            visible=False
        ),
        
    )
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    return fig

def update_cuts(fig):
    logger.info("Entering update_cuts function")
    
    updated_cuts = []
    for i, shape in enumerate(fig.layout.shapes):
        if shape.type == 'line':
            start = (shape.x0, shape.y0)
            end = (shape.x1, shape.y1)
            updated_cuts.append(Cut(start, end))
            logger.info(f"Cut {i}: Start {start}, End {end}")
    logger.info(f"Total updated cuts: {len(updated_cuts)}")

    # compare the shapes in the figure to the cuts in the session state
    if updated_cuts != st.session_state.cuts:
        logger.info("Cuts updated by user interaction")
        st.session_state.cuts = updated_cuts
        st.session_state.update_triggered = True
        logger.info(f"New cuts from user interaction: {updated_cuts}")
        st.experimental_rerun()
    else:
        logger.info("No changes detected in cuts")
    return updated_cuts
        
def visualize_pieces(polygons, onion):
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
        height=250,
        width=500,
        showlegend=False,
        # title_text=f"{title} - Resulting Pieces"
        yaxis=dict(range=[0, onion.radius+0.5]),
        margin=dict(l=0, r=0, b=0, t=0),        
    )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes( scaleratio=1, showgrid=False, zeroline=False, showticklabels=False) # scaleanchor="x",    

    return fig

def initialize_session_state():
    if 'onion' not in st.session_state:
        st.session_state.onion = None
    if 'cuts' not in st.session_state:
        st.session_state.cuts = None
    if 'cutting_method' not in st.session_state:
        st.session_state.cutting_method = None
    if 'update_triggered' not in st.session_state:
        st.session_state.update_triggered = False
    
    logger.info(f"Session state initialized: onion={st.session_state.onion}, cuts={st.session_state.cuts}, cutting_method={st.session_state.cutting_method}, update_triggered={st.session_state.update_triggered}")


def create_onion():
    st.sidebar.header("Onion Parameters")
    onion_diameter = st.sidebar.slider("Onion Diameter (inches)", 1.0, 10.0, 5.0)
    n_layers = st.sidebar.slider("Number of Layers", 3, 20, 11)
    
    onion = HalfOnion(onion_diameter, n_layers)
    st.session_state.onion = onion
    logger.info(f"Onion created: diameter={onion_diameter}, layers={n_layers}")
    return onion

def select_cutting_method():
    st.sidebar.header("Cutting Method")
    cutting_method = st.sidebar.selectbox("Select Cutting Method", ["Josh's Method", "Classic", "Kenji", "Custom"])
    logger.info(f"Cutting method selected: '{cutting_method}'")
    st.session_state.cuts = []
    return cutting_method

def generate_cuts(onion, cutting_method):
    st.sidebar.header("Cut Parameters")
    if cutting_method == "Josh's Method":
        n_horizontal = st.sidebar.slider("Number of Horizontal Cuts", 2, 10, 3)
        vertical_height = st.sidebar.slider("Vertical Cut Height (fraction of radius)", 0.1, 1.0, 0.2)
        horizontal_depth = st.sidebar.slider("Horizontal Cut Depth (fraction of radius)", 0.1, 1.0, 0.85)
        n_vertical = st.sidebar.slider("Number of Vertical Cuts", 3, 20, 10)
        cuts = custom_cuts(onion, n_horizontal, vertical_height, horizontal_depth, n_vertical)
    elif cutting_method == "Classic":
        n_vertical = st.sidebar.slider("Number of Vertical Cuts", 4, 16, 10)
        n_horizontal = st.sidebar.slider("Number of Horizontal Cuts", 0, 10, 2)
        cuts = classic_cuts(onion, n_vertical, n_horizontal)
    elif cutting_method == "Kenji":  # Kenji method
        n_cuts = st.sidebar.slider("Number of Cuts", 3, 20, 10)
        pct_below = st.sidebar.slider("Target Point (fraction of radius below center)", 0.1, 0.9, 0.6)
        cuts = kenji_cuts(onion, n_cuts, pct_below)
    elif cutting_method == "Custom":
        cuts = [Cut((-onion.radius, 0), (onion.radius, 0))]
        
    
    logger.info(f"New cuts generated: {cuts}")
    return cuts


from onion_slice_component import onion_slice_component
import traceback


def display_interactive_cuts(onion, cuts):
    global logger
    st.subheader("Onion with Cuts (Interactive)")
    try:
        # logger.info(f"Generating fig_cuts with {len(cuts)} cuts")
        fig_cuts = visualize_onion_and_cuts(onion, cuts)
        # logger.info("Fig_cuts generated successfully")
        
        # logger.info("Calling slice_viz_events")
        figJSON = json.dumps(fig_cuts, cls=plotly.utils.PlotlyJSONEncoder)
        component_value = onion_slice_component(fig=figJSON)
        # logger.info(f"slice_viz_events initial return: {component_value}")
        
        # Initialize a placeholder for the component
        component_placeholder = st.empty()
        
        # Function to check for updates
        def check_for_updates():
            if component_value:
                # logger.info(f"check_for_updates: Received component value: {component_value}")
                # logger.info(f"check_for_updates: len(cuts): {len(component_value)}")
                if isinstance(component_value, list):
                    new_cuts = [Cut((shape['x0'], shape['y0']), (shape['x1'], shape['y1'])) for shape in component_value]
                    # logger.info(f"New cuts from user interaction: {new_cuts}")
                    # logger.info(f"Current cuts: {st.session_state.cuts}")
                    st.session_state.cutting_method = "Custom"
                    if new_cuts != st.session_state.cuts:
                        st.session_state.cuts = new_cuts
                        st.session_state.update_triggered = True
                        # logger.info(f"New cuts from user interaction: {new_cuts}")
                        # st.rerun()
                elif isinstance(component_value, dict) and 'error' in component_value:
                    logger.error(f"Error in Plotly: {component_value['error']}")
                    st.error(f"An error occurred in the Plotly visualization: {component_value['error']}")
        
        # Use a button to trigger updates (you can remove this if you want automatic updates)
        # if st.button("Update Cuts"):
        check_for_updates()
        
        # st.write("Drag the red lines to adjust cuts or use the 'Remove active shape' button to delete a cut. Click 'Update Cuts' to apply changes.")
        
    except Exception as e:
        logger.error(f"Error in display_interactive_cuts: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error(f"An error occurred while displaying the interactive cuts: {str(e)}")
        st.write("Please check the console for more detailed error information.")
        
def display_piece_area_distribution(areas):
    st.subheader("Piece Area Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    
    
    ax_hist.hist(areas, bins=20)
    logger.info(f"st.session_state.onion.radius: {st.session_state.onion.radius}")
    logger.info(f"st.session_state.onion.n_layers: {st.session_state.onion.n_layers}")
    ax_hist.set_xlim(0, st.session_state.onion.radius * 2/ st.session_state.onion.n_layers) 
    ax_hist.set_title("Distribution of Piece Areas")
    ax_hist.set_xlabel('Area (sq inches)')
    ax_hist.set_ylabel('Frequency')
    median_area = np.median(areas)
    std_dev_area = np.std(areas)
    plt.tight_layout()
    st.pyplot(fig_hist)
    st.caption(f"Median: {median_area:.4f} sq inches, Std Dev: {std_dev_area:.4f} sq inches")

def display_resulting_pieces(polygons, onion):
    st.subheader("Resulting Pieces")
    fig_pieces = visualize_pieces(polygons, onion)
    st.plotly_chart(fig_pieces, use_container_width=True)

def display_aspect_ratio_distribution(shapes):
    st.subheader("Piece Aspect Ratio Distribution")
    fig_square, ax_square = plt.subplots(figsize=(8, 5))
    ax_square.hist([s[2] for s in shapes], bins=20)
    ax_square.set_title("Distribution of Piece Aspect Ratio")
    ax_square.set_xlabel('Aspect Ratio (1 is perfect square)')
    ax_square.set_ylabel('Frequency')
    st.pyplot(fig_square)

def display_piece_statistics(areas, shapes):
    st.subheader("Piece Statistics")
    st.markdown(f"""
    Number of pieces: {len(areas)}  
    Average piece area: {np.mean(areas):.4f} sq inches  
    Std dev of areas: {np.std(areas):.4f} sq inches  
    Average aspect ratio (1 is perfect): {np.mean([s[2] for s in shapes]):.4f}
    """)

def display_piece_cross_sections(polygons, cutting_method):
    st.header("Piece Cross-Sections (Largest to Smallest)")
    try:
        piece_shape_plot = visualize_piece_shapes(polygons, cutting_method)
        st.plotly_chart(piece_shape_plot, use_container_width=True)
    except Exception as e:
        logger.error(f"Error in visualizing piece shapes: {str(e)}")
        st.error(f"An error occurred while visualizing piece shapes: {str(e)}")
        st.write("Please try adjusting the cut parameters.")


def encode_settings_to_url(onion, cuts, cutting_method):
    settings = {
        'diameter': onion.radius * 2,
        'n_layers': onion.n_layers,
        'cuts': [(cut.start, cut.end) for cut in cuts],
        'cutting_method': cutting_method
    }
    encoded_settings = urllib.parse.urlencode({'settings': json.dumps(settings)})
    return f"?{encoded_settings}"

def decode_settings_from_url():
    query_params = st.query_params
    if 'settings' in query_params:
        logger.info(f"Decoding settings from URL: {query_params['settings']}")
        try:
            settings = json.loads(query_params['settings'])
            onion = HalfOnion(settings['diameter'], settings['n_layers'])
            cuts = [Cut(start, end) for start, end in settings['cuts']]
            cutting_method = settings['cutting_method']
            return onion, cuts, cutting_method
        except json.JSONDecodeError as e:
            st.error(f"Invalid settings in URL: {str(e)}. Using default values. raw json {query_params['settings']}")
            logger.error(f"Error decoding settings: {str(e)}")
        except KeyError as e:
            st.error(f"Missing key in settings: {str(e)}. Using default values.")
        except Exception as e:
            st.error(f"Error decoding settings: {str(e)}. Using default values.")
    return None, None, None


def update_url(onion, cuts, cutting_method):
    url = encode_settings_to_url(onion, cuts, cutting_method)
    st.query_params.update(urllib.parse.parse_qs(url[1:]))



def main():
    st.set_page_config(layout="wide")
    st.title("Onion Cutting Simulator")

    logger.info("Starting main function")

    initialize_session_state()

    # Decode settings from URL if present
    onion, cuts, cutting_method = decode_settings_from_url()
    if onion and cuts and cutting_method:
        logger.info(f"Decoded settings from URL: {onion}, {cuts}, {cutting_method}")
        st.session_state.onion = onion
        st.session_state.cuts = cuts
        st.session_state.cutting_method = cutting_method     
    else:
        onion = create_onion()
    

    # Create sidebar for onion parameters
    with st.sidebar:
        st.header("Onion Parameters")
        diameter = st.slider("Onion Diameter (cm)", 5.0, 15.0, onion.radius * 2, 0.1)
        n_layers = st.slider("Number of Layers", 3, 15, onion.n_layers)

        if diameter != onion.radius * 2 or n_layers != onion.n_layers:
            onion = HalfOnion(diameter, n_layers)
            st.session_state.onion = onion
            st.session_state.cuts = []  # Reset cuts when onion changes

    cutting_method = select_cutting_method()
    if cutting_method != st.session_state.cutting_method or st.session_state.cuts is None or st.session_state.cuts == []:
        logger.info("Cutting method changed, generating new cuts")
        st.session_state.cutting_method = cutting_method
        st.session_state.cuts = generate_cuts(onion, cutting_method)
        logger.info(f"Generated new cuts: {st.session_state.cuts}")
    else:
        logger.info("Using existing cuts from session state")

    logger.info(f"Current cuts: {st.session_state.cuts}")

    # st.header("Interactive Onion Cuts and Piece Size Distribution")
    col1, col2 = st.columns(2)

    with col1:
        display_interactive_cuts(onion, st.session_state.cuts)

    # Apply cuts and calculate results
    polygons = apply_cuts(onion, st.session_state.cuts)
    areas, shapes = calculate_areas_and_shapes(polygons)
    logger.info(f"Number of polygons: {len(polygons)}, Number of areas: {len(areas)}")

    with col2:
        display_piece_area_distribution(areas)
        
    col3, col4 = st.columns(2)

    with col3:
        display_resulting_pieces(polygons, onion)

    with col4:
        display_piece_statistics(areas, shapes)
    
    col5, col6 = st.columns(2)
    with col5:
        display_aspect_ratio_distribution(shapes)

    logger.info("Attempting to display piece cross-sections")
    display_piece_cross_sections(polygons, cutting_method)

    # Update the URL with the current settings
    update_url(onion, st.session_state.cuts, st.session_state.cutting_method)

    # Reset update trigger
    if st.session_state.update_triggered:
        logger.info("Resetting update trigger")
        st.session_state.update_triggered = False

    logger.info("Main function completed")

if __name__ == "__main__":
    main()



