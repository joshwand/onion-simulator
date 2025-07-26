# Onion Simulator: Technical Context

## Technology Stack

### Core Technologies

| Category | Technologies | Purpose |
|----------|--------------|---------|
| **Language** | Python 3.x | Primary programming language |
| **Web Framework** | Streamlit | Application UI, interactivity, and deployment |
| **Data Processing** | NumPy | Numerical operations and array manipulation |
| **Visualization** | Plotly, Matplotlib | Interactive charts and static visualizations |
| **2D Geometry** | Shapely | 2D geometric operations and analysis |
| **3D Geometry** | Trimesh, PyRender | 3D mesh operations and rendering |
| **Graphics** | SVG (via xml.etree.ElementTree) | Profile importing and processing |
| **Data Interchange** | JSON, URL query parameters | Configuration sharing and persistence |
| **Testing** | Pytest | Unit testing and integration testing |
### Dependencies

The project relies on the following key packages defined in `requirements.txt`:

```
streamlit       # Web application framework
numpy           # Array operations
matplotlib      # Some plotting capabilities
plotly          # Interactive visualizations
shapely         # 2D geometric operations
scipy           # Scientific computing utilities
trimesh         # 3D mesh processing
pyrender        # 3D rendering
pytest          # Testing framework
```

### Development Environment

- **Local Development**: The application runs locally via the Streamlit server
- **Version Control**: Git version control system
- **Python Environment**: Virtual environment recommended for dependency isolation
- **File Organization**: Modular structure separated by functional areas

## System Components

### Core Module Structure

```
onion-simulator/
├── core/                  # Core domain models and utilities
│   ├── config.py          # Configuration settings
│   ├── events.py          # Event system
│   ├── exceptions.py      # Custom exceptions
│   └── models.py          # Domain model definitions
├── models/                # Data models
│   ├── onion_2d.py        # 2D onion representation (legacy)
│   ├── onion_3d.py        # 3D onion representation
│   ├── svg_profile.py     # SVG profile handling
│   ├── svg_profile_onion.py # SVG-based onion profile
│   └── cut.py             # Cut definitions and operations
├── cutting_methods/       # Cutting strategy implementations
│   ├── base.py            # Base class and factory
│   ├── classic.py         # Classical cutting method
│   ├── kenji.py           # Kenji's method implementation
│   └── josh.py            # Josh's method implementation
├── services/              # Service layer
│   ├── analysis_service.py    # Analysis and measurement
│   ├── geometry_service.py    # Geometric operations
│   ├── persistence_service.py # State management
│   └── visualization_service.py # Visualization utilities
├── utils/                 # Utility functions
│   └── svg_parser.py      # SVG parsing functionality
├── pages/                 # Streamlit pages
│   ├── simulator_2d.py    # 2D simulator interface (legacy)
│   └── simulator_3d.py    # 3D simulator interface (active)
└── app.py                 # Main application entry point
```

### Technical Implementation Details

#### 2D Simulator (Legacy)
- Uses Shapely for geometry calculations
- Represents the onion as layers of concentric half-circles
- Applies cuts as line segments
- Calculates intersections to determine resulting pieces
- Analyzes area distribution of resulting polygons

#### 3D Simulator (Active Development)
- Uses SVG profiles to define realistic onion shapes
- Employs Trimesh for 3D mesh operations
- Represents cuts as planes in 3D space
- Visualizes results using Plotly 3D capabilities
- Calculates volumes and surface areas of resulting pieces

#### Cutting Methods
- Implements Strategy pattern via the base `CuttingMethod` class
- Uses Factory pattern (`CuttingMethodFactory`) to create specific implementations
- Each method calculates cut positions based on its algorithm and parameters
- Methods are pluggable and configurable

#### Data Flow
1. User selects simulator (2D/3D) and configures onion parameters
2. User selects cutting method and adjusts its parameters
3. Application creates the onion model and applies the cutting method
4. Geometry service processes the cuts and calculates the resulting pieces
5. Analysis service measures properties of the pieces
6. Visualization service renders the results
7. (Optional) Persistence service encodes the configuration to URL parameters

## Technical Constraints

### Performance Considerations
- 3D mesh operations can be computationally expensive
- Complex SVG profiles may impact load time
- Interactive visualizations need to remain responsive

### Browser Compatibility
- Streamlit applications should work in modern browsers
- WebGL support required for 3D visualizations

### Deployment Limitations
- Streamlit sharing has memory constraints
- Large meshes may exceed available resources

## Development Workflow

### Setting Up the Environment
```bash
# Clone the repository
git clone https://github.com/joshwand/onion-simulator.git
cd onion-simulator

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Development Practices
- Ensure backward compatibility for shared URLs
- Use typed function signatures where possible
- Document complex geometric operations
- Keep visualization code separate from business logic 
- Use caching where appropriate to improve performance