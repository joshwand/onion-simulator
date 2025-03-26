# Onion Simulator 3D: Comprehensive Design Document

## 1. Project Overview

### 1.1. Purpose
Enhance the existing 2D onion cutting simulator by adding 3D analysis capabilities while maintaining the original 2D functionality as a separate option. The enhanced application will:
1. Model onions with realistic shapes
2. Visualize and analyze cuts from multiple perspectives
3. Calculate 3D volumes and surface areas
4. Compare cutting techniques in three dimensions

### 1.2. Core Features
- Multi-page application (2D and 3D simulators)
- Realistic parametric onion shape model
- Synchronized side-view and top-view visualizations
- Cross-cut controls with configurable parameters
- Surface area analysis for all piece faces
- Distribution visualizations for piece metrics
- Shareable URLs for both 2D and 3D configurations

## 2. Enhanced Architecture

### 2.1. Project Structure
```
onion-simulator/
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── models.py              # Core data models (shared)
│   ├── events.py              # Event system for UI updates
│   └── exceptions.py          # Custom exceptions
├── models/
│   ├── __init__.py
│   ├── onion_2d.py            # 2D onion model
│   ├── onion_3d.py            # 3D onion model
│   └── cut.py                 # Cut models (2D and 3D)
├── services/
│   ├── __init__.py
│   ├── geometry_service.py    # Geometric calculations
│   ├── analysis_service.py    # Statistical analysis
│   └── persistence_service.py # URL & state management
├── controllers/
│   ├── __init__.py
│   ├── cutting_controller.py  # Controls cut generation
│   └── view_controller.py     # Controls view rendering
├── views/
│   ├── __init__.py
│   ├── onion_view.py          # Onion visualization
│   ├── cut_view.py            # Cut visualization
│   ├── pieces_view.py         # Pieces visualization
│   └── stats_view.py          # Statistics visualization
├── cutting_methods/
│   ├── __init__.py
│   ├── base.py                # Abstract base class
│   ├── classic.py             # Classic implementation
│   ├── kenji.py               # Kenji implementation
│   └── josh.py                # Josh implementation
├── utils/
│   ├── __init__.py
│   ├── performance.py         # Performance monitoring
│   ├── validation.py          # Input validation
│   └── logging.py             # Enhanced logging
├── tests/
│   ├── __init__.py
│   ├── test_models.py         # Unit tests for models
│   ├── test_cutting.py        # Unit tests for cutting
│   └── test_geometry.py       # Unit tests for geometry
├── onion_slice_component/     # Interactive component (existing)
├── app.py                     # Main landing page
└── pages/                     # Streamlit pages
    ├── simulator_2d.py        # 2D simulator
    └── simulator_3d.py        # 3D simulator
```

### 2.2. Key Design Patterns

#### Factory Pattern
Used for creating visualizations, cutting methods, and model instances:

```python
class CuttingMethodFactory:
    @staticmethod
    def create_method(method_name, onion, **params):
        if method_name == "classic":
            return ClassicCuttingMethod(onion, **params)
        elif method_name == "kenji":
            return KenjiCuttingMethod(onion, **params)
        elif method_name == "josh":
            return JoshCuttingMethod(onion, **params)
        else:
            raise ValueError(f"Unknown cutting method: {method_name}")
```

#### Strategy Pattern
For interchangeable cutting methods:

```python
class CuttingMethod(ABC):
    @abstractmethod
    def generate_cuts(self, **params):
        pass
        
class ClassicCuttingMethod(CuttingMethod):
    def generate_cuts(self, **params):
        # Implementation here
        
class KenjiCuttingMethod(CuttingMethod):
    def generate_cuts(self, **params):
        # Implementation here
```

#### Observer Pattern
For UI updates when data changes:

```python
class EventBus:
    _subscribers = defaultdict(list)
    
    @classmethod
    def subscribe(cls, event_type, callback):
        cls._subscribers[event_type].append(callback)
    
    @classmethod
    def publish(cls, event_type, data=None):
        for callback in cls._subscribers[event_type]:
            callback(data)
```

### 2.3. Dependency Management
- Clear interfaces between components
- Configuration-based defaults
- Centralized event system for UI updates
- Unified error handling

## 3. Data Models

### 3.1. Base Onion Model
Shared properties and methods between 2D and 3D models:

```python
class BaseOnion(ABC):
    """Abstract base class for onion models"""
    
    def __init__(self, diameter, n_layers, start_y=0.5, end_y=0.8, p1=(0.2, 1.2), p2=(0.8, 1.0)):
        self.radius = diameter / 2
        self.n_layers = n_layers
        self.start_y = start_y
        self.end_y = end_y
        self.p1 = p1
        self.p2 = p2
        self.layer_radii = self.calculate_layer_radii()
    
    def layer_thickness_curve(self, x):
        """Calculate the thickness at a normalized radius"""
        points = [(0, self.start_y), self.p1, self.p2, (1.0, self.end_y)]
        x_values, y_values = zip(*points)
        spline = interp.CubicSpline(x_values, y_values)
        return spline(x)
    
    @abstractmethod
    def calculate_layer_radii(self):
        """Calculate the radius of each layer"""
        pass
    
    @abstractmethod
    def create_layer_boundaries(self):
        """Create the boundaries between layers"""
        pass
```

### 3.2. 3D Onion Model
Enhanced model with SVG profile-based realistic onion shape


```python
class RealisticOnion(BaseOnion):
    """3D realistic onion model based on SVG profiles"""
    
    def __init__(self, 
                 max_diameter,
                 profile_name):
        """
        Create a 3D onion model based on SVG profiles
        
        Args:
            max_diameter: Maximum diameter of the widest profile (in units)
            profile_name: Name of the selected real-life onion profile
        """
        self.max_diameter = max_diameter
        self.profile_name = profile_name
        self.profiles = self.load_svg_profiles(profile_name)
        self.n_layers = len(self.profiles)
        
    def load_svg_profiles(self, profile_name):
        """
        Load Bezier curve paths from SVG files
        
        The SVG contains the profiles of the layers, with the root side down,
        showing only one side of the root axis. These profiles will be revolved
        180 degrees to form the boundaries of the layers.

        ****The SVG is in the XY plane, but we are using it to represent the XZ plane!****

        """
        # Implementation loading SVG paths and converting to Bezier curves
        pass
    
    def calculate_layer_boundaries(self):
        """
        Calculate the boundaries of each layer by revolving the profiles
        
        Returns 3D meshes representing each layer boundary
        """
        # Implementation revolving the bezier curves 180 degrees
        # to form the half-onion structure
        pass
    
    def generate_surface_points(self, resolution=50):
        """Generate points on the surface of the onion"""
        # Implementation based on the revolved profiles
        pass
    
    def get_scale_factor(self):
        """
        Calculate the scale factor to achieve the desired max_diameter
        """
        # Find the maximum width in the profiles and calculate the scaling factor
        pass
```

### 3.3. Cut Models

#### 2D Cut (Existing)
```python
class Cut:
    def __init__(self, start, end):
        self.start = (float(start[0]), float(start[1]))  # (x, y) tuple
        self.end = (float(end[0]), float(end[1]))        # (x, y) tuple

    def as_shapely(self):
        return LineString([self.start, self.end])
    
    def __repr__(self) -> str:
        return f"Cut({self.start}, {self.end})"
```

#### 3D Cross Cut (New)
```python
class CrossCut:
    def __init__(self, angle, distance_from_center=0):
        """
        Create a cut across the onion in 3D space
        
        Args:
            angle: Angle in radians from horizontal
            distance_from_center: Distance from center (0 = through center)
        """
        self.angle = angle
        self.distance_from_center = distance_from_center
    
    def as_plane(self):
        """Return a representation as a plane for 3D calculations"""
        # Implementation using a plane equation ax + by + cz + d = 0
    
    def __repr__(self) -> str:
        return f"CrossCut(angle={self.angle}, distance={self.distance_from_center})"
```

### 3.4. 3D Piece Model
```python
class OnionPiece3D:
    def __init__(self, geometry):
        """
        Represents a 3D piece of onion
        
        Args:
            geometry: 3D geometric representation (e.g., from trimesh)
        """
        self.geometry = geometry
        self._volume = None
        self._surface_areas = None
    
    @property
    def volume(self):
        """Calculate the volume of the piece"""
        if self._volume is None:
            self._volume = self.geometry.volume
        return self._volume
    
    @property
    def surface_areas(self):
        """Calculate surface areas for different faces"""
        if self._surface_areas is None:
            # Calculate areas for the faces
            # Group faces by type (external, layer, cut)
            self._surface_areas = {
                'external': 0.0,  # External surface area
                'layer': 0.0,     # Layer interface area
                'cut': 0.0        # Cut surface area
            }
        return self._surface_areas
    
    @property
    def total_surface_area(self):
        """Calculate the total surface area"""
        areas = self.surface_areas
        return areas['external'] + areas['layer'] + areas['cut']
```

## 4. Cutting Methods

### 4.1. Base Cutting Method
```python
class CuttingMethod(ABC):
    def __init__(self, onion):
        self.onion = onion
    
    @abstractmethod
    def generate_cuts(self, **params):
        """Generate cuts based on the method's algorithm"""
        pass
    
    @abstractmethod
    def generate_cross_cuts(self, **params):
        """Generate cross cuts for 3D simulation"""
        pass
        
    @abstractmethod
    def get_editable_parameters(self):
        """Get parameters that can be edited by the user"""
        pass
        
    @abstractmethod
    def update_parameters(self, **params):
        """Update cutting parameters based on user input"""
        pass
```

### 4.2. Extending Existing Methods
Each of the three current cutting methods (Classic, Kenji, Josh) will be extended to support 3D cross-cuts with appropriate parameters and maintain the same editable behavior as in the 2D simulator.

Example for Classic method:
```python
class ClassicCuttingMethod(CuttingMethod):
    def generate_cuts(self, n_vertical, n_horizontal):
        cuts = []
        # Current implementation as in 2D simulator...
        return cuts
    
    def generate_cross_cuts(self, n_cross_cuts):
        cross_cuts = []
        # Evenly distribute cross cuts
        for i in range(n_cross_cuts):
            angle = np.pi * (i + 1) / (n_cross_cuts + 1)
            cross_cuts.append(CrossCut(angle))
        return cross_cuts
        
    def get_editable_parameters(self):
        return {
            'n_vertical': self.n_vertical,
            'n_horizontal': self.n_horizontal,
            'n_cross_cuts': self.n_cross_cuts
        }
        
    def update_parameters(self, **params):
        if 'n_vertical' in params:
            self.n_vertical = params['n_vertical']
        if 'n_horizontal' in params:
            self.n_horizontal = params['n_horizontal']
        if 'n_cross_cuts' in params:
            self.n_cross_cuts = params['n_cross_cuts']
```

## 5. Services

### 5.1. Geometry Service
Responsible for 3D geometric operations:

```python
class GeometryService:
    @staticmethod
    def apply_cuts_3d(onion, cuts, cross_cuts):
        """Apply both standard cuts and cross-cuts to get 3D pieces"""
        # Implementation using 3D geometric libraries
    
    @staticmethod
    def calculate_volume(piece):
        """Calculate the volume of a 3D piece"""
        return piece.volume
    
    @staticmethod
    def calculate_surface_areas(piece):
        """Calculate the surface areas for a piece"""
        return piece.surface_areas
```

### 5.2. Analysis Service
Responsible for statistical analysis:

```python
class AnalysisService:
    @staticmethod
    def calculate_volume_statistics(pieces):
        """Calculate statistics on volumes"""
        volumes = [piece.volume for piece in pieces]
        return {
            'min': min(volumes),
            'max': max(volumes),
            'mean': np.mean(volumes),
            'median': np.median(volumes),
            'std_

```

## 6. Standard Coordinate System

To maintain consistency throughout the 3D simulator, we define the following standard coordinate system:

### 6.1. Axes Definition
- **Z-axis**: The longitude axis of the onion, around which the profiles are revolved. The cut side of the onion faces down (negative Z).
- **X-axis**: Horizontal axis when viewed along the Z-axis.
- **Y-axis**: Vertical axis when viewed along the Z-axis.

### 6.2. Standard Views
- **Cross-section view**: XY plane, looking along the Z-axis
- **Top view**: ZX plane, with the root side up (positive Z)

### 6.3. Coordinate System Visualization
The visualization will include indicators for the coordinate system to help users understand the orientation:
```
          Y
          ↑
          |
          |
          |
          O--→ X
         /
        /
       ↙
      Z
```

## 7. UI Components

### 7.1. Profile Selection
```python
class ProfileSelector:
    """UI component for selecting onion profiles"""
    
    def __init__(self):
        self.available_profiles = self.load_available_profiles()
        
    def load_available_profiles(self):
        """Load the list of available real-life onion profiles"""
        # Implementation returning a dictionary of profile names and preview images
        
    def render(self, selected_profile, on_change):
        """Render the profile selection UI with thumbnails"""
        # Implementation for rendering selection UI
```

### 7.2. Scale Setting
```python
class ScaleSetting:
    """UI component for setting the onion scale"""
    
    def render(self, current_diameter, on_change):
        """Render the scale setting UI"""
        # Implementation for diameter input
```

### 7.3. Cut Editor
The cut editor will maintain the same functionality as the 2D simulator, allowing users to:
- Add, remove, or modify cuts
- Adjust cutting parameters
- Visualize the effects of cuts in real-time
