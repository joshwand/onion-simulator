# Onion Simulator: System Patterns

## System Architecture

The Onion Simulator employs a modular architecture that separates concerns while maintaining clean interfaces between components. The system is built using the following architectural patterns:

### Multi-Page Streamlit Application

The application is built on Streamlit, with multiple pages for different simulators:
- `app.py`: Main entry point and navigation
- `pages/simulator_2d.py`: Legacy 2D simulation interface
- `pages/simulator_3d.py`: Current 3D simulation interface

### Domain-Driven Design Influences

The codebase is organized around domain concepts rather than technical layers:

```mermaid
graph TD
    App[app.py] --> Pages[pages/]
    Pages --> Sim2D[simulator_2d.py]
    Pages --> Sim3D[simulator_3d.py]
    
    Sim2D --> Models[models/]
    Sim2D --> Methods[cutting_methods/]
    Sim2D --> Services[services/]
    
    Sim3D --> Models
    Sim3D --> Methods
    Sim3D --> Services
    
    Models --> Onion2D[onion_2d.py]
    Models --> Onion3D[onion_3d.py]
    Models --> SVGProfile[svg_profile.py]
    Models --> SVGOnion[svg_profile_onion.py]
    Models --> Cut[cut.py]
    
    Methods --> Base[base.py]
    Methods --> Classic[classic.py]
    Methods --> Kenji[kenji.py]
    Methods --> Josh[josh.py]
    
    Services --> Analysis[analysis_service.py]
    Services --> Geometry[geometry_service.py]
    Services --> Persistence[persistence_service.py]
    Services --> Visualization[visualization_service.py]
    
    SVGProfile --> Utils[utils/]
    SVGOnion --> Utils
    Utils --> SVGParser[svg_parser.py]
    
    classDef legacy fill:#f9f,stroke:#333,stroke-width:1px;
    class Sim2D,Onion2D legacy;
```

## Key Design Patterns

### Strategy Pattern: Cutting Methods

The cutting methods are implemented using the Strategy pattern, allowing different cutting algorithms to be used interchangeably:

```mermaid
classDiagram
    class CuttingMethod {
        <<abstract>>
        +apply(onion, parameters)
        +get_parameters_schema()
        +get_name()
        +get_description()
    }
    
    class ClassicCuttingMethod {
        +apply(onion, parameters)
        +get_parameters_schema()
    }
    
    class KenjiCuttingMethod {
        +apply(onion, parameters)
        +get_parameters_schema()
    }
    
    class JoshCuttingMethod {
        +apply(onion, parameters)
        +get_parameters_schema()
    }
    
    CuttingMethod <|-- ClassicCuttingMethod
    CuttingMethod <|-- KenjiCuttingMethod
    CuttingMethod <|-- JoshCuttingMethod
```

### Factory Pattern: Method Creation

The `CuttingMethodFactory` creates instances of specific cutting methods based on name:

```python
class CuttingMethodFactory:
    @staticmethod
    def create(method_name, **kwargs):
        if method_name == "classic":
            return ClassicCuttingMethod(**kwargs)
        elif method_name == "kenji":
            return KenjiCuttingMethod(**kwargs)
        elif method_name == "josh":
            return JoshCuttingMethod(**kwargs)
        else:
            raise ValueError(f"Unknown cutting method: {method_name}")
```

### Service Pattern: Domain Operations

Services encapsulate related functionality and provide a clean API for domain operations:

- `GeometryService`: Handles geometric calculations (intersections, volumes)
- `AnalysisService`: Analyzes results (statistics, piece distribution)
- `VisualizationService`: Creates visualizations of onions and cuts
- `PersistenceService`: Manages state persistence and URL generation

### Builder Pattern: Onion Construction

The 3D onion models use a builder-like pattern for progressive construction:

```python
onion = Onion3D()
onion.set_profile(svg_profile)
onion.set_parameters(diameter=5.0, height=4.0)
onion.generate_layers(n_layers=10)
onion.finalize()
```

## Component Relationships

### Model Relationships

```mermaid
classDiagram
    class Onion2D {
        +radius: float
        +n_layers: int
        +layer_radii: list
        +calculate_layer_radii()
        +create_layer_boundaries()
    }
    
    class Onion3D {
        +profile: SVGProfile
        +height: float
        +diameter: float
        +layers: list
        +mesh: Trimesh
        +generate_layers()
        +apply_cuts(cuts)
        +get_pieces()
    }
    
    class SVGProfile {
        +points: list
        +normalize()
        +get_points_at_height(height)
        +get_outline()
    }
    
    class SVGProfileOnion {
        +profile: SVGProfile
        +diameter: float
        +height: float
        +create_3d_model()
    }
    
    class Cut {
        +start_point: Point3D
        +end_point: Point3D
        +direction: Vector3D
        +as_plane()
        +apply_to_mesh(mesh)
    }
    
    Onion3D --> SVGProfile: uses
    SVGProfileOnion --> SVGProfile: uses
    Onion3D --> Cut: applies
```

### Service Interactions

```mermaid
sequenceDiagram
    participant User
    participant UI as Simulator UI
    participant CM as CuttingMethod
    participant GS as GeometryService
    participant AS as AnalysisService
    participant VS as VisualizationService
    participant PS as PersistenceService
    
    User->>UI: Select method & parameters
    UI->>CM: Create method(parameters)
    UI->>UI: Create onion model
    UI->>CM: Apply method to onion
    CM->>GS: Calculate cuts
    GS-->>CM: Return cut objects
    CM-->>UI: Return cuts
    UI->>GS: Apply cuts to onion
    GS-->>UI: Return pieces
    UI->>AS: Analyze pieces
    AS-->>UI: Return statistics
    UI->>VS: Visualize results
    VS-->>UI: Return visualizations
    UI->>PS: Encode configuration
    PS-->>UI: Return URL
    UI-->>User: Display results & URL
```

## Data Flow Architecture

The data flow through the system follows a clear pattern:

1. **Configuration**: User selects parameters through the UI
2. **Model Creation**: Onion model is instantiated based on parameters
3. **Method Application**: Selected cutting method generates cuts
4. **Geometric Processing**: Cuts are applied to the onion model
5. **Analysis**: Resulting pieces are analyzed for metrics
6. **Visualization**: Results are visualized for the user
7. **Persistence**: Configuration is encoded for sharing

### Key Data Structures

- **Onion Models**: Represent the onion geometry (2D polygons or 3D mesh)
- **Cuts**: Represent cutting planes or lines through the onion
- **Pieces**: Resulting fragments after cuts are applied
- **Statistics**: Metrics about the pieces (size distribution, uniformity)
- **Visualizations**: Visual representations for the UI

## Error Handling Approach

The system uses structured exception handling:

```python
try:
    # Operation that might fail
    result = operation()
except GeometryError as e:
    # Handle specific geometry errors
    logger.error(f"Geometry error: {str(e)}")
    st.error("There was a problem with the geometry calculation.")
except ValueError as e:
    # Handle validation errors
    logger.warning(f"Invalid input: {str(e)}")
    st.warning(f"Please check your inputs: {str(e)}")
except Exception as e:
    # Catch-all for unexpected errors
    logger.exception(f"Unexpected error: {str(e)}")
    st.error("An unexpected error occurred. Please try again.")
```

## Event System

The core `events.py` module implements a lightweight event system for component communication:

```python
# Publishing an event
EventBus.publish("onion_cut_applied", {
    "onion_id": onion.id,
    "cut_method": method.name,
    "cut_count": len(cuts),
    "piece_count": len(pieces)
})

# Subscribing to an event
@EventBus.subscribe("onion_cut_applied")
def log_cut_application(event_data):
    logger.info(f"Cut applied: {event_data['cut_method']} " +
                f"with {event_data['cut_count']} cuts " +
                f"resulting in {event_data['piece_count']} pieces")
```

## Performance Patterns

### Caching

Computationally expensive operations use Streamlit's caching:

```python
@st.cache_data
def apply_cuts_to_onion(onion, cuts, method_name):
    # Expensive operation
    return pieces
```

### Progressive Loading

3D visualizations should (but don't currently) use progressive loading techniques to maintain responsiveness:

1. Show wireframe model first
2. Add surface details progressively
3. Apply advanced lighting last 