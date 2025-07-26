# Onion Simulator

The Onion Simulator is a tool for experimenting with different onion cutting techniques and analyzing the results. It provides both 2D and 3D simulations to help you find the optimal way to dice an onion.

## Features

### 2D Simulator
- Visualize cuts in a cross-section view
- Analyze piece areas and distributions
- Compare different cutting methods (Classic, Kenji's, Josh's)
- Create custom cutting patterns
- Shareable URLs for your cutting configurations

### 3D Simulator
- Model onions with realistic shapes
- Visualize cuts from multiple perspectives
- Calculate volumes and surface areas
- Analyze piece distribution in three dimensions
- Compare cutting techniques in 3D space

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/joshwand/onion-simulator.git
cd onion-simulator
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run onion_simulator/app.py
```

## Usage

1. Choose between the 2D and 3D simulator
2. Select a cutting method from the sidebar
3. Adjust parameters to customize your cuts
4. Analyze the resulting pieces
5. Share your configuration using the generated URL

## Cutting Methods

1. **Josh's Method**: A novel approach optimized for uniform piece size
2. **Classic**: Traditional vertical and horizontal cuts
3. **Kenji**: Method inspired by Kenji López-Alt's technique with cuts targeting a point
4. **Custom**: Create your own cutting pattern

## TODO

- Improve UI for manipulating cuts including deletion
- Add more advanced geometry calculations
- Support for custom onion shapes
- Export/import of cutting configurations

## DONE
- Multi-page application with 2D and 3D simulators
- Shareable URLs for configurations
- 3D visualization with cross-sectional views
- Volume and surface area analysis
- Realistic onion shape modeling

## License

This project is licensed under the MIT License - see the LICENSE file for details.
