"""
Josh's cutting method implementation.
"""

import numpy as np
from typing import List

from core.models import BaseOnion
from models.cut import Cut, CrossCut
from cutting_methods.base import CuttingMethod, CuttingMethodFactory


class JoshCuttingMethod(CuttingMethod):
    """
    Josh's cutting method with optimized horizontal and vertical cuts.
    """
    
    def generate_cuts(self, n_horizontal: int = 3, vertical_height: float = 0.17, 
                    horizontal_depth: float = 0.85, n_vertical: int = 10) -> List[Cut]:
        """
        Generate cuts based on Josh's method.
        
        Args:
            n_horizontal: Number of horizontal cuts
            vertical_height: Vertical cut depth (fraction of radius)
            horizontal_depth: Horizontal cut depth (fraction of radius)
            n_vertical: Number of vertical cuts
            
        Returns:
            List of Cut objects
        """
        cuts = []
        layer_width = self.onion.radius / self.onion.n_layers

        # Calculate vertical cut positions
        vertical_positions = [
            -self.onion.radius + i * self.onion.radius * 2 / (n_vertical - 1) 
            for i in range(n_vertical)
        ]

        # Find the vertical cuts closest to the desired horizontal depth
        left_cut_index = np.argmin(np.abs(np.array(vertical_positions) + self.onion.radius * (1 - horizontal_depth)))
        right_cut_index = np.argmin(np.abs(np.array(vertical_positions) - self.onion.radius * (1 - horizontal_depth)))

        # Horizontal cuts
        horizontal_cuts = []
        for i in range(n_horizontal):
            y = self.onion.radius * (i + 1) / (n_horizontal + 1) * (1 - vertical_height)
            start_x_left = -np.sqrt(self.onion.radius**2 - y**2)
            end_x_left = vertical_positions[left_cut_index]
            start_x_right = np.sqrt(self.onion.radius**2 - y**2)
            end_x_right = vertical_positions[right_cut_index]
            horizontal_cuts.append((start_x_left, end_x_left, y))
            horizontal_cuts.append((end_x_right, start_x_right, y))

        # Add horizontal cuts to the cuts list
        for start_x, end_x, y in horizontal_cuts:
            cuts.append(Cut((start_x, y), (end_x, y)))

        # Function to check if a vertical cut is too close to the edge
        def is_cut_too_narrow(x, y):
            edge_x = np.sqrt(self.onion.radius**2 - y**2)
            return abs(abs(x) - edge_x) < layer_width

        # Vertical cuts with width checking
        topmost_y = max(cut[2] for cut in horizontal_cuts) if horizontal_cuts else 0
        for i, x in enumerate(vertical_positions):
            y_start = np.sqrt(self.onion.radius**2 - x**2) if abs(x) < self.onion.radius else 0
            
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
            
            # Add the vertical cut if y_start is valid
            if y_start > 0:  # Ensure the cut is within the circle
                cuts.append(Cut((x, y_start), (x, y_end)))

        # Add static bottom cut
        cuts.append(Cut((-self.onion.radius, 0), (self.onion.radius, 0)))

        return cuts
    
    
    @staticmethod
    def get_default_params():
        """
        Get the default parameters for Josh's method.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'n_horizontal': 3,
            'vertical_height': 0.17,
            'horizontal_depth': 0.85,
            'n_vertical': 10,            
        }


# Register the method with the factory
CuttingMethodFactory.register("Josh's Method", JoshCuttingMethod) 