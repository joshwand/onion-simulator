"""
Classic cutting method implementation.
"""

import numpy as np
from typing import List

from core.models import BaseOnion
from models.cut import Cut, CrossCut
from cutting_methods.base import CuttingMethod, CuttingMethodFactory


class ClassicCuttingMethod(CuttingMethod):
    """
    Classic cutting method with evenly spaced vertical and horizontal cuts.
    """
    
    def generate_cuts(self, n_vertical: int = 10, n_horizontal: int = 2) -> List[Cut]:
        """
        Generate cuts based on the classic method.
        
        Args:
            n_vertical: Number of vertical cuts
            n_horizontal: Number of horizontal cuts
            
        Returns:
            List of Cut objects
        """
        cuts = []
        
        # Vertical cuts
        for i in range(1, n_vertical):
            x = -self.onion.radius + i * self.onion.radius * 2 / n_vertical
            cuts.append(Cut((x, 0), (x, self.onion.radius)))
        
        # Horizontal cuts (excluding the bottom static cut)
        for i in range(1, n_horizontal + 1):
            y = i * self.onion.radius / (n_horizontal + 1)
            cuts.append(Cut((-self.onion.radius, y), (self.onion.radius, y)))
        
        # Add static bottom cut (always included)
        cuts.append(Cut((-self.onion.radius, 0), (self.onion.radius, 0)))
        
        return cuts
    
    @staticmethod
    def get_default_params():
        """
        Get the default parameters for the classic method.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'n_vertical': 10,
            'n_horizontal': 2,            
        }


# Register the method with the factory
CuttingMethodFactory.register('Classic', ClassicCuttingMethod) 