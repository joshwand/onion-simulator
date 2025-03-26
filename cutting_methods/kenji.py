"""
Kenji cutting method implementation.
"""

import numpy as np
from typing import List

from core.models import BaseOnion
from models.cut import Cut, CrossCut
from cutting_methods.base import CuttingMethod, CuttingMethodFactory


class KenjiCuttingMethod(CuttingMethod):
    """
    Kenji cutting method with cuts targeting a point below the center.
    """
    
    def generate_cuts(self, n_cuts: int = 10, pct_below: float = 0.6) -> List[Cut]:
        """
        Generate cuts based on Kenji's method.
        
        Args:
            n_cuts: Number of cuts
            pct_below: Target point (fraction of radius below center)
            
        Returns:
            List of Cut objects
        """
        cuts = []
        target_point = (0, -pct_below * self.onion.radius)  # Point below the center
        
        n_cuts += 2  # Account for the special cases
        for i in range(n_cuts):
            angle = np.pi * i / (n_cuts - 1)
            start_x = self.onion.radius * np.cos(angle)
            start_y = self.onion.radius * np.sin(angle)
            
            # Skip cuts at (-r, 0) and (r, 0)
            if abs(start_x) == self.onion.radius and start_y == 0:
                continue
            
            # Calculate the intersection with the circle
            dx = start_x - target_point[0]
            dy = start_y - target_point[1]
            a = dx**2 + dy**2
            b = 2 * (dx * target_point[0] + dy * target_point[1])
            c = target_point[0]**2 + target_point[1]**2 - self.onion.radius**2
            
            # Solve quadratic equation
            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                t = (-b - np.sqrt(discriminant)) / (2 * a)
                end_x = target_point[0] + t * dx
                end_y = target_point[1] + t * dy
                
                cuts.append(Cut((start_x, start_y), (end_x, end_y)))
        
        # Add static bottom cut (always included)
        cuts.append(Cut((-self.onion.radius, 0), (self.onion.radius, 0)))
        
        return cuts
    
    
    
    @staticmethod
    def get_default_params():
        """
        Get the default parameters for Kenji's method.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'n_cuts': 10,
            'pct_below': 0.6,            
        }


# Register the method with the factory
CuttingMethodFactory.register('Kenji', KenjiCuttingMethod) 