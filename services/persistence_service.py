"""
Persistence service for URL and state management.
"""

import json
import urllib.parse
from typing import Dict, List, Any, Tuple, Optional, Union

import streamlit as st

from core.config import URL_SETTINGS_KEY
from core.exceptions import PersistenceError
from models.onion_2d import HalfOnion
from models.svg_profile_onion import SvgProfileOnion
from models.cut import Cut, CrossCut


class PersistenceService:
    """Service for URL and state management."""
    
    @staticmethod
    def encode_settings_to_url_2d(onion: HalfOnion, cuts: List[Cut], cutting_method: str) -> str:
        """
        Encode 2D settings to a URL query string.
        
        Args:
            onion: The 2D onion model
            cuts: List of cuts
            cutting_method: The cutting method used
            
        Returns:
            URL query string
        """
        settings = {
            'mode': '2d',
            'diameter': onion.radius * 2,
            'n_layers': onion.n_layers,
            'cuts': [(cut.start, cut.end) for cut in cuts],
            'cutting_method': cutting_method,
            'start_y': onion.start_y,
            'end_y': onion.end_y,
            'p1': onion.p1,
            'p2': onion.p2
        }
        
        encoded_settings = urllib.parse.urlencode({'settings': json.dumps(settings)})
        return f"?{encoded_settings}"
    
    @staticmethod
    def encode_settings_to_url_3d(onion: SvgProfileOnion, cuts: List[Cut], cutting_method: str) -> str:
        """
        Encode 3D SVG profile-based onion settings to a URL query string.
        
        Args:
            onion: The 3D SVG profile-based onion model
            cuts: List of 2D cuts
            cutting_method: The cutting method used
            
        Returns:
            URL query string
        """
        settings = {
            'mode': '3d',
            'onion_type': 'svg',
            'diameter': onion.radius * 2,
            'cuts': [(cut.start, cut.end) for cut in cuts],
            'cutting_method': cutting_method,
            'svg_file_path': onion.svg_file_path
        }
        
        encoded_settings = urllib.parse.urlencode({'settings': json.dumps(settings)})
        return f"?{encoded_settings}"
    
    @staticmethod
    def decode_settings_from_url() -> Tuple[Optional[Union[HalfOnion, SvgProfileOnion]], 
                                          Optional[List[Cut]],
                                            Optional[str]]:
        """
        Decode settings from a URL query string.
        
        Returns:
            Tuple of (onion, cuts, cutting_method)
        """
        query_params = st.query_params
        
        if URL_SETTINGS_KEY not in query_params:
            return None, None, None
        
        try:
            settings = json.loads(query_params[URL_SETTINGS_KEY])
            mode = settings.get('mode', '2d')
            cutting_method = settings.get('cutting_method')
            
            if mode == '2d':
                # Decode 2D settings
                onion = HalfOnion(
                    settings.get('diameter', 5.0),
                    settings.get('n_layers', 9),
                    settings.get('start_y', 0.5),
                    settings.get('end_y', 0.8),
                    settings.get('p1', (0.2, 1.2)),
                    settings.get('p2', (0.8, 1.0))
                )
                
                cuts = [Cut(start, end) for start, end in settings.get('cuts', [])]
            
            
            elif mode == '3d':
                # Decode 3D settings - only SVG profile onions are supported
                try:
                    onion = SvgProfileOnion(
                        diameter=settings.get('diameter', 5.0),
                        svg_file_path=settings.get('svg_file_path', 'assets/onion1.svg')
                    )
                except Exception as e:
                    # If there's an error loading the SVG, don't fall back
                    raise PersistenceError(f"Error loading SVG profile onion: {str(e)}")
                
                cuts = [Cut(start, end) for start, end in settings.get('cuts', [])]
                
            
            else:
                raise PersistenceError(f"Unknown mode: {mode}")
            
            return onion, cuts, cutting_method
        
        except json.JSONDecodeError as e:
            raise PersistenceError(f"Invalid settings JSON: {str(e)}")
        
        except Exception as e:
            raise PersistenceError(f"Error decoding settings: {str(e)}")
    
    @staticmethod
    def update_url(onion: Union[HalfOnion, SvgProfileOnion], 
                 cuts: List[Cut], 
                 cutting_method: str) -> None:
        """
        Update the URL with the current settings.
        
        Args:
            onion: The onion model
            cuts: List of cuts
            cutting_method: The cutting method used
        """
        if isinstance(onion, SvgProfileOnion):
            url = PersistenceService.encode_settings_to_url_3d(onion, cuts, cutting_method)
        else:
            url = PersistenceService.encode_settings_to_url_2d(onion, cuts, cutting_method)
        
        st.query_params.update(urllib.parse.parse_qs(url[1:]))
    
    @staticmethod
    def initialize_session_state() -> None:
        """Initialize the session state with default values."""
        if 'onion_2d' not in st.session_state:
            st.session_state.onion_2d = None
        
        if 'onion_3d' not in st.session_state:
            st.session_state.onion_3d = None
        
        if 'cuts' not in st.session_state:
            st.session_state.cuts = None
        
        if 'cutting_method' not in st.session_state:
            st.session_state.cutting_method = None
        
        if 'root_end_offset' not in st.session_state:
            st.session_state.root_end_offset = 0.1
            
        if 'onion_type' not in st.session_state:
            st.session_state.onion_type = 'svg'
            
        if 'svg_file_path' not in st.session_state:
            st.session_state.svg_file_path = 'assets/onion1.svg'

        if 'update_triggered' not in st.session_state:
            st.session_state.update_triggered = False 