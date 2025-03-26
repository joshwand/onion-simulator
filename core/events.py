"""
Event system for UI updates.
Implements a simple event bus using the Observer pattern.
"""

from collections import defaultdict
from typing import Callable, Any, Dict, List, Optional


class EventBus:
    """
    A simple event bus that allows components to subscribe to and publish events.
    Implements the Observer pattern.
    """
    _subscribers: Dict[str, List[Callable]] = defaultdict(list)
    
    @classmethod
    def subscribe(cls, event_type: str, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to an event.
        
        Args:
            event_type: The type of event to subscribe to
            callback: A function to call when the event is published
        """
        cls._subscribers[event_type].append(callback)
    
    @classmethod
    def unsubscribe(cls, event_type: str, callback: Callable[[Any], None]) -> None:
        """
        Unsubscribe from an event.
        
        Args:
            event_type: The type of event to unsubscribe from
            callback: The function to remove from the subscribers list
        """
        if event_type in cls._subscribers and callback in cls._subscribers[event_type]:
            cls._subscribers[event_type].remove(callback)
    
    @classmethod
    def publish(cls, event_type: str, data: Optional[Any] = None) -> None:
        """
        Publish an event.
        
        Args:
            event_type: The type of event to publish
            data: Optional data to pass to subscribers
        """
        for callback in cls._subscribers[event_type]:
            callback(data)
    
    @classmethod
    def clear_all(cls) -> None:
        """
        Clear all subscribers.
        Useful for testing or resetting the application state.
        """
        cls._subscribers.clear()


# Define common event types
class EventTypes:
    """Constants for common event types used in the application."""
    ONION_UPDATED = "onion_updated"
    CUTS_UPDATED = "cuts_updated"
    PIECES_UPDATED = "pieces_updated"
    VIEW_REFRESH_NEEDED = "view_refresh_needed"
    METHOD_CHANGED = "method_changed"
    PARAMETERS_CHANGED = "parameters_changed" 