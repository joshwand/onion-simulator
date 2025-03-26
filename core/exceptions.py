"""
Custom exceptions for the onion simulator.
"""


class OnionSimulatorError(Exception):
    """Base exception class for all onion simulator errors."""
    pass


class ModelError(OnionSimulatorError):
    """Raised when there's an error with the onion model."""
    pass


class CutError(OnionSimulatorError):
    """Raised when there's an error with a cut."""
    pass


class InvalidParameterError(OnionSimulatorError):
    """Raised when an invalid parameter is provided."""
    pass


class GeometryError(OnionSimulatorError):
    """Raised when there's an error with geometric calculations."""
    pass


class VisualizationError(OnionSimulatorError):
    """Raised when there's an error with visualization."""
    pass


class PersistenceError(OnionSimulatorError):
    """Raised when there's an error with persistence (URL or state)."""
    pass 