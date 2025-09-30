"""
Base Metric Interface - Abstract base class for all metrics
"""

from abc import ABC


class Metric(ABC):
    """
    Abstract base class for stability metrics.
    
    Each metric follows Single Responsibility Principle:
    - One metric = one class
    - One clear calculation method
    - Clean, testable interface
    
    Note: No abstract calculate() method - subclasses define their own
    signatures based on their specific needs (duck typing).
    """
    
    def __init__(self, name: str):
        """Initialize metric with a name."""
        self.name = name
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
