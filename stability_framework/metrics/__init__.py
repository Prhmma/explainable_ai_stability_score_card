"""
Metrics Module - Clean OOP implementation for stability assessment

Each metric is a separate class following Single Responsibility Principle.
"""

from .base import Metric
from .jaccard import JaccardSimilarity
from .spearman import SpearmanCorrelation
from .flip_rate import FlipRate
from .attribution_diff import AttributionDifference
from .aggregator import MetricsAggregator
from .calculator import StabilityMetrics

__all__ = [
    'Metric',
    'JaccardSimilarity',
    'SpearmanCorrelation',
    'FlipRate',
    'AttributionDifference',
    'MetricsAggregator',
    'StabilityMetrics'
]
