"""
Stability Metrics Calculator - Facade for all metrics

Simple, clean interface maintaining backward compatibility.
"""

import logging
from typing import Dict, List, Any
from .jaccard import JaccardSimilarity
from .spearman import SpearmanCorrelation
from .flip_rate import FlipRate
from .attribution_diff import AttributionDifference
from .aggregator import MetricsAggregator

logger = logging.getLogger(__name__)


class StabilityMetrics:
    """
    Facade for calculating stability metrics.
    
    Provides backward-compatible API while delegating to individual metric classes.
    """
    
    def __init__(self):
        """Initialize all metric calculators."""
        # Individual metrics (can be configured if needed)
        self._jaccard_calculators = {}  # Cache for different K values
        self._spearman = SpearmanCorrelation()
        self._flip_rate = FlipRate()
        self._attr_diff = AttributionDifference()
        self._aggregator = MetricsAggregator()
    
    def jaccard_similarity_at_k(
        self,
        original_attrs: Dict[str, float],
        perturbed_attrs: Dict[str, float],
        k: int = 5
    ) -> float:
        """
        Calculate Jaccard@K similarity.
        
        Args:
            original_attrs: Original attribution scores
            perturbed_attrs: Perturbed attribution scores
            k: Number of top tokens
            
        Returns:
            Jaccard similarity (0.0 to 1.0)
        """
        # Cache calculators for different K values
        if k not in self._jaccard_calculators:
            self._jaccard_calculators[k] = JaccardSimilarity(k=k)
        
        return self._jaccard_calculators[k].calculate(original_attrs, perturbed_attrs)
    
    def spearman_rank_correlation(
        self,
        original_attrs: Dict[str, float],
        perturbed_attrs: Dict[str, float]
    ) -> float:
        """
        Calculate Spearman rank correlation.
        
        Args:
            original_attrs: Original attribution scores
            perturbed_attrs: Perturbed attribution scores
            
        Returns:
            Spearman correlation (-1.0 to 1.0)
        """
        return self._spearman.calculate(original_attrs, perturbed_attrs)
    
    def flip_rate(
        self,
        explanations_data: List[Dict[str, Any]],
        confidence_threshold: float = 0.1,
        jaccard_threshold: float = 0.5,
        spearman_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Calculate flip-rate.
        
        Args:
            explanations_data: List of explanation dictionaries
            confidence_threshold: Max confidence change
            jaccard_threshold: Min Jaccard for stable explanation
            spearman_threshold: Min Spearman for stable explanation
            
        Returns:
            Dictionary with flip-rate statistics
        """
        # Create flip-rate calculator with specified thresholds
        flip_rate_calc = FlipRate(
            confidence_threshold=confidence_threshold,
            jaccard_threshold=jaccard_threshold,
            spearman_threshold=spearman_threshold
        )
        
        return flip_rate_calc.calculate(explanations_data)
    
    def attribution_difference(
        self,
        original_attrs: Dict[str, float],
        perturbed_attrs: Dict[str, float],
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate attribution difference.
        
        Args:
            original_attrs: Original attribution scores
            perturbed_attrs: Perturbed attribution scores
            normalize: Whether to normalize by max attribution
            
        Returns:
            Dictionary with difference metrics
        """
        # Create calculator with normalize setting
        attr_diff_calc = AttributionDifference(normalize=normalize)
        
        return attr_diff_calc.calculate(original_attrs, perturbed_attrs)
    
    def calculate_all_metrics(
        self,
        original_exp: Dict[str, Any],
        variant_exp: Dict[str, Any],
        jaccard_k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate all stability metrics.
        
        Args:
            original_exp: Original explanation dictionary
            variant_exp: Variant explanation dictionary
            jaccard_k_values: List of K values for Jaccard
            
        Returns:
            Dictionary with all metrics
        """
        if jaccard_k_values is None:
            jaccard_k_values = [3, 5, 10]
        
        original_attrs = original_exp['attributions']
        variant_attrs = variant_exp['attributions']
        
        metrics = {}
        
        # Jaccard at different K values
        for k in jaccard_k_values:
            metrics[f'jaccard_at_{k}'] = self.jaccard_similarity_at_k(
                original_attrs, variant_attrs, k
            )
        
        # Spearman correlation
        metrics['spearman_correlation'] = self.spearman_rank_correlation(
            original_attrs, variant_attrs
        )
        
        # Attribution difference
        attr_diff = self.attribution_difference(original_attrs, variant_attrs)
        metrics.update({
            f'attr_diff_{key}': value
            for key, value in attr_diff.items()
        })
        
        # Add comparison metadata
        metrics['comparison_info'] = {
            'original_confidence': original_exp.get('confidence', 0.0),
            'variant_confidence': variant_exp.get('confidence', 0.0),
            'confidence_change': abs(
                original_exp.get('confidence', 0.0) -
                variant_exp.get('confidence', 0.0)
            ),
            'same_prediction': (
                original_exp.get('predicted_class') ==
                variant_exp.get('predicted_class')
            )
        }
        
        return metrics
    
    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple comparisons.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated statistics
        """
        return self._aggregator.aggregate(metrics_list)
    
    def format_metric_summary(
        self,
        aggregated_metrics: Dict[str, Any]
    ) -> str:
        """
        Format aggregated metrics summary.
        
        Args:
            aggregated_metrics: Output from aggregate_metrics()
            
        Returns:
            Formatted string
        """
        return self._aggregator.format_summary(aggregated_metrics)
