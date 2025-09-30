"""
Metrics Aggregator - Statistical aggregation across multiple comparisons
"""

import numpy as np
import logging
from typing import List, Dict, Any
from scipy import stats

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """
    Aggregates metrics across multiple comparisons.
    
    Provides: mean, std, min, max, median, confidence intervals
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize aggregator.
        
        Args:
            confidence_level: Confidence level for intervals (default 95%)
        """
        self.confidence_level = confidence_level
    
    def aggregate(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple comparisons.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Dictionary with aggregated statistics
        """
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Identify all numeric metrics
        numeric_keys = self._find_numeric_keys(metrics_list)
        
        # Calculate statistics for each metric
        for key in numeric_keys:
            values = self._extract_values(metrics_list, key)
            
            if values:
                aggregated[key] = self._calculate_stats(values)
        
        # Add summary
        aggregated['summary'] = {
            'total_comparisons': len(metrics_list),
            'metrics_computed': len(numeric_keys)
        }
        
        return aggregated
    
    def _find_numeric_keys(self, metrics_list: List[Dict[str, Any]]) -> set:
        """Find all numeric metric keys."""
        numeric_keys = set()
        
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    numeric_keys.add(key)
        
        return numeric_keys
    
    def _extract_values(
        self,
        metrics_list: List[Dict[str, Any]],
        key: str
    ) -> List[float]:
        """Extract non-NaN values for a specific metric key."""
        values = []
        
        for metrics in metrics_list:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    values.append(float(value))
        
        return values
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, Any]:
        """Calculate statistics for a list of values."""
        values_array = np.array(values)
        
        stats_dict = {
            'mean': np.mean(values_array),
            'std': np.std(values_array),
            'min': np.min(values_array),
            'max': np.max(values_array),
            'median': np.median(values_array),
            'count': len(values)
        }
        
        # Calculate confidence interval
        if len(values) > 1:
            ci = stats.t.interval(
                self.confidence_level,
                len(values) - 1,
                loc=np.mean(values_array),
                scale=stats.sem(values_array)
            )
            stats_dict['ci_95'] = ci
        else:
            stats_dict['ci_95'] = (values_array[0], values_array[0])
        
        return stats_dict
    
    def format_summary(self, aggregated_metrics: Dict[str, Any]) -> str:
        """
        Format aggregated metrics into readable summary.
        
        Args:
            aggregated_metrics: Output from aggregate()
            
        Returns:
            Formatted string summary
        """
        if not aggregated_metrics:
            return "No metrics available"
        
        lines = ["=== STABILITY METRICS SUMMARY ==="]
        
        # Key metrics with display names
        key_metrics = [
            ('jaccard_at_5', 'Jaccard@5'),
            ('jaccard_at_3', 'Jaccard@3'),
            ('jaccard_at_10', 'Jaccard@10'),
            ('spearman_correlation', 'Spearman Correlation'),
            ('attr_diff_mean_absolute_difference', 'Attribution Difference'),
            ('attr_diff_relative_difference_percentage', 'Relative Diff %')
        ]
        
        for metric_key, display_name in key_metrics:
            if metric_key in aggregated_metrics:
                stats_dict = aggregated_metrics[metric_key]
                lines.append(
                    f"{display_name}: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f} "
                    f"(CI: {stats_dict['ci_95'][0]:.3f}, {stats_dict['ci_95'][1]:.3f})"
                )
        
        if 'summary' in aggregated_metrics:
            lines.append(
                f"\nTotal comparisons: {aggregated_metrics['summary']['total_comparisons']}"
            )
        
        return "\n".join(lines)
