"""
Attribution Difference Metric - Mean absolute difference in attributions
"""

import numpy as np
import logging
from typing import Dict
from .base import Metric

logger = logging.getLogger(__name__)


class AttributionDifference(Metric):
    """
    Attribution difference: measures mean absolute difference between attribution scores.
    
    Provides both absolute and relative (normalized) differences.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize attribution difference metric.
        
        Args:
            normalize: Whether to normalize by maximum attribution value
        """
        super().__init__("Attribution Difference")
        self.normalize = normalize
    
    def calculate(
        self,
        original_attrs: Dict[str, float],
        perturbed_attrs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate mean absolute difference between attributions.
        
        Args:
            original_attrs: Original attribution scores {token: score}
            perturbed_attrs: Perturbed attribution scores {token: score}
            
        Returns:
            Dictionary with difference metrics
        """
        # Find common tokens
        common_tokens = set(original_attrs.keys()).intersection(perturbed_attrs.keys())
        
        if not common_tokens:
            return {
                'mean_absolute_difference': 0.0,
                'relative_difference_percentage': 0.0,
                'max_difference': 0.0,
                'common_tokens': 0
            }
        
        # Calculate differences
        differences = []
        original_scores = []
        perturbed_scores = []
        
        for token in common_tokens:
            orig_score = original_attrs[token]
            pert_score = perturbed_attrs[token]
            differences.append(abs(orig_score - pert_score))
            original_scores.append(orig_score)
            perturbed_scores.append(pert_score)
        
        mean_abs_diff = np.mean(differences)
        max_diff = max(differences)
        
        # Calculate relative difference (normalized)
        relative_diff_pct = 0.0
        if self.normalize:
            max_attr = max(
                max(abs(s) for s in original_scores) if original_scores else 0,
                max(abs(s) for s in perturbed_scores) if perturbed_scores else 0
            )
            if max_attr > 0:
                relative_diff_pct = (mean_abs_diff / max_attr) * 100
        
        result = {
            'mean_absolute_difference': mean_abs_diff,
            'relative_difference_percentage': relative_diff_pct,
            'max_difference': max_diff,
            'common_tokens': len(common_tokens)
        }
        
        logger.debug(
            "%s: mean=%.4f, relative=%.1f%%",
            self.name, mean_abs_diff, relative_diff_pct
        )
        
        return result
