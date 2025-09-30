"""
Spearman Rank Correlation Metric - Ranking consistency
"""

import numpy as np
import logging
import warnings
from scipy.stats import spearmanr
from typing import Dict
from .base import Metric

logger = logging.getLogger(__name__)


class SpearmanCorrelation(Metric):
    """
    Spearman rank correlation: measures ranking consistency between attributions.
    
    Returns: -1.0 (perfect negative correlation) to 1.0 (perfect positive correlation)
    """
    
    def __init__(self):
        super().__init__("Spearman Correlation")
    
    def calculate(
        self,
        original_attrs: Dict[str, float],
        perturbed_attrs: Dict[str, float]
    ) -> float:
        """
        Calculate Spearman rank correlation.
        
        Args:
            original_attrs: Original attribution scores {token: score}
            perturbed_attrs: Perturbed attribution scores {token: score}
            
        Returns:
            Spearman correlation coefficient (-1.0 to 1.0)
        """
        # Find common tokens
        common_tokens = set(original_attrs.keys()).intersection(perturbed_attrs.keys())
        
        if len(common_tokens) < 2:
            logger.warning("Less than 2 common tokens for Spearman correlation")
            return 0.0
        
        # Extract scores for common tokens
        original_scores = [original_attrs[token] for token in common_tokens]
        perturbed_scores = [perturbed_attrs[token] for token in common_tokens]
        
        # Calculate Spearman correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about constant arrays
            correlation, p_value = spearmanr(original_scores, perturbed_scores)
        
        # Handle NaN (e.g., when one array is constant)
        if np.isnan(correlation):
            correlation = 0.0
        
        logger.debug("%s: %.3f (p=%.3f)", self.name, correlation, p_value)
        return correlation
