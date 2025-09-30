"""
Jaccard Similarity Metric - Top-K token overlap
"""

import logging
from typing import Dict, Set
from .base import Metric

logger = logging.getLogger(__name__)


class JaccardSimilarity(Metric):
    """
    Jaccard similarity at K: measures overlap of top-K important tokens.
    
    Formula: |A ∩ B| / |A ∪ B|
    Returns: 0.0 (no overlap) to 1.0 (perfect overlap)
    """
    
    def __init__(self, k: int = 5):
        """
        Initialize Jaccard similarity metric.
        
        Args:
            k: Number of top tokens to consider
        """
        super().__init__(f"Jaccard@{k}")
        self.k = k
    
    def calculate(
        self, 
        original_attrs: Dict[str, float], 
        perturbed_attrs: Dict[str, float]
    ) -> float:
        """
        Calculate Jaccard@K similarity.
        
        Args:
            original_attrs: Original attribution scores {token: score}
            perturbed_attrs: Perturbed attribution scores {token: score}
            
        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        top_k_original = self._get_top_k_tokens(original_attrs)
        top_k_perturbed = self._get_top_k_tokens(perturbed_attrs)
        
        intersection = len(top_k_original.intersection(top_k_perturbed))
        union = len(top_k_original.union(top_k_perturbed))
        
        if union == 0:
            return 1.0  # Both sets empty = perfect agreement
        
        jaccard = intersection / union
        logger.debug(
            "%s: intersection=%d, union=%d, score=%.3f",
            self.name, intersection, union, jaccard
        )
        
        return jaccard
    
    def _get_top_k_tokens(self, attributions: Dict[str, float]) -> Set[str]:
        """
        Get top-K tokens by absolute attribution score.
        
        Args:
            attributions: Attribution scores {token: score}
            
        Returns:
            Set of top-K token names
        """
        if not attributions:
            return set()
        
        # Sort by absolute value
        sorted_tokens = sorted(
            attributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return set(token for token, _ in sorted_tokens[:self.k])
