"""
Flip Rate Metric - Prediction stability vs explanation stability
"""

import logging
from typing import List, Dict, Any
from .base import Metric
from .jaccard import JaccardSimilarity
from .spearman import SpearmanCorrelation

logger = logging.getLogger(__name__)


class FlipRate(Metric):
    """
    Flip-rate: percentage of cases where prediction is stable but explanation changes.
    
    "Flip" = Same prediction + Similar confidence, BUT different explanation
    High flip-rate = explanations are unreliable/unstable
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.1,
        jaccard_threshold: float = 0.5,
        spearman_threshold: float = 0.7
    ):
        """
        Initialize flip-rate metric.
        
        Args:
            confidence_threshold: Maximum allowed confidence change (±)
            jaccard_threshold: Minimum Jaccard@5 for stable explanation
            spearman_threshold: Minimum Spearman for stable explanation
        """
        super().__init__("Flip Rate")
        self.confidence_threshold = confidence_threshold
        self.jaccard_threshold = jaccard_threshold
        self.spearman_threshold = spearman_threshold
        
        # Initialize sub-metrics
        self.jaccard_metric = JaccardSimilarity(k=5)
        self.spearman_metric = SpearmanCorrelation()
    
    def calculate(
        self,
        explanations_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate flip-rate across multiple explanations.
        
        Args:
            explanations_data: List of {original: exp, variants: [exp1, exp2, ...]}
            
        Returns:
            Dictionary with flip-rate statistics
        """
        total_comparisons = 0
        flip_cases = 0
        flip_details = []
        
        for data in explanations_data:
            if 'original' not in data or 'variants' not in data:
                continue
            
            original_exp = data['original']
            
            for variant_exp in data['variants']:
                total_comparisons += 1
                
                # Check prediction stability
                pred_stable = self._is_prediction_stable(original_exp, variant_exp)
                
                if pred_stable:
                    # Check explanation stability
                    explanation_unstable = self._is_explanation_unstable(
                        original_exp, variant_exp
                    )
                    
                    if explanation_unstable:
                        flip_cases += 1
                        flip_details.append(self._create_flip_detail(
                            original_exp, variant_exp
                        ))
        
        flip_rate_pct = (
            (flip_cases / total_comparisons * 100)
            if total_comparisons > 0 else 0.0
        )
        
        result = {
            'flip_rate_percentage': flip_rate_pct,
            'flip_cases': flip_cases,
            'total_comparisons': total_comparisons,
            'thresholds': {
                'confidence': self.confidence_threshold,
                'jaccard': self.jaccard_threshold,
                'spearman': self.spearman_threshold
            },
            'examples': flip_details[:5]  # Top 5 examples
        }
        
        logger.info(
            "%s: %.1f%% (%d/%d)",
            self.name, flip_rate_pct, flip_cases, total_comparisons
        )
        
        return result
    
    def _is_prediction_stable(
        self,
        original_exp: Dict[str, Any],
        variant_exp: Dict[str, Any]
    ) -> bool:
        """Check if prediction is stable (same class, similar confidence)."""
        same_class = (
            original_exp['predicted_class'] == variant_exp['predicted_class']
        )
        
        confidence_diff = abs(
            original_exp['confidence'] - variant_exp['confidence']
        )
        similar_confidence = confidence_diff <= self.confidence_threshold
        
        return same_class and similar_confidence
    
    def _is_explanation_unstable(
        self,
        original_exp: Dict[str, Any],
        variant_exp: Dict[str, Any]
    ) -> bool:
        """Check if explanation changed significantly."""
        jaccard_5 = self.jaccard_metric.calculate(
            original_exp['attributions'],
            variant_exp['attributions']
        )
        
        spearman_corr = self.spearman_metric.calculate(
            original_exp['attributions'],
            variant_exp['attributions']
        )
        
        return (
            jaccard_5 < self.jaccard_threshold or
            spearman_corr < self.spearman_threshold
        )
    
    def _create_flip_detail(
        self,
        original_exp: Dict[str, Any],
        variant_exp: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create detailed flip case record."""
        jaccard_5 = self.jaccard_metric.calculate(
            original_exp['attributions'],
            variant_exp['attributions']
        )
        
        spearman_corr = self.spearman_metric.calculate(
            original_exp['attributions'],
            variant_exp['attributions']
        )
        
        return {
            'text': original_exp.get('text', 'N/A')[:50] + '...',
            'jaccard_5': jaccard_5,
            'spearman': spearman_corr,
            'confidence_diff': abs(
                original_exp['confidence'] - variant_exp['confidence']
            )
        }
