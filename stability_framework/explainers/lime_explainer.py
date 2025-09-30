"""
LIME Explainer - Local Interpretable Model-agnostic Explanations
"""

from typing import Any, Dict, Optional
import logging

from .base import BaseExplainer
from .config import LIME_AVAILABLE

if LIME_AVAILABLE:
    from lime.lime_text import LimeTextExplainer

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """LIME explanation method."""
    
    def __init__(self, random_seed: int = 42, num_samples: int = 1000):
        super().__init__("LIME", random_seed, num_samples=num_samples)
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for LIMEExplainer")
        self.explainer = LimeTextExplainer(class_names=['negative', 'positive'], random_state=random_seed)
    
    def explain(self, model: Any, tokenizer: Any, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Generate LIME explanation."""
        inputs, predicted_class, confidence = self._prepare_inputs(model, tokenizer, text)
        target = target_class if target_class is not None else predicted_class
        
        predict_fn = self._create_prediction_function(model, tokenizer)
        explanation = self.explainer.explain_instance(
            text, predict_fn, num_features=len(text.split()), 
            num_samples=self.params['num_samples'], labels=[target]
        )
        
        word_attributions = {word: score for word, score in explanation.as_list(label=target)}
        
        return {
            'method': self.name, 'text': text,
            'predicted_class': predicted_class, 'target_class': target,
            'confidence': confidence, 'attributions': word_attributions,
            'parameters': {'num_samples': self.params['num_samples'], 'random_seed': self.random_seed}
        }
