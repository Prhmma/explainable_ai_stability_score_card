"""
SHAP Explainer - SHapley Additive exPlanations
"""

from typing import Any, Dict, Optional
import logging

from .base import BaseExplainer
from .config import SHAP_AVAILABLE

if SHAP_AVAILABLE:
    import shap

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """SHAP explanation method."""
    
    def __init__(self, random_seed: int = 42):
        super().__init__("SHAP", random_seed)
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAPExplainer")
    
    def explain(self, model: Any, tokenizer: Any, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Generate SHAP explanation."""
        inputs, predicted_class, confidence = self._prepare_inputs(model, tokenizer, text)
        target = target_class if target_class is not None else predicted_class
        
        predict_fn = self._create_prediction_function(model, tokenizer)
        explainer = shap.Explainer(predict_fn, shap.maskers.Text(tokenizer))
        shap_values = explainer([text])
        
        # Extract attributions for target class
        attributions = shap_values.values[0, :, target] if len(shap_values.values.shape) > 2 else shap_values.values[0, :]
        tokens = shap_values.data[0] if hasattr(shap_values, 'data') else text.split()
        word_attributions = dict(zip(tokens, attributions.tolist()))
        
        return {
            'method': self.name, 'text': text,
            'predicted_class': predicted_class, 'target_class': target,
            'confidence': confidence, 'attributions': word_attributions,
            'parameters': {'random_seed': self.random_seed}
        }
