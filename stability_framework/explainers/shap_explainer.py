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
        # Locate each Text-masker token in the input, in order. Tokens are slices
        # of the input, but SHAP may drop leading/trailing whitespace, so match
        # on each token's non-whitespace content rather than on cumulative lengths.
        tokens = [str(t) for t in shap_values.data[0]]
        offsets, pos = [], 0
        for token in tokens:
            content = token.strip()
            if not content:
                # Empty strings are special tokens; whitespace tokens sit at the current position
                offsets.append((pos, pos))
                continue
            start = text.find(content, pos)
            if start < 0:
                raise ValueError(f"SHAP token {token!r} not found in input text; cannot align to words")
            offsets.append((start, start + len(content)))
            pos = start + len(content)
        special = [token == "" for token in tokens]
        words = self._aggregate_to_words(text, offsets, attributions.tolist(), special)
        word_attributions = self._key_attributions(words)
        
        return {
            'method': self.name, 'text': text,
            'predicted_class': predicted_class, 'target_class': target,
            'confidence': confidence, 'attributions': word_attributions, 'words': words,
            'parameters': {'random_seed': self.random_seed}
        }
