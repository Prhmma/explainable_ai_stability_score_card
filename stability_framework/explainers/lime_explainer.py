"""
LIME Explainer - Local Interpretable Model-agnostic Explanations
"""

import logging
import re
from typing import Any, Dict, Optional

from .base import BaseExplainer
from .config import LIME_AVAILABLE

if LIME_AVAILABLE:
    from lime.lime_text import LimeTextExplainer

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """LIME explanation method."""

    stochastic = True
    
    def __init__(self, random_seed: int = 42, num_samples: int = 1000):
        super().__init__("LIME", random_seed, num_samples=num_samples)
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for LIMEExplainer")
    
    def explain(self, model: Any, tokenizer: Any, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Generate LIME explanation."""
        inputs, predicted_class, confidence = self._prepare_inputs(model, tokenizer, text)
        target = target_class if target_class is not None else predicted_class
        
        predict_fn = self._create_prediction_function(model, tokenizer)

        # A fresh explainer per call, seeded with this explainer's seed, so each
        # explanation depends only on (text, seed) and not on how many texts were
        # explained before it. Seed variation then compares seed 42 vs 43-46 exactly.
        explainer = LimeTextExplainer(random_state=self.random_seed)
        # LIME's default bag-of-words features are the \w+ runs of the text;
        # request all of them so no word is silently assigned zero.
        lime_words = re.findall(r"\w+", text)
        explanation = explainer.explain_instance(
            text, predict_fn, num_features=max(1, len(set(lime_words))),
            num_samples=self.params['num_samples'], labels=[target]
        )
        weights = dict(explanation.as_list(label=target))

        # Map LIME's word weights back onto every occurrence of each word,
        # then onto the shared whitespace-word segmentation.
        offsets = [m.span() for m in re.finditer(r"\w+", text)]
        scores = [weights.get(m.group(), 0.0) for m in re.finditer(r"\w+", text)]
        words = self._aggregate_to_words(text, offsets, scores)
        word_attributions = self._key_attributions(words)
        
        return {
            'method': self.name, 'text': text,
            'predicted_class': predicted_class, 'target_class': target,
            'confidence': confidence, 'attributions': word_attributions, 'words': words,
            'parameters': {'num_samples': self.params['num_samples'], 'random_seed': self.random_seed}
        }
