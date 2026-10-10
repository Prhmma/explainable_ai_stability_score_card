"""
Legacy explainer variants for the implementation-sensitivity analysis.

- LegacyConfigIG: Integrated Gradients as released (train mode, so dropout is
  active; for DistilBERT the [CLS] state goes straight into the classifier,
  skipping pre_classifier; baseline = embedding of token id 0), with the
  current word alignment. Dropout is seeded per call for reproducibility.
- LegacyAlignment{IG,LIME,SHAP}: the current explainers with the released
  attribution keys: IG merges '##'/'▁' pieces into raw, case-sensitive words
  (repeated words overwrite each other); LIME keys are its own words, capped at
  len(text.split()) features; SHAP keys are its raw tokens, including
  whitespace and empty special tokens.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..explainers.base import BaseExplainer, PAD_TO_MULTIPLE_OF
from ..explainers.config import CAPTUM_AVAILABLE, LIME_AVAILABLE, SHAP_AVAILABLE
from ..explainers.integrated_gradients import IntegratedGradientsExplainer
from ..explainers.lime_explainer import LIMEExplainer
from ..explainers.shap_explainer import SHAPExplainer

if CAPTUM_AVAILABLE:
    from captum.attr import IntegratedGradients
if LIME_AVAILABLE:
    from lime.lime_text import LimeTextExplainer
if SHAP_AVAILABLE:
    import shap


def legacy_aggregate(tokens: List[str], scores: List[float], tokenizer: Any) -> Dict[str, float]:
    """The released subword aggregation (verbatim logic from commit cc9153f)."""
    word_attributions = {}
    current_word = ""
    current_attribution = 0.0
    for token, attr in zip(tokens, scores):
        if token in tokenizer.special_tokens_map.values():
            continue
        if token.startswith('##') or token.startswith('▁'):
            current_word += token[2:] if token.startswith('##') else token[1:]
            current_attribution += attr
        elif token.startswith('Ġ'):
            if current_word:
                word_attributions[current_word] = current_attribution
            current_word = token[1:]
            current_attribution = attr
        else:
            if current_word:
                word_attributions[current_word] = current_attribution
            current_word = token
            current_attribution = attr
    if current_word:
        word_attributions[current_word] = current_attribution
    return word_attributions


class LegacyConfigIG(IntegratedGradientsExplainer):
    """IG with the released execution choices; current alignment."""

    def __init__(self, random_seed: int = 42, n_steps: int = 50):
        super().__init__(random_seed=random_seed, n_steps=n_steps)
        self.name = "IntegratedGradientsLegacyConfig"

    def explain(self, model: Any, tokenizer: Any, text: str, target_class: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512,
                           pad_to_multiple_of=PAD_TO_MULTIPLE_OF, return_offsets_mapping=True)
        offsets = [tuple(o) for o in inputs.pop('offset_mapping')[0].tolist()]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        predicted_class, confidence = self._get_prediction(model, inputs)
        target = target_class if target_class is not None else predicted_class
        attention_mask = inputs.get('attention_mask')

        embeddings_layer = model.get_input_embeddings()
        baseline_ids = torch.zeros_like(inputs['input_ids'])  # token id 0, not a zero vector

        def forward(embeddings):
            if hasattr(model, 'distilbert'):
                outputs = model.distilbert(inputs_embeds=embeddings, attention_mask=attention_mask)
                return model.classifier(outputs.last_hidden_state[:, 0])  # skips pre_classifier + ReLU
            outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
            return outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        torch.manual_seed(self.random_seed)  # dropout is active; seed it per call
        model.train()
        try:
            input_embeddings = embeddings_layer(inputs['input_ids'])
            baseline_embeddings = embeddings_layer(baseline_ids)
            attributions = IntegratedGradients(forward).attribute(
                input_embeddings, baselines=baseline_embeddings, target=target,
                n_steps=self.n_steps, internal_batch_size=1)
        finally:
            model.eval()

        attr_scores = attributions.sum(dim=-1)[0].detach().cpu().numpy()
        input_ids = inputs['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        special_ids = set(tokenizer.all_special_ids)
        words = self._aggregate_to_words(text, offsets, attr_scores.tolist(), [i in special_ids for i in input_ids])
        return {
            'method': self.name, 'text': text, 'predicted_class': predicted_class, 'target_class': target,
            'confidence': confidence, 'attributions': self._key_attributions(words), 'words': words,
            'token_attributions': list(zip(tokens, attr_scores.tolist())),
            'parameters': {'n_steps': self.n_steps, 'baseline': 'token_id_0', 'mode': 'train', 'random_seed': self.random_seed}
        }


class LegacyAlignmentIG(IntegratedGradientsExplainer):
    """Current IG computation; released attribution keys."""

    def __init__(self, random_seed: int = 42, n_steps: int = 50):
        super().__init__(random_seed=random_seed, n_steps=n_steps)
        self.name = "IntegratedGradientsLegacyAlignment"

    def explain(self, model: Any, tokenizer: Any, text: str, target_class: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        result = super().explain(model, tokenizer, text, target_class, **kwargs)
        tokens, scores = zip(*result['token_attributions'])
        result['attributions'] = legacy_aggregate(list(tokens), list(scores), tokenizer)
        result['method'] = self.name
        return result


class LegacyAlignmentLIME(LIMEExplainer):
    """Current LIME computation (seeded per call); released keys and feature cap."""

    def __init__(self, random_seed: int = 42, num_samples: int = 1000):
        super().__init__(random_seed=random_seed, num_samples=num_samples)
        self.name = "LIMELegacyAlignment"

    def explain(self, model: Any, tokenizer: Any, text: str, target_class: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        inputs, predicted_class, confidence = self._prepare_inputs(model, tokenizer, text)
        target = target_class if target_class is not None else predicted_class
        explanation = LimeTextExplainer(random_state=self.random_seed).explain_instance(
            text, self._create_prediction_function(model, tokenizer), num_features=len(text.split()),
            num_samples=self.params['num_samples'], labels=[target])
        return {
            'method': self.name, 'text': text, 'predicted_class': predicted_class, 'target_class': target,
            'confidence': confidence, 'attributions': dict(explanation.as_list(label=target)),
            'parameters': {'num_samples': self.params['num_samples'], 'random_seed': self.random_seed}
        }


class LegacyAlignmentSHAP(SHAPExplainer):
    """Current SHAP computation; released keys (raw tokens)."""

    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed=random_seed)
        self.name = "SHAPLegacyAlignment"

    def explain(self, model: Any, tokenizer: Any, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        inputs, predicted_class, confidence = self._prepare_inputs(model, tokenizer, text)
        target = target_class if target_class is not None else predicted_class
        shap_values = shap.Explainer(self._create_prediction_function(model, tokenizer), shap.maskers.Text(tokenizer))([text])
        values = shap_values.values[0, :, target] if len(shap_values.values.shape) > 2 else shap_values.values[0, :]
        return {
            'method': self.name, 'text': text, 'predicted_class': predicted_class, 'target_class': target,
            'confidence': confidence, 'attributions': dict(zip([str(t) for t in shap_values.data[0]], values.tolist())),
            'parameters': {'random_seed': self.random_seed}
        }
