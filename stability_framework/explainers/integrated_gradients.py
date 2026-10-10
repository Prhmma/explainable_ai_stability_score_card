"""
Integrated Gradients Explainer - Captum-based gradient attribution
"""

import torch
import numpy as np
from typing import Any, Dict, Optional
import logging

from .base import BaseExplainer, PAD_TO_MULTIPLE_OF
from .config import CAPTUM_AVAILABLE

if CAPTUM_AVAILABLE:
    from captum.attr import IntegratedGradients

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients explanation method using Captum.
    """
    
    # Interpolation steps x (padded) sequence length per forward/backward batch.
    # Bounds activation memory on long inputs (all 50 steps of a long AG News
    # input took up to 14.7 GB); inputs of up to 64 tokens still run all 50
    # steps in one batch.
    TOKEN_BUDGET = 3200

    def __init__(self, random_seed: int = 42, n_steps: int = 50):
        super().__init__("IntegratedGradients", random_seed, n_steps=n_steps)
        self.n_steps = n_steps
        
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is required for IntegratedGradients")
    
    def explain(
        self, 
        model: Any, 
        tokenizer: Any, 
        text: str, 
        target_class: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate Integrated Gradients explanation."""
        logger.debug(f"Generating IG explanation for: {text[:50]}...")
        
        # Prepare inputs
        inputs = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True,
            max_length=512,
            pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
            return_offsets_mapping=True
        )
        offsets = [tuple(o) for o in inputs.pop('offset_mapping')[0].tolist()]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get prediction
        predicted_class, confidence = self._get_prediction(model, inputs)
        target = target_class if target_class is not None else predicted_class
        attention_mask = inputs.get('attention_mask')

        # Attribute with respect to the input embeddings, using the model's own
        # forward pass so the explained function is exactly the one that made
        # the prediction. The model stays in eval mode: gradients do not need
        # train mode, and train mode would enable dropout and make IG
        # non-deterministic.
        model.eval()
        embeddings_layer = model.get_input_embeddings()
        with torch.no_grad():
            input_embeddings = embeddings_layer(inputs['input_ids'])

        # Zero-vector baseline in embedding space
        baseline_embeddings = torch.zeros_like(input_embeddings)

        def embedding_forward_func(embeddings, attention_mask):
            outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
            return outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        ig = IntegratedGradients(embedding_forward_func)
        seq_len = input_embeddings.shape[1]
        internal_batch_size = max(1, min(self.n_steps, self.TOKEN_BUDGET // seq_len))
        attributions, convergence_delta = ig.attribute(
            input_embeddings,
            baselines=baseline_embeddings,
            target=target,
            additional_forward_args=(attention_mask,),
            n_steps=self.n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True
        )

        # Convert to numpy and process
        # Sum over the embedding dimension to get per-token attributions
        attr_scores = attributions.sum(dim=-1)[0].detach().cpu().numpy()
        input_ids = inputs['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Aggregate to word level
        special_ids = set(tokenizer.all_special_ids)
        special = [token_id in special_ids for token_id in input_ids]
        words = self._aggregate_to_words(text, offsets, attr_scores.tolist(), special)
        word_attributions = self._key_attributions(words)

        return {
            'method': self.name,
            'text': text,
            'predicted_class': predicted_class,
            'target_class': target,
            'confidence': confidence,
            'attributions': word_attributions,
            'words': words,
            'token_attributions': list(zip(tokens, attr_scores.tolist())),
            'convergence_delta': float(convergence_delta.abs().max().item()),
            'parameters': {'n_steps': self.n_steps, 'baseline': 'zero_embedding', 'random_seed': self.random_seed}
        }
