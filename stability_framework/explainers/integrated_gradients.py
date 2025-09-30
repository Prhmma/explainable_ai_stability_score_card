"""
Integrated Gradients Explainer - Captum-based gradient attribution
"""

import torch
import numpy as np
from typing import Any, Dict, Optional
import logging

from .base import BaseExplainer
from .config import CAPTUM_AVAILABLE

if CAPTUM_AVAILABLE:
    from captum.attr import IntegratedGradients

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients explanation method using Captum.
    """
    
    def __init__(self, random_seed: int = 42, n_steps: int = 50):
        super().__init__("IntegratedGradients", random_seed)
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
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get prediction
        predicted_class, confidence = self._get_prediction(model, inputs)
        target = target_class if target_class is not None else predicted_class
        
        # Create baseline (zero embeddings)
        baseline_ids = torch.zeros_like(inputs['input_ids'])
        
        # Setup Integrated Gradients - target the embeddings layer
        if hasattr(model, 'distilbert') and hasattr(model.distilbert, 'embeddings'):
            embeddings_layer = model.distilbert.embeddings.word_embeddings
        elif hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
            embeddings_layer = model.bert.embeddings.word_embeddings
        elif hasattr(model, 'roberta') and hasattr(model.roberta, 'embeddings'):
            embeddings_layer = model.roberta.embeddings.word_embeddings
        else:
            # Fallback: try to find any embeddings layer
            embeddings_layer = None
            for module in model.modules():
                if hasattr(module, 'weight') and 'embed' in str(type(module)).lower():
                    embeddings_layer = module
                    break
            if embeddings_layer is None:
                raise ValueError("Could not find embeddings layer in model")
        
        ig = IntegratedGradients(embeddings_layer)
        
        # Generate attributions (ensure gradients are computed)
        model.train()  # Enable gradients
        try:
            # Get embeddings for the input
            input_embeddings = embeddings_layer(inputs['input_ids'])
            baseline_embeddings = embeddings_layer(baseline_ids)
            
            # Create a simplified forward function that takes embeddings
            def embedding_forward_func(embeddings):
                # Replace the embeddings in the model forward pass
                if hasattr(model, 'distilbert'):
                    # For DistilBERT, we need to manually construct the forward pass
                    attention_mask = inputs.get('attention_mask')
                    outputs = model.distilbert(inputs_embeds=embeddings, attention_mask=attention_mask)
                    return model.classifier(outputs.last_hidden_state[:, 0])
                else:
                    # For other models, this might need adjustment
                    outputs = model(inputs_embeds=embeddings, attention_mask=inputs.get('attention_mask'))
                    return outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Temporarily replace the forward function
            original_ig = ig
            ig = IntegratedGradients(embedding_forward_func)
            
            attributions = ig.attribute(
                input_embeddings,
                baselines=baseline_embeddings,
                target=target,
                n_steps=self.n_steps,
                return_convergence_delta=False,
                internal_batch_size=1
            )
        finally:
            model.eval()  # Return to eval mode
        
        # Convert to numpy and process
        # Sum over the embedding dimension to get per-token attributions
        attr_scores = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        
        # Aggregate to word level
        word_attributions = self._aggregate_subword_attributions(
            attr_scores, tokens, inputs['input_ids'].squeeze().tolist(), tokenizer
        )
        
        return {
            'method': self.name,
            'text': text,
            'predicted_class': predicted_class,
            'target_class': target,
            'confidence': confidence,
            'attributions': word_attributions,
            'token_attributions': dict(zip(tokens, attr_scores.tolist())),
            'parameters': {'n_steps': self.n_steps, 'random_seed': self.random_seed}
        }
    
    def _forward_func(self, input_ids: torch.Tensor, model: Any, attention_mask: Optional[torch.Tensor] = None):
        """Forward function for Integrated Gradients."""
        # Ensure input_ids are Long tensors (required for embeddings)
        input_ids = input_ids.long()
        
        inputs = {'input_ids': input_ids}
        if attention_mask is not None:
            # Ensure attention_mask is also proper type
            inputs['attention_mask'] = attention_mask.long() if attention_mask.dtype != torch.long else attention_mask
            
        outputs = model(**inputs)
        return outputs.logits if hasattr(outputs, 'logits') else outputs[0]
