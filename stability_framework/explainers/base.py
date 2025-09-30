"""
Base Explainer Class - Shared functionality for all explainers
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseExplainer(ABC):
    """Base class for explanation methods with shared utilities."""
    
    def __init__(self, name: str, random_seed: int = 42, **params):
        self.name = name
        self.random_seed = random_seed
        self.params = params
        self._setup_random_seed(random_seed)
        
    def _setup_random_seed(self, seed: int):
        """Setup random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    @abstractmethod
    def explain(
        self, 
        model: Any, 
        tokenizer: Any, 
        text: str, 
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate explanation for given text."""
        pass
    
    def _get_prediction(self, model: Any, inputs: Dict[str, torch.Tensor]) -> Tuple[int, float]:
        """Get model prediction and confidence - shared across all explainers."""
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
        return predicted_class, confidence
    
    def _create_prediction_function(self, model: Any, tokenizer: Any):
        """Create prediction function for LIME/SHAP - shared utility."""
        def predict_fn(texts: List[str]) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]
            
            inputs = tokenizer(
                texts, 
                return_tensors='pt', 
                truncation=True, 
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                probabilities = torch.softmax(logits, dim=-1)
            
            return probabilities.cpu().numpy()
        
        return predict_fn
    
    def _prepare_inputs(self, model: Any, tokenizer: Any, text: str) -> Tuple[Dict, int, float]:
        """Prepare inputs and get prediction - DRY helper."""
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        predicted_class, confidence = self._get_prediction(model, inputs)
        return inputs, predicted_class, confidence
    
    def _aggregate_subword_attributions(
        self, 
        attributions: np.ndarray, 
        tokens: List[str],
        input_ids: List[int],
        tokenizer: Any
    ) -> Dict[str, float]:
        """
        Aggregate subword-level attributions to word-level.
        
        Args:
            attributions: Subword-level attribution scores
            tokens: List of subword tokens
            input_ids: List of token IDs
            tokenizer: Tokenizer used
            
        Returns:
            Dictionary mapping words to aggregated attribution scores
        """
        word_attributions = {}
        current_word = ""
        current_attribution = 0.0
        
        for i, (token, attr) in enumerate(zip(tokens, attributions)):
            # Skip special tokens
            if token in tokenizer.special_tokens_map.values():
                continue
                
            # Handle different tokenizer conventions
            if token.startswith('##') or token.startswith('▁'):  # BERT/ALBERT style
                # Continuation of previous word
                current_word += token[2:] if token.startswith('##') else token[1:]
                current_attribution += attr
            elif token.startswith('Ġ'):  # GPT/RoBERTa style
                # New word
                if current_word:
                    word_attributions[current_word] = current_attribution
                current_word = token[1:]
                current_attribution = attr
            else:
                # New word or single character
                if current_word:
                    word_attributions[current_word] = current_attribution
                current_word = token
                current_attribution = attr
        
        # Add the last word
        if current_word:
            word_attributions[current_word] = current_attribution
            
        return word_attributions
