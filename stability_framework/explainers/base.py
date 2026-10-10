"""
Base Explainer Class - Shared functionality for all explainers
"""

import bisect
import re
import string

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# Inputs are padded to a multiple of this length (padding is masked out). On MPS,
# every distinct tensor shape compiles and caches a new kernel graph; with free
# input lengths that cache grew by gigabytes over a long run.
PAD_TO_MULTIPLE_OF = 16


class BaseExplainer(ABC):
    """Base class for explanation methods with shared utilities."""

    # Whether explanations depend on a random seed (seed variation applies only then)
    stochastic = False
    
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
    
    def get_params(self) -> Dict[str, Any]:
        """Constructor parameters (excluding the seed), to re-create this explainer with another seed."""
        return dict(self.params)

    def with_seed(self, seed: int) -> "BaseExplainer":
        """A new explainer of the same class and parameters with a different seed."""
        return type(self)(random_seed=seed, **self.get_params())

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
    
    def _create_prediction_function(self, model: Any, tokenizer: Any, batch_size: int = 128):
        """
        Create prediction function for LIME/SHAP - shared utility.

        Texts are scored in chunks of `batch_size`: LIME sends its 1000
        perturbed samples at once, and a single padded batch of that size
        made the MPS allocator cache many large, differently sized buffers
        (an 18 GB footprint on a 24 GB machine).
        """
        def predict_fn(texts: List[str]) -> np.ndarray:
            # SHAP passes a numpy array of strings, which tokenizers reject
            texts = [texts] if isinstance(texts, str) else [str(t) for t in texts]

            chunks = []
            for start in range(0, len(texts), batch_size):
                inputs = tokenizer(
                    texts[start:start + batch_size],
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
                    max_length=512
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())

            return np.concatenate(chunks)

        return predict_fn
    
    def _prepare_inputs(self, model: Any, tokenizer: Any, text: str) -> Tuple[Dict, int, float]:
        """Prepare inputs and get prediction - DRY helper."""
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        predicted_class, confidence = self._get_prediction(model, inputs)
        return inputs, predicted_class, confidence
    
    @staticmethod
    def _aggregate_to_words(
        text: str,
        offsets: List[Tuple[int, int]],
        scores: List[float],
        special: Optional[List[bool]] = None
    ) -> List[Tuple[str, float]]:
        """
        Aggregate token-level scores to the whitespace-delimited words of `text`.

        Every explainer uses this same segmentation, so explanations from
        different tokenizers (WordPiece, BPE, SentencePiece) and methods are
        comparable. Each token is assigned to the word containing the start of
        its character span; a token that starts on whitespace (e.g. a
        SentencePiece word-start marker) belongs to the following word, or to
        the last word if no word follows. Tokens flagged in `special`
        (e.g. [CLS], </s>) are skipped; whitespace-only tokens may have
        zero-width offsets and are still placed by their start position.

        Returns:
            Ordered list of (word, summed score), one entry per word occurrence.
        """
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if not spans:
            return []
        starts = [a for a, _ in spans]
        totals = [0.0] * len(spans)
        covered = [False] * len(spans)

        if special is None:
            special = [False] * len(offsets)
        for (a, b), score, is_special in zip(offsets, scores, special):
            if is_special:
                continue
            idx = bisect.bisect_right(starts, a) - 1
            if idx < 0 or a >= spans[idx][1]:
                idx += 1
            # Trailing whitespace after the last word: keep its score with that word
            idx = min(idx, len(spans) - 1)
            totals[idx] += float(score)
            covered[idx] = True

        return [(text[a:b], totals[i]) for i, (a, b) in enumerate(spans) if covered[i]]

    @staticmethod
    def _normalize_word(word: str) -> str:
        """Case- and edge-punctuation-insensitive identity of a word ("Film," -> "film")."""
        lowered = word.lower()
        return lowered.strip(string.punctuation) or lowered

    @classmethod
    def _key_attributions(cls, word_scores: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Key word scores by normalized word, keeping repeated words distinct.

        The n-th occurrence of a word is keyed "word #n" (the first is just
        "word"), so repeated words no longer overwrite each other and the same
        word occurrence matches across preprocessing variants (e.g. "Film" in
        the original and "FILM" in an uppercased variant). Words never contain
        spaces, so these keys cannot collide with a literal word such as "#2".
        """
        counts: Dict[str, int] = {}
        keyed: Dict[str, float] = {}
        for word, score in word_scores:
            norm = cls._normalize_word(word)
            counts[norm] = counts.get(norm, 0) + 1
            key = norm if counts[norm] == 1 else f"{norm} #{counts[norm]}"
            keyed[key] = float(score)
        return keyed
