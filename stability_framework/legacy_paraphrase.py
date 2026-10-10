"""
The paraphrase generator released with the IKT 2025 paper (commit cc9153f), kept
verbatim for comparison studies only.

It differs from the current generator (stress_tests.SemanticParaphrasingTest) in
ways that change stability estimates: synonyms come from a Python set, so the
output depends on the process's hash seed (set PYTHONHASHSEED for a reproducible
comparison); any word can be substituted, including function words; synonym
variants fill the four slots before reorderings or transformations are tried;
and reordering moves single words. Do not use it for new studies.
"""

import logging
from typing import List

from .stress_tests import NLTK_AVAILABLE, SemanticParaphrasingTest  # noqa: F401 (used by the verbatim methods)

logger = logging.getLogger(__name__)


class LegacyParaphrasingTest(SemanticParaphrasingTest):
    """Paraphrasing stress test exactly as released with the paper (v1)."""

    def __init__(self, random_seed: int = 42, similarity_threshold: float = 0.8, cache_dir: str = None):
        super().__init__(random_seed=random_seed, similarity_threshold=similarity_threshold, cache_dir=cache_dir)
        self.name = "SemanticParaphrasingV1"

    def _get_cache_key(self, text: str) -> str:
        return "v1_" + super()._get_cache_key(text)

    # --- methods below are copied verbatim from commit cc9153f ---

    def apply(self, text: str) -> List[str]:
        """Generate semantic paraphrases of the text."""
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_paraphrases = self._load_from_cache(cache_key)
        
        if cached_paraphrases is not None:
            logger.debug(f"Using cached paraphrases for text: {text[:50]}...")
            return cached_paraphrases
        
        # Generate paraphrases if not cached
        logger.debug(f"Generating paraphrases for text: {text[:50]}...")
        variants = []
        
        # Original text
        variants.append(text)
        
        # Generate different types of paraphrases
        variants.extend(self._generate_synonym_substitutions(text))
        variants.extend(self._generate_reorderings(text))
        variants.extend(self._generate_simple_transformations(text))
        
        # Filter variants by semantic similarity if model available
        if self.similarity_model is not None:
            variants = self._filter_by_similarity(text, variants)
        
        # Remove duplicates and limit
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen and variant != text:
                seen.add(variant)
                unique_variants.append(variant)
                
        final_paraphrases = [text] + unique_variants[:4]  # Original + up to 4 variants
        
        # Save to cache
        self._save_to_cache(cache_key, text, final_paraphrases)
        
        return final_paraphrases

    def _generate_synonym_substitutions(self, text: str) -> List[str]:
        """Generate variants using synonym substitution."""
        variants = []
        
        if not NLTK_AVAILABLE:
            return variants
        
        try:
            import nltk
            from nltk.corpus import wordnet
            
            words = text.split()
            if len(words) < 2:
                return variants
            
            # Try substituting each word
            for i, word in enumerate(words):
                # Get synonyms for the word
                synonyms = self._get_synonyms(word.lower())
                
                if synonyms:
                    # Create variants with synonym substitution
                    for synonym in synonyms[:2]:  # Limit to 2 synonyms per word
                        new_words = words.copy()
                        # Preserve original case
                        if word.isupper():
                            new_words[i] = synonym.upper()
                        elif word.istitle():
                            new_words[i] = synonym.capitalize()
                        else:
                            new_words[i] = synonym
                        variants.append(" ".join(new_words))
            
        except Exception as e:
            logger.warning(f"Error in synonym substitution: {e}")
            
        return variants

    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        
        try:
            import nltk
            from nltk.corpus import wordnet
            
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower() and len(synonym.split()) == 1:
                        synonyms.add(synonym)
                        
        except Exception as e:
            logger.debug(f"Error getting synonyms for '{word}': {e}")
            
        return list(synonyms)[:3]  # Return up to 3 synonyms

    def _generate_reorderings(self, text: str) -> List[str]:
        """Generate variants with minor word reordering."""
        variants = []
        words = text.split()
        
        if len(words) < 3:
            return variants
        
        # Simple reorderings for short sentences
        if len(words) <= 6:
            # Move first word to end
            if len(words) >= 3:
                reordered = words[1:] + [words[0]]
                variants.append(" ".join(reordered))
                
            # Move last word to beginning
            if len(words) >= 3:
                reordered = [words[-1]] + words[:-1]
                variants.append(" ".join(reordered))
                
        # For longer sentences, try swapping adjacent words
        elif len(words) > 6:
            # Swap middle words
            mid = len(words) // 2
            if mid > 0 and mid < len(words) - 1:
                swapped = words.copy()
                swapped[mid], swapped[mid + 1] = swapped[mid + 1], swapped[mid]
                variants.append(" ".join(swapped))
        
        return variants
