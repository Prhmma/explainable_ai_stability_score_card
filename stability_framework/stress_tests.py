"""
Stress Tests Module for Stability Assessment Framework

Implements three categories of stress tests: seed variation, preprocessing variants,
and semantic paraphrasing to evaluate explanation stability.
"""

import random
import re
import string
import hashlib
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from abc import ABC, abstractmethod

# Try to import additional libraries for paraphrasing
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, semantic similarity filtering will be limited")

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, synonym substitution will be limited")

logger = logging.getLogger(__name__)


class BaseStressTest(ABC):
    """Base class for stress tests."""
    
    def __init__(self, name: str, random_seed: int = 42):
        self.name = name
        self.random_seed = random_seed
        self._setup_random_seed(random_seed)
        
    def _setup_random_seed(self, seed: int):
        """Setup random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        
    @abstractmethod
    def apply(self, text: str) -> List[str]:
        """
        Apply stress test to generate variants of the input text.
        
        Args:
            text: Original input text
            
        Returns:
            List of text variants
        """
        pass
    
    def get_test_info(self) -> Dict[str, Any]:
        """Get information about this stress test."""
        return {
            'name': self.name,
            'random_seed': self.random_seed,
            'description': self.__doc__ or "No description available"
        }


class SeedVariationTest(BaseStressTest):
    """
    Seed variation stress test - applies different random seeds during explanation computation.
    This tests method reproducibility and stochastic variation.
    """
    
    def __init__(self, seeds: Optional[List[int]] = None, random_seed: int = 42):
        super().__init__("SeedVariation", random_seed)
        self.seeds = seeds or [42, 43, 44, 45, 46]
        
    def apply(self, text: str) -> List[str]:
        """
        For seed variation, we return the same text multiple times.
        The variation happens during explanation computation with different seeds.
        """
        # Return the same text for each seed - the variation is in explanation computation
        return [text] * len(self.seeds)
    
    def get_seeds(self) -> List[int]:
        """Get the list of seeds to use for explanation computation."""
        return self.seeds.copy()


class PreprocessingVariationTest(BaseStressTest):
    """
    Preprocessing variation stress test - generates variants with case, punctuation, and spacing changes.
    Tests robustness to common text preprocessing variations.
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__("PreprocessingVariation", random_seed)
        
    def apply(self, text: str) -> List[str]:
        """Generate preprocessing variants of the text."""
        variants = []
        
        # Original text (as reference)
        variants.append(text)
        
        # Case variations
        variants.extend(self._generate_case_variants(text))
        
        # Punctuation variations
        variants.extend(self._generate_punctuation_variants(text))
        
        # Spacing variations
        variants.extend(self._generate_spacing_variants(text))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
                
        return unique_variants[:8]  # Limit to reasonable number
    
    def _generate_case_variants(self, text: str) -> List[str]:
        """Generate case variations."""
        variants = []
        
        # All uppercase
        variants.append(text.upper())
        
        # All lowercase
        variants.append(text.lower())
        
        # Title case
        variants.append(text.title())
        
        # Random case changes (keep original structure mostly)
        words = text.split()
        if len(words) > 1:
            # Randomly change case of some words
            modified_words = []
            for word in words:
                if random.random() < 0.3:  # 30% chance to modify
                    if word.islower():
                        modified_words.append(word.upper())
                    elif word.isupper():
                        modified_words.append(word.lower())
                    else:
                        modified_words.append(word.lower())
                else:
                    modified_words.append(word)
            variants.append(" ".join(modified_words))
        
        return variants
    
    def _generate_punctuation_variants(self, text: str) -> List[str]:
        """Generate punctuation variations."""
        variants = []
        
        # Remove all punctuation
        no_punct = text.translate(str.maketrans('', '', string.punctuation))
        variants.append(no_punct)
        
        # Change common punctuation
        punct_changes = [
            ('!', '.'),
            ('?', '.'),
            ('.', '!'),
            (',', ' '),
            (';', ','),
            (':', ' -'),
        ]
        
        for old_punct, new_punct in punct_changes:
            if old_punct in text:
                variants.append(text.replace(old_punct, new_punct))
        
        # Add extra punctuation
        if not text.endswith(('.', '!', '?')):
            variants.append(text + '.')
            variants.append(text + '!')
            
        return variants
    
    def _generate_spacing_variants(self, text: str) -> List[str]:
        """Generate spacing variations."""
        variants = []
        
        # Double spaces
        variants.append(re.sub(r' ', '  ', text))
        
        # Remove extra spaces and normalize
        variants.append(re.sub(r'\s+', ' ', text).strip())
        
        # Add leading/trailing spaces
        variants.append(' ' + text + ' ')
        
        # Random extra spaces between words
        words = text.split()
        if len(words) > 2:
            # Add extra space randomly
            idx = random.randint(1, len(words) - 1)
            spaced_words = words[:idx] + [''] + words[idx:]
            variants.append(' '.join(spaced_words))
            
        return variants


class SemanticParaphrasingTest(BaseStressTest):
    """
    Semantic paraphrasing stress test - generates meaning-preserving variants.
    Uses synonym substitution and minor reordering with semantic similarity filtering.
    """
    
    def __init__(self, random_seed: int = 42, similarity_threshold: float = 0.8, cache_dir: str = None):
        super().__init__("SemanticParaphrasing", random_seed)
        self.similarity_threshold = similarity_threshold
        
        # Set up caching
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'cache', 'paraphrases')
        self.use_cache = True
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize semantic similarity model if available
        self.similarity_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model for similarity filtering")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
        
        # Initialize NLTK wordnet if available
        if NLTK_AVAILABLE:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                logger.info("NLTK wordnet initialized")
            except Exception as e:
                logger.warning(f"Could not initialize NLTK: {e}")
        
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text and parameters."""
        # Create a hash based on text content and paraphrasing parameters
        content = {
            'text': text,
            'similarity_threshold': self.similarity_threshold,
            'random_seed': self.random_seed,
            'has_similarity_model': self.similarity_model is not None,
            'nltk_available': NLTK_AVAILABLE
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[str]]:
        """Load paraphrases from cache if available."""
        if not self.use_cache:
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    logger.debug(f"Loaded paraphrases from cache: {cache_key}")
                    return cached_data['paraphrases']
        except Exception as e:
            logger.warning(f"Error loading from cache {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, text: str, paraphrases: List[str]):
        """Save paraphrases to cache."""
        if not self.use_cache:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            cache_data = {
                'original_text': text,
                'paraphrases': paraphrases,
                'parameters': {
                    'similarity_threshold': self.similarity_threshold,
                    'random_seed': self.random_seed,
                    'has_similarity_model': self.similarity_model is not None
                },
                'timestamp': datetime.now().isoformat(),
                'cache_version': '1.0'
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Saved paraphrases to cache: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Error saving to cache {cache_key}: {e}")
    
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
    
    def _generate_simple_transformations(self, text: str) -> List[str]:
        """Generate simple transformations that preserve meaning."""
        variants = []
        
        # Common phrase transformations
        transformations = [
            (r'\bgood\b', 'great'),
            (r'\bgreat\b', 'excellent'),
            (r'\bbad\b', 'terrible'),
            (r'\bterrible\b', 'awful'),
            (r'\bmovie\b', 'film'),
            (r'\bfilm\b', 'movie'),
            (r'\bvery\b', 'extremely'),
            (r'\breally\b', 'very'),
            (r'\bawesome\b', 'amazing'),
            (r'\bnice\b', 'pleasant'),
        ]
        
        for pattern, replacement in transformations:
            if re.search(pattern, text, re.IGNORECASE):
                variant = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                if variant != text:
                    variants.append(variant)
                    
        return variants
    
    def _filter_by_similarity(self, original: str, variants: List[str]) -> List[str]:
        """Filter variants by semantic similarity to original."""
        if not self.similarity_model:
            return variants
            
        try:
            # Calculate embeddings
            original_embedding = self.similarity_model.encode([original])
            variant_embeddings = self.similarity_model.encode(variants)
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(original_embedding, variant_embeddings)[0]
            
            # Filter by threshold
            filtered_variants = []
            for variant, similarity in zip(variants, similarities):
                if similarity >= self.similarity_threshold or variant == original:
                    filtered_variants.append(variant)
                    
            logger.debug(f"Filtered {len(variants)} to {len(filtered_variants)} variants by similarity")
            return filtered_variants
            
        except Exception as e:
            logger.warning(f"Error in similarity filtering: {e}")
            return variants


# Factory function for creating stress tests
def create_stress_test(test_name: str, random_seed: int = 42, **kwargs) -> BaseStressTest:
    """
    Factory function to create stress tests.
    
    Args:
        test_name: Name of the stress test
        random_seed: Random seed for reproducibility
        **kwargs: Additional parameters for the test
        
    Returns:
        Configured stress test instance
    """
    test_name = test_name.lower()
    
    if test_name in ['seed', 'seed_variation', 'seedvariation']:
        return SeedVariationTest(random_seed=random_seed, **kwargs)
    elif test_name in ['preprocessing', 'preprocessing_variation', 'preprocessingvariation']:
        return PreprocessingVariationTest(random_seed=random_seed, **kwargs)
    elif test_name in ['paraphrasing', 'semantic_paraphrasing', 'semanticparaphrasing']:
        return SemanticParaphrasingTest(random_seed=random_seed, **kwargs)
    else:
        raise ValueError(f"Unknown stress test: {test_name}")


# Available stress tests configuration
AVAILABLE_STRESS_TESTS = {
    'seed_variation': {
        'class': SeedVariationTest,
        'description': 'Tests explanation reproducibility with different random seeds',
        'parameters': {'seeds': [42, 43, 44, 45, 46]}
    },
    'preprocessing_variation': {
        'class': PreprocessingVariationTest,
        'description': 'Tests robustness to case, punctuation, and spacing changes',
        'parameters': {}
    },
    'semantic_paraphrasing': {
        'class': SemanticParaphrasingTest,
        'available': SENTENCE_TRANSFORMERS_AVAILABLE or NLTK_AVAILABLE,
        'description': 'Tests stability with meaning-preserving text variations',
        'parameters': {'similarity_threshold': 0.8}
    }
}