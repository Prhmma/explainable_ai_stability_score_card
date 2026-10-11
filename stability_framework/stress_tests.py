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


# Version of the paraphrase generator; part of the paraphrase cache key
PARAPHRASE_GENERATOR_VERSION = 3


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
        
    MAX_VARIANTS = 7

    def apply(self, text: str) -> List[str]:
        """
        Generate preprocessing variants of the text: the original first, then up
        to MAX_VARIANTS variants taken round-robin from the case, punctuation and
        spacing categories, so that every category is represented (taking them
        in order let case and punctuation fill every slot).
        """
        categories = [
            self._generate_case_variants(text),
            self._generate_punctuation_variants(text),
            self._generate_spacing_variants(text),
        ]
        seen = {text}
        variants = [text]
        for rank in range(max(len(c) for c in categories)):
            for category in categories:
                if rank < len(category) and category[rank] not in seen and category[rank].strip():
                    seen.add(category[rank])
                    variants.append(category[rank])
        return variants[:1 + self.MAX_VARIANTS]
    
    def _rng(self, text: str) -> random.Random:
        """
        Random generator seeded by the experiment seed and the text itself.

        Variants of a text are then identical for every explainer and do not
        depend on how many texts were perturbed before (the global random
        stream made each explainer see different random variants of the same
        sentence, and a resumed run differ from an uninterrupted one).
        """
        return random.Random(f"{self.random_seed}:{text}")

    def _generate_case_variants(self, text: str) -> List[str]:
        """Generate case variations."""
        variants = []
        rng = self._rng(text)
        
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
                if rng.random() < 0.3:  # 30% chance to modify
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
        
        # Add terminal punctuation only where there is none (ignoring trailing
        # whitespace, which SST-2 sentences carry)
        body = text.rstrip()
        if body and not body.endswith(('.', '!', '?')):
            variants.append(body + '.')
            variants.append(body + '!')
            
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
            idx = self._rng(text).randint(1, len(words) - 1)
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
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser('~'), '.cache', 'stability_framework', 'paraphrases'
        )
        self.use_cache = True
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize semantic similarity model if available
        # Paraphrases are only valid if their similarity to the original is
        # verified, so the test fails closed when the model is unavailable
        # (previously unvalidated paraphrases were used, and cached).
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Semantic paraphrasing requires sentence-transformers for similarity validation")
        # On CPU: on Apple Silicon (MPS) its default SDPA attention can give wrong
        # outputs, and variable-length batches grew memory without bound
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        logger.info("Loaded sentence transformer model for similarity filtering")
        
        # Initialize NLTK wordnet if available
        if NLTK_AVAILABLE:
            try:
                # Download only what is missing (a network call each time slowed every start-up)
                for resource, path in (('wordnet', 'corpora/wordnet'),
                                       ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng'),
                                       ('stopwords', 'corpora/stopwords')):
                    try:
                        nltk.data.find(path)
                    except LookupError:
                        nltk.download(resource, quiet=True)
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
            'nltk_available': NLTK_AVAILABLE,
            # Bump when paraphrase generation changes so stale caches are not reused
            'generator_version': PARAPHRASE_GENERATOR_VERSION
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
        
        # Generate each category of paraphrase, filtered by semantic similarity
        categories = [
            self._generate_synonym_substitutions(text),
            self._generate_reorderings(text),
            self._generate_simple_transformations(text),
        ]
        categories = [self._filter_by_similarity(text, c) if c else c for c in categories]

        # Take variants round-robin across categories so that every category
        # is represented, rather than the first category filling every slot
        seen = {text}
        unique_variants = []
        for rank in range(max((len(c) for c in categories), default=0)):
            for category in categories:
                if rank < len(category) and category[rank] not in seen:
                    seen.add(category[rank])
                    unique_variants.append(category[rank])

        final_paraphrases = [text] + unique_variants[:4]  # Original + up to 4 variants
        
        # Save to cache
        self._save_to_cache(cache_key, text, final_paraphrases)
        
        return final_paraphrases
    
    def _generate_synonym_substitutions(self, text: str) -> List[str]:
        """
        Replace one content word at a time with a WordNet synonym of the same part of speech.

        Only nouns, verbs, adjectives and adverbs in their base form are
        substituted, never stopwords or auxiliaries (WordNet would otherwise
        map "a" to "ampere" or "is" to "exist"), and inflected words are left
        alone because WordNet synonyms are uninflected ("puts" -> "put").
        Capitalized synonyms (acronyms, proper nouns) are skipped. Synonyms
        keep WordNet's frequency order, so the output is deterministic. The
        rest of the text, including whitespace, is left unchanged.
        """
        variants = []
        if not NLTK_AVAILABLE:
            return variants

        try:
            import nltk

            spans = [m.span() for m in re.finditer(r"\S+", text)]
            if len(spans) < 2:
                return variants

            words = [text[a:b] for a, b in spans]
            cores = [w.strip(string.punctuation) for w in words]
            tags = nltk.pos_tag([c or w for c, w in zip(cores, words)])
            stop = self._stopwords()

            for (a, b), word, core, (_, tag) in zip(spans, words, cores, tags):
                pos = self._wordnet_pos(tag)
                if pos is None or not core.isalpha() or core.lower() in stop:
                    continue
                if wordnet.morphy(core.lower(), pos) != core.lower():
                    continue  # inflected form
                for synonym in self._get_synonyms(core.lower(), pos)[:2]:  # Limit to 2 synonyms per word
                    if core.isupper():
                        synonym = synonym.upper()
                    elif core.istitle():
                        synonym = synonym.capitalize()
                    variants.append(text[:a] + word.replace(core, synonym, 1) + text[b:])

        except Exception as e:
            logger.warning(f"Error in synonym substitution: {e}")

        return variants

    @staticmethod
    def _has_finite_verb(phrase: str) -> bool:
        """Whether a phrase contains a finite verb (VBZ, VBP, VBD) or modal (MD)."""
        if not NLTK_AVAILABLE:
            return False
        return any(tag in ('VBZ', 'VBP', 'VBD', 'MD') for _, tag in nltk.pos_tag(phrase.split()))

    @staticmethod
    def _wordnet_pos(treebank_tag: str) -> Optional[str]:
        """Map a Penn Treebank tag to a WordNet POS (content words only)."""
        if treebank_tag.startswith('NN'):
            return wordnet.NOUN
        if treebank_tag.startswith('VB'):
            return wordnet.VERB
        if treebank_tag.startswith('JJ'):
            return wordnet.ADJ
        if treebank_tag.startswith('RB'):
            return wordnet.ADV
        return None

    @staticmethod
    def _stopwords() -> set:
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
        except Exception:
            return {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
                    "have", "has", "had", "do", "does", "did", "not", "no", "it", "its", "s", "t"}

    def _get_synonyms(self, word: str, pos: Optional[str] = None) -> List[str]:
        """Single-word WordNet synonyms of `word` (optionally of one POS), in WordNet order."""
        synonyms: Dict[str, None] = {}
        try:
            for synset in wordnet.synsets(word, pos=pos):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower() and synonym.isalpha() and synonym.islower():
                        synonyms.setdefault(synonym, None)
        except Exception as e:
            logger.debug(f"Error getting synonyms for '{word}': {e}")

        return list(synonyms)[:3]  # Return up to 3 synonyms

    def _generate_reorderings(self, text: str) -> List[str]:
        """
        Swap two clauses joined by a comma or "and" ("A , B" -> "B , A").

        Both parts must contain a finite verb (so phrases such as "a moving,
        beautifully acted film" are not treated as clauses), and clauses
        starting with a subordinating or contrastive word ("but", "because",
        ...) are not moved, since swapping them changes meaning. Word-level
        shuffles, which break grammaticality, are not generated.
        """
        variants = []
        body = text.rstrip()
        tail_space = text[len(body):]
        trailing = ""
        match = re.search(r"\s*[.!?]+$", body)
        if match:
            trailing = body[match.start():].strip()
            body = body[:match.start()]

        for joiner in (" , ", ", ", " and "):
            parts = body.split(joiner)
            if len(parts) == 2 and all(len(p.split()) >= 2 and self._has_finite_verb(p) for p in parts):
                first, second = parts[0].strip(), parts[1].strip()
                if second.split()[0].lower() in {"but", "yet", "so", "because", "although", "though", "while"}:
                    continue
                swapped = f"{second}{joiner}{first}"
                variants.append((f"{swapped} {trailing}" if trailing else swapped) + tail_space)
                break

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
        """Keep variants whose cosine similarity to the original meets the threshold (errors propagate)."""
        from sklearn.metrics.pairwise import cosine_similarity
        original_embedding = self.similarity_model.encode([original])
        variant_embeddings = self.similarity_model.encode(variants)
        similarities = cosine_similarity(original_embedding, variant_embeddings)[0]
        filtered_variants = [v for v, sim in zip(variants, similarities) if sim >= self.similarity_threshold or v == original]
        logger.debug(f"Filtered {len(variants)} to {len(filtered_variants)} variants by similarity")
        return filtered_variants


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
    elif test_name in ['semantic_paraphrasing_v1', 'semanticparaphrasingv1']:
        from .legacy_paraphrase import LegacyParaphrasingTest
        return LegacyParaphrasingTest(random_seed=random_seed, **kwargs)
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
    },
    'semantic_paraphrasing_v1': {
        'available': SENTENCE_TRANSFORMERS_AVAILABLE or NLTK_AVAILABLE,
        'description': 'The paraphrase generator released with the paper, for comparison studies only',
        'parameters': {'similarity_threshold': 0.8}
    }
}