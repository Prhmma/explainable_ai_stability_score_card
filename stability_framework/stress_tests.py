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


# Version of the paraphrase generator; part of the paraphrase cache key and run fingerprint
# (4: context- and entailment-validated substitutions; no capitalised words; no fixed phrase list)
PARAPHRASE_GENERATOR_VERSION = 4


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


# Verdicts of the LLM judge on paraphrase candidates (see resources/paraphrase_judge_prompt.md)
DEFAULT_JUDGMENTS_PATH = os.path.join(os.path.dirname(__file__), 'resources', 'paraphrase_judgments.jsonl')
JUDGE_PROMPT_VERSION = 1


class PendingJudgmentError(RuntimeError):
    """Raised when paraphrase candidates have not been judged yet."""

    def __init__(self, message: str, candidates: List[Dict[str, Any]]):
        super().__init__(message)
        self.candidates = candidates


def candidate_id(text: str, candidate: Dict[str, Any]) -> str:
    """Stable id of a paraphrase candidate (the text and its edit)."""
    if candidate['type'] == 'synonym':
        key = [text, candidate['position'], candidate['original'], candidate['substitute']]
    else:
        key = [text, 'reorder', candidate['text']]
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:16]


def load_judgments(path: str) -> Dict[str, str]:
    """Candidate id -> verdict from a JSONL file of judge verdicts (empty if the file does not exist)."""
    verdicts: Dict[str, str] = {}
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    verdicts[entry['id']] = entry['verdict']
    return verdicts


class SemanticParaphrasingTest(BaseStressTest):
    """
    Semantic paraphrasing stress test: meaning-preserving variants of a text.

    Candidates are single-word synonym substitutions and clause swaps. A
    candidate is kept only if it passes every check, and the test fails closed
    when a check cannot be run:
    - substitutions touch only lowercase alphabetic content words in their base
      form (never stopwords, capitalised words such as proper nouns and
      headline words, acronyms, numbers or markup), and synonyms come from the
      word's most frequent WordNet senses and are attested in SemCor;
    - the substitute must fit the context: under a masked language model
      (bert-base-cased, a different family from the explained classifiers) its
      log-probability at the word's position is at most `mlm_max_log_drop`
      below the original word's;
    - sentence embeddings (all-MiniLM-L6-v2) must reach `similarity_threshold`;
    - original and variant must entail each other (NLI cross-encoder,
      nli-deberta-v3-base), each direction with probability at least
      `nli_min_entailment`.
    - an LLM judge must accept the candidate in context (verdicts are stored
      in resources/paraphrase_judgments.jsonl; unjudged candidates raise
      PendingJudgmentError rather than being used).
    The best-validated variants (highest weaker-direction entailment) are kept,
    at most one per word position and `max_variants` in total.
    All auxiliary models run on CPU.
    """

    SENSES_PER_WORD = 2
    SENSE_MIN_SHARE = 0.2
    SYNONYMS_PER_WORD = 3
    MLM_NAME = 'bert-base-cased'
    NLI_NAME = 'cross-encoder/nli-deberta-v3-base'

    def __init__(self, random_seed: int = 42, similarity_threshold: float = 0.8, cache_dir: str = None,
                 nli_min_entailment: float = 0.9, mlm_max_log_drop: float = 5.0, max_variants: int = 4,
                 judgments_path: Optional[str] = DEFAULT_JUDGMENTS_PATH):
        super().__init__("SemanticParaphrasing", random_seed)
        self.similarity_threshold = similarity_threshold
        self.nli_min_entailment = nli_min_entailment
        self.mlm_max_log_drop = mlm_max_log_drop
        self.max_variants = max_variants
        # Judge verdicts (candidate id -> "Y"/"N"); None disables judging (diagnostics only)
        self.judgments_path = judgments_path
        self.judgments = load_judgments(judgments_path) if judgments_path else None

        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser('~'), '.cache', 'stability_framework', 'paraphrases'
        )
        self.use_cache = True
        os.makedirs(self.cache_dir, exist_ok=True)

        # Paraphrases are only valid if they are verified, so the test fails
        # closed when the validation models are unavailable
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Semantic paraphrasing requires sentence-transformers for validation")
        if not NLTK_AVAILABLE:
            raise RuntimeError("Semantic paraphrasing requires NLTK (WordNet, POS tagger)")
        # On CPU: on Apple Silicon (MPS) SDPA attention can give wrong outputs
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self._nli = None
        self._mlm = None
        for resource, path in (('wordnet', 'corpora/wordnet'),
                               ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng'),
                               ('stopwords', 'corpora/stopwords')):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)

    # --- validation models (loaded on first use; cache hits never need them) ---

    def _nli_model(self):
        if self._nli is None:
            from sentence_transformers import CrossEncoder
            self._nli = CrossEncoder(self.NLI_NAME, device='cpu')
            labels = {v.lower(): int(k) for k, v in self._nli.model.config.id2label.items()}
            self._entailment_index = labels['entailment']
        return self._nli

    def _mlm_model(self):
        if self._mlm is None:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.MLM_NAME)
            model = AutoModelForMaskedLM.from_pretrained(self.MLM_NAME, attn_implementation='eager').eval()
            self._mlm = (tokenizer, model, torch)
        return self._mlm

    # --- cache ---

    def _parameters(self) -> Dict[str, Any]:
        return {'similarity_threshold': self.similarity_threshold, 'nli_min_entailment': self.nli_min_entailment,
                'mlm_max_log_drop': self.mlm_max_log_drop, 'max_variants': self.max_variants,
                'random_seed': self.random_seed, 'judged': self.judgments is not None,
                'judge_prompt_version': JUDGE_PROMPT_VERSION if self.judgments is not None else None}

    def _get_cache_key(self, text: str) -> str:
        """Cache key for the text, the generator version and its parameters."""
        content = {'text': text, 'generator_version': PARAPHRASE_GENERATOR_VERSION, **self._parameters()}
        return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[List[str]]:
        if not self.use_cache:
            return None
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)['paraphrases']
        except Exception as e:
            logger.warning(f"Error loading from cache {cache_key}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, text: str, paraphrases: List[str],
                       edits: Optional[List[Dict[str, Any]]] = None):
        if not self.use_cache:
            return
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            cache_data = {'original_text': text, 'paraphrases': paraphrases, 'edits': edits or [],
                          'parameters': self._parameters(), 'generator_version': PARAPHRASE_GENERATOR_VERSION,
                          'timestamp': datetime.now().isoformat()}
            # Write atomically: concurrent runs share the cache and must never read a partial file
            tmp_file = f"{cache_file}.{os.getpid()}.tmp"
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_file, cache_file)
        except Exception as e:
            logger.warning(f"Error saving to cache {cache_key}: {e}")

    # --- generation ---

    def apply(self, text: str) -> List[str]:
        """The original text followed by up to `max_variants` validated paraphrases."""
        cache_key = self._get_cache_key(text)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached
        variants, edits = self.generate(text)
        paraphrases = [text] + variants
        self._save_to_cache(cache_key, text, paraphrases, edits)
        return paraphrases

    def candidates(self, text: str) -> List[Dict[str, Any]]:
        """Every candidate that passes the local checks (context, similarity, mutual entailment), with an id."""
        proposals = self._generate_synonym_substitutions(text) + self._generate_reorderings(text)
        if not proposals:
            return []
        sims = self._similarities(text, [c['text'] for c in proposals])
        kept = [dict(c, similarity=float(s)) for c, s in zip(proposals, sims) if s >= self.similarity_threshold]
        if not kept:
            return []
        forward, backward = self._entailment(text, [c['text'] for c in kept])
        validated = []
        for c, f, b in zip(kept, forward, backward):
            c.update(entail_forward=float(f), entail_backward=float(b), id=candidate_id(text, c))
            if min(f, b) >= self.nli_min_entailment:
                validated.append(c)
        return validated

    def generate(self, text: str):
        """
        Validated variants and a record of each one's edit and validation scores.

        With judgments loaded (the default), a candidate is used only if the
        judge accepted it; a candidate without a verdict raises
        PendingJudgmentError, so unjudged paraphrases are never used.
        """
        validated = self.candidates(text)
        if self.judgments is not None:
            missing = [c for c in validated if c['id'] not in self.judgments]
            if missing:
                raise PendingJudgmentError(f"{len(missing)} paraphrase candidate(s) have no judge verdict", missing)
            validated = [dict(c, verdict='Y') for c in validated if self.judgments[c['id']] == 'Y']
        # Most confidently meaning-preserving first; at most one variant per word position
        validated.sort(key=lambda c: (-min(c['entail_forward'], c['entail_backward']), str(c.get('position', -1))))
        chosen, used = [], set()
        for c in validated:
            key = c.get('position', ('reorder',))
            if key in used:
                continue
            used.add(key)
            chosen.append(c)
            if len(chosen) == self.max_variants:
                break
        return [c['text'] for c in chosen], chosen

    def _generate_synonym_substitutions(self, text: str) -> List[Dict[str, Any]]:
        """Context-checked single-word synonym substitutions (see the class docstring)."""
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        if len(spans) < 2:
            return []
        words = [text[a:b] for a, b in spans]
        cores = [w.strip(string.punctuation) for w in words]
        tags = nltk.pos_tag([c or w for c, w in zip(cores, words)])
        stop = self._stopwords()
        proposals = []
        for index, ((a, b), word, core, (_, tag)) in enumerate(zip(spans, words, cores, tags)):
            pos = self._wordnet_pos(tag)
            if pos is None or not core.isalpha() or not core.islower() or core in stop:
                continue  # also skips capitalised words: proper nouns, headline words, acronyms
            if wordnet.morphy(core, pos) != core:
                continue  # inflected form (WordNet synonyms are uninflected)
            for synonym in self._get_synonyms(core, pos):
                variant = text[:a] + word.replace(core, synonym, 1) + text[b:]
                proposals.append({'type': 'synonym', 'position': index, 'original': core,
                                  'substitute': synonym, 'text': variant})
        return [p for p in proposals if self._fits_context(text, spans, p)]

    def _fits_context(self, text: str, spans, proposal: Dict[str, Any]) -> bool:
        """Masked-LM check: the substitute is not much less probable than the original word in context."""
        tokenizer, model, torch = self._mlm_model()
        ids = [tokenizer.convert_tokens_to_ids(w) for w in (proposal['original'], proposal['substitute'])]
        if tokenizer.unk_token_id in ids:
            proposal['mlm_log_drop'] = None
            return False  # multi-piece words cannot be scored at one position
        a, b = spans[proposal['position']]
        word = text[a:b]
        masked = text[:a] + word.replace(proposal['original'], tokenizer.mask_token, 1) + text[b:]
        enc = tokenizer(masked, return_tensors='pt', truncation=True, max_length=512)
        positions = (enc['input_ids'][0] == tokenizer.mask_token_id).nonzero()
        if len(positions) != 1:
            return False
        with torch.no_grad():
            log_probs = torch.log_softmax(model(**enc).logits[0, positions[0, 0]], dim=-1)
        drop = float(log_probs[ids[0]] - log_probs[ids[1]])
        proposal['mlm_log_drop'] = drop
        return drop <= self.mlm_max_log_drop

    def _similarities(self, original: str, variants: List[str]) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(self.similarity_model.encode([original]), self.similarity_model.encode(variants))[0]

    def _entailment(self, original: str, variants: List[str]):
        nli = self._nli_model()
        pairs = [(original, v) for v in variants] + [(v, original) for v in variants]
        probs = nli.predict(pairs, apply_softmax=True, batch_size=32, show_progress_bar=False)[:, self._entailment_index]
        return probs[:len(variants)], probs[len(variants):]

    @staticmethod
    def _has_finite_verb(phrase: str) -> bool:
        """Whether a phrase contains a finite verb (VBZ, VBP, VBD) or modal (MD)."""
        return any(tag in ('VBZ', 'VBP', 'VBD', 'MD') for _, tag in nltk.pos_tag(phrase.split()))

    @staticmethod
    def _wordnet_pos(treebank_tag: str) -> Optional[str]:
        """Map a Penn Treebank tag to a WordNet POS (content words only; never proper nouns)."""
        if treebank_tag in ('NNP', 'NNPS'):
            return None
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
        """
        Single-word synonyms from the word's most frequent senses, attested in SemCor.

        Restricting to the leading senses (WordNet orders senses by frequency),
        to senses in which the word itself is commonly used, and to lemmas with
        a non-zero SemCor count avoids rare senses such as "company" ->
        "companionship", "buy" -> "bribe" or "good" -> "full".
        """
        synonyms: Dict[str, None] = {}
        senses = wordnet.synsets(word, pos=pos)[:self.SENSES_PER_WORD]
        # A sense counts only if the word is used in it at least SENSE_MIN_SHARE as
        # often as in its main sense (SemCor counts), e.g. not "buy" as "bribe"
        uses = [sum(l.count() for l in s.lemmas() if l.name().lower() == word) for s in senses]
        if uses and max(uses) > 0:
            senses = [s for s, u in zip(senses, uses) if u >= self.SENSE_MIN_SHARE * max(uses)]
        else:
            senses = senses[:1]
        for synset in senses:
            for lemma in synset.lemmas():
                synonym = lemma.name()
                if (synonym.lower() != word and synonym.isalpha() and synonym.islower()
                        and lemma.count() > 0):
                    synonyms.setdefault(synonym, None)
        return list(synonyms)[:self.SYNONYMS_PER_WORD]

    def _generate_reorderings(self, text: str) -> List[Dict[str, Any]]:
        """
        Swap two clauses joined by a comma or "and" ("A , B" -> "B , A").

        Both parts must contain a finite verb, and clauses starting with a
        subordinating or contrastive word are not moved. Validation then
        requires mutual entailment, which rejects swaps that change meaning.
        """
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
                return [{'type': 'reorder', 'text': (f"{swapped} {trailing}" if trailing else swapped) + tail_space}]
        return []


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