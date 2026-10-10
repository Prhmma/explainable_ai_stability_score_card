"""
Deletion-based faithfulness (ERASER comprehensiveness and sufficiency).

For an explanation of the predicted class, words are ranked by attribution
(most supportive first). At each fraction q of the input's words:

- comprehensiveness(q) = p(y | x) - p(y | x without its top-q words)
- sufficiency(q)       = p(y | x) - p(y | only the top-q words of x)

Higher comprehensiveness and lower sufficiency indicate a more faithful
ranking. Each is averaged over the fractions (AOPC) and reported next to the
same quantities for a random ranking, so a method is judged against chance.
"""

from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch

FRACTIONS = (0.1, 0.2, 0.5)


def make_predict_proba(model: Any, tokenizer: Any, batch_size: int = 64) -> Callable[[List[str]], np.ndarray]:
    def predict(texts: List[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            batch = [t if t.strip() else tokenizer.pad_token or "" for t in texts[start:start + batch_size]]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out.append(torch.softmax(model(**inputs).logits, dim=-1).float().cpu().numpy())
        return np.concatenate(out)
    return predict


def word_scores_for_text(text: str, words: Sequence[Sequence[Any]]) -> np.ndarray:
    """
    Align an explanation's ordered (word, score) list to the whitespace words of `text`.

    Explainers that skip some words (LIME has no features for punctuation-only
    words) leave those positions at score 0.
    """
    tokens = text.split()
    scores = np.zeros(len(tokens))
    pos = 0
    for word, score in words:
        while pos < len(tokens) and tokens[pos] != word:
            pos += 1
        if pos == len(tokens):
            break
        scores[pos] = score
        pos += 1
    return scores


def _variants(tokens: List[str], order: np.ndarray) -> List[str]:
    """Texts with the top-k words removed and with only the top-k words kept, for each fraction."""
    texts = []
    for q in FRACTIONS:
        k = max(1, int(round(q * len(tokens))))
        top = set(order[:k].tolist())
        texts.append(" ".join(t for i, t in enumerate(tokens) if i not in top))
        texts.append(" ".join(t for i, t in enumerate(tokens) if i in top))
    return texts


def faithfulness(records: List[Dict[str, Any]], predict: Callable[[List[str]], np.ndarray],
                 random_draws: int = 3, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """AOPC comprehensiveness and sufficiency for explanations vs. random rankings."""
    rng = np.random.RandomState(seed)
    plans = []  # (instance index, kind, texts)
    for i, r in enumerate(records):
        text = r['original_text']
        tokens = text.split()
        if len(tokens) < 2:
            continue
        scores = word_scores_for_text(text, r['original_explanation']['words'])
        plans.append((i, 'explainer', _variants(tokens, np.argsort(-scores, kind='stable'))))
        for _ in range(random_draws):
            plans.append((i, 'random', _variants(tokens, rng.permutation(len(tokens)))))

    full = predict([records[i]['original_text'] for i in sorted({p[0] for p in plans})])
    full_prob = {i: full[j] for j, i in enumerate(sorted({p[0] for p in plans}))}
    flat = [t for _, _, texts in plans for t in texts]
    probs = predict(flat)

    acc = {'explainer': {'comprehensiveness': [], 'sufficiency': []},
           'random': {'comprehensiveness': [], 'sufficiency': []}}
    offset = 0
    for i, kind, texts in plans:
        target = records[i]['original_explanation']['predicted_class']
        p_full = full_prob[i][target]
        p = probs[offset:offset + len(texts), target]
        offset += len(texts)
        acc[kind]['comprehensiveness'].append(float(np.mean(p_full - p[0::2])))
        acc[kind]['sufficiency'].append(float(np.mean(p_full - p[1::2])))

    return {
        kind: {metric: float(np.mean(values)) for metric, values in parts.items()} | {'n': len(parts['sufficiency'])}
        for kind, parts in acc.items()
    }
