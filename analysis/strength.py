"""
Perturbation strength and its relation to explanation stability.

For each original/variant pair:
- edit_fraction: word-level Levenshtein distance divided by the original's word count
  (case-insensitive, so pure case changes have strength 0 on this axis)
- semantic_distance: 1 - cosine similarity of sentence embeddings (all-MiniLM-L6-v2)

stability_by_strength() bins comparisons by strength and reports mean J@5 / ρ per bin,
plus the Spearman correlation between strength and each stability metric.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import spearmanr


def word_edit_fraction(original: str, variant: str) -> float:
    a, b = original.lower().split(), variant.lower().split()
    if not a:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, wb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb))
        prev = cur
    return prev[-1] / len(a)


def annotate_strength(records: List[Dict[str, Any]], embed=None) -> List[Dict[str, float]]:
    """One row per comparison: strength measures, stability metrics and prediction invariance."""
    rows, pairs = [], []
    for r in records:
        for variant, m in zip(r['variant_texts'][1:], r['metrics']):
            rows.append({
                'edit_fraction': word_edit_fraction(r['original_text'], variant),
                'jaccard_at_5': m['jaccard_at_5'],
                'spearman_correlation': m['spearman_correlation'],
                'same_prediction': m['comparison_info']['same_prediction'],
            })
            pairs.append((r['original_text'], variant))
    if embed is not None and pairs:
        originals = embed([p[0] for p in pairs])
        variants = embed([p[1] for p in pairs])
        cos = np.sum(originals * variants, axis=1) / (
            np.linalg.norm(originals, axis=1) * np.linalg.norm(variants, axis=1))
        for row, c in zip(rows, cos):
            row['semantic_distance'] = float(1 - c)
    return rows


def make_embedder(device: str = 'cpu'):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return lambda texts: model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=False)


def stability_by_strength(rows: List[Dict[str, float]], axis: str = 'edit_fraction',
                          edges: Sequence[float] = (0.0, 0.05, 0.15, 0.3, 1.01)) -> Dict[str, Any]:
    values = np.array([r[axis] for r in rows])
    out = {'axis': axis, 'bins': []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, v in enumerate(values) if lo <= v < hi]
        out['bins'].append({
            'range': [lo, hi], 'n': len(idx),
            'jaccard_at_5': float(np.nanmean([rows[i]['jaccard_at_5'] for i in idx])) if idx else None,
            'spearman_correlation': float(np.nanmean([rows[i]['spearman_correlation'] for i in idx])) if idx else None,
            'prediction_retention': float(np.mean([rows[i]['same_prediction'] for i in idx])) if idx else None,
        })
    for metric in ('jaccard_at_5', 'spearman_correlation'):
        y = np.array([r[metric] for r in rows], dtype=float)
        keep = ~np.isnan(y)
        rho = spearmanr(values[keep], y[keep]).correlation if len(set(values[keep])) > 1 else float('nan')
        out[f'spearman_strength_vs_{metric}'] = float(rho)
    return out
