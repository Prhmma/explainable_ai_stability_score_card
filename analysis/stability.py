"""Stability summaries from per-instance records, including prediction-invariant subsets."""

import glob
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

METRICS = ('jaccard_at_5', 'spearman_correlation', 'attr_diff_relative_difference_percentage')


def load_per_instance(instance_dir: str) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Load {(explainer, stress test): [instance record, ...]} from a per_instance directory."""
    records = {}
    for path in sorted(glob.glob(os.path.join(instance_dir, '*.jsonl'))):
        explainer, stress_test = os.path.basename(path)[:-len('.jsonl')].split('_', 1)
        with open(path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        records[(explainer, stress_test)] = sorted(rows, key=lambda r: r['sample_idx'])
    return records


def _summarize(values: Iterable[float]) -> Dict[str, float]:
    """Mean, sd and count over defined values; undefined (NaN) comparisons are counted separately."""
    arr = np.asarray(list(values), dtype=float)
    defined = arr[~np.isnan(arr)]
    undefined = int(arr.size - defined.size)
    if defined.size == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'n': 0, 'undefined': undefined}
    return {'mean': float(defined.mean()), 'std': float(defined.std()), 'n': int(defined.size), 'undefined': undefined}


def stability_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summaries over all comparisons and over prediction-invariant comparisons only.

    A comparison is prediction-invariant when the perturbed input receives the
    same predicted class as the original. Restricting to these separates
    explanation instability from changes in the model's decision.
    """
    comparisons = [m for r in records for m in r['metrics']]
    invariant = [m for m in comparisons if m['comparison_info']['same_prediction']]
    return {
        'all': {k: _summarize(m[k] for m in comparisons) for k in METRICS},
        'prediction_invariant': {k: _summarize(m[k] for m in invariant) for k in METRICS},
        'n_instances': len(records),
        'n_comparisons': len(comparisons),
        'prediction_retention': len(invariant) / len(comparisons) if comparisons else float('nan'),
    }


def cross_method_agreement(records: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """
    Agreement between explainers on the same unperturbed inputs (J@5 and Spearman ρ).

    Uses the original explanations stored in each explainer's preprocessing file.
    """
    from stability_framework.metrics import StabilityMetrics
    metrics = StabilityMetrics()
    originals = {
        explainer: {r['sample_idx']: r['original_explanation']['attributions'] for r in rows}
        for (explainer, test), rows in records.items() if test == 'PreprocessingVariation'
    }
    names = sorted(originals)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(set(originals[a]) & set(originals[b]))
            j = [metrics.jaccard_similarity_at_k(originals[a][s], originals[b][s], 5) for s in shared]
            rho = [metrics.spearman_rank_correlation(originals[a][s], originals[b][s]) for s in shared]
            out[f'{a} vs {b}'] = {'jaccard_at_5': _summarize(j), 'spearman_correlation': _summarize(rho)}
    return out
