"""
Measurement choices: how reported stability depends on aggregation and metric definitions.

Computed post hoc from the per-instance records, per explainer and stress test:
- aggregation: mean over comparisons (as reported) vs. over instances (each
  instance weighted equally); 95% intervals that treat comparisons as
  independent vs. a cluster bootstrap over instances (comparisons of the same
  instance share its original explanation and are correlated);
- J@5 by absolute attribution (as reported) vs. signed attribution (the five
  words pushing most strongly toward the explained class);
- AD over the words both explanations share (as reported) vs. over their union,
  a word missing from one explanation scored 0 there;
- perturbation strength in the model's input space: token-level edit fraction
  with the model's own tokenizer, next to the case-insensitive word-level one;
- explainer differences: paired over the instances both explainers kept, the
  mean difference in instance-level J@5 with a bootstrap 95% interval, so that
  rankings are only read where the interval excludes 0.

Usage (from the repository root):
    uv run python -m analysis.measurement experiment_1
"""

import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

from analysis.report import REPO_ROOT, RUNNER_NAMES, latest_instance_dir
from analysis.stability import load_latest_records, load_per_instance
from analysis.strength import word_edit_fraction

METRIC_LABELS = {'jaccard_at_5': 'J@5', 'spearman_correlation': 'ρ',
                 'attr_diff_relative_difference_percentage': 'AD (%)'}
N_BOOTSTRAP = 2000


def compared_pairs(record: Dict[str, Any]) -> List[Tuple[Dict[str, float], Dict[str, float], str, Dict[str, Any]]]:
    """(original attributions, variant attributions, variant text, stored metrics) per comparison."""
    indices = record.get('compared_variant_indices') or range(1, len(record['variant_explanations']))
    original = record['original_explanation']['attributions']
    return [(original, record['variant_explanations'][i]['attributions'], record['variant_texts'][i], m)
            for i, m in zip(indices, record['metrics'])]


def jaccard_at_k(a: Dict[str, float], b: Dict[str, float], k: int = 5, signed: bool = False) -> float:
    key = (lambda w: d[w]) if signed else (lambda w: abs(d[w]))
    tops = []
    for d in (a, b):
        tops.append(set(sorted(d, key=key, reverse=True)[:k]))
    union = tops[0] | tops[1]
    return len(tops[0] & tops[1]) / len(union) if union else 1.0


def ad_union(a: Dict[str, float], b: Dict[str, float]) -> float:
    """AD (%) over the union of words; a word absent from one explanation counts as 0 there."""
    words = set(a) | set(b)
    if not words:
        return float('nan')
    x = np.array([a.get(w, 0.0) for w in words])
    y = np.array([b.get(w, 0.0) for w in words])
    scale = max(np.abs(x).max(), np.abs(y).max())
    return float(np.mean(np.abs(x - y)) / scale * 100) if scale > 0 else 0.0


def levenshtein(a: List[int], b: List[int]) -> int:
    """Edit distance, one numpy pass per row (the left-to-right dependency becomes a running minimum)."""
    if not a or not b:
        return max(len(a), len(b))
    b_arr = np.asarray(b)
    offsets = np.arange(len(b) + 1)
    prev = offsets.copy()
    for i, token in enumerate(a, 1):
        x = np.empty(len(b) + 1, dtype=np.int64)
        x[0] = i
        x[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (b_arr != token))
        prev = np.minimum.accumulate(x - offsets) + offsets
    return int(prev[-1])


def interval_summary(per_instance: List[np.ndarray], rng: np.random.Generator) -> Dict[str, float]:
    """Comparison- and instance-level means with naive and cluster-bootstrap 95% intervals."""
    groups = [g[~np.isnan(g)] for g in per_instance]
    groups = [g for g in groups if g.size]
    flat = np.concatenate(groups)
    sums = np.array([g.sum() for g in groups])
    counts = np.array([g.size for g in groups])
    means = sums / counts
    n, m = flat.size, len(groups)
    naive_half = stats.t.ppf(0.975, n - 1) * flat.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    idx = rng.integers(0, m, size=(N_BOOTSTRAP, m))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    inst_half = stats.t.ppf(0.975, m - 1) * means.std(ddof=1) / np.sqrt(m) if m > 1 else float('nan')
    return {
        'comparison_mean': float(flat.mean()), 'n_comparisons': int(n),
        'naive_ci_half_width': float(naive_half),
        'cluster_bootstrap_ci': [float(lo), float(hi)],
        'cluster_ci_half_width': float((hi - lo) / 2),
        'instance_mean': float(means.mean()), 'n_instances': int(m),
        'instance_ci_half_width': float(inst_half),
    }


def make_tokenizer(experiment_key: str):
    from stability_framework.model_loader import EXPERIMENT_MODELS, ModelLoader
    # The sentence tokenizer alone (for T5 the prompt is fixed, so it adds no edits)
    tokenizer = ModelLoader(device='cpu')._load_tokenizer(EXPERIMENT_MODELS[experiment_key]['name'])
    return lambda text: tokenizer(text, add_special_tokens=False)['input_ids']


def build(experiment_key: str) -> Dict[str, Any]:
    records = load_latest_records(experiment_key)
    tokenize = make_tokenizer(experiment_key)
    token_cache: Dict[str, List[int]] = {}
    strength_cache: Dict[Tuple[str, str], float] = {}
    rng = np.random.default_rng(0)

    def token_edit_fraction(original: str, variant: str) -> float:
        if (original, variant) not in strength_cache:
            ids = [token_cache.setdefault(t, tokenize(t)) for t in (original, variant)]
            strength_cache[(original, variant)] = levenshtein(*ids) / max(len(ids[0]), 1)
        return strength_cache[(original, variant)]

    out = {'experiment': experiment_key, 'combinations': {}}
    for (explainer, test), rows in records.items():
        per_metric = {k: [] for k in METRIC_LABELS}
        signed, union = [], []
        strength = {'word_edit_fraction': [], 'token_edit_fraction': [], 'jaccard_at_5': [], 'spearman_correlation': []}
        for r in rows:
            pairs = compared_pairs(r)
            for k in METRIC_LABELS:
                per_metric[k].append(np.array([m[k] for *_, m in pairs], dtype=float))
            for a, b, variant, m in pairs:
                signed.append((jaccard_at_k(a, b), jaccard_at_k(a, b, signed=True)))
                union.append((m['attr_diff_relative_difference_percentage'], ad_union(a, b)))
                if test != 'SeedVariation':
                    strength['word_edit_fraction'].append(word_edit_fraction(r['original_text'], variant))
                    strength['token_edit_fraction'].append(token_edit_fraction(r['original_text'], variant))
                    strength['jaccard_at_5'].append(m['jaccard_at_5'])
                    strength['spearman_correlation'].append(m['spearman_correlation'])
        signed, union = np.array(signed, dtype=float), np.array(union, dtype=float)
        entry = {
            'intervals': {k: interval_summary(v, rng) for k, v in per_metric.items()},
            'jaccard_at_5_absolute': float(np.mean(signed[:, 0])),
            'jaccard_at_5_signed': float(np.mean(signed[:, 1])),
            'ad_intersection': float(np.nanmean(union[:, 0])),
            'ad_union': float(np.nanmean(union[:, 1])),
        }
        if strength['token_edit_fraction']:
            s = {k: np.array(v, dtype=float) for k, v in strength.items()}
            entry['strength'] = {'mean_word_edit_fraction': float(s['word_edit_fraction'].mean()),
                                 'mean_token_edit_fraction': float(s['token_edit_fraction'].mean())}
            for axis in ('word_edit_fraction', 'token_edit_fraction'):
                for metric in ('jaccard_at_5', 'spearman_correlation'):
                    keep = ~np.isnan(s[metric])
                    x = s[axis][keep]
                    rho = spearmanr(x, s[metric][keep]).correlation if len(set(x)) > 1 else float('nan')
                    entry['strength'][f'{axis}_vs_{metric}'] = float(rho)
        out['combinations'][f'{explainer}/{test}'] = entry
    out['explainer_differences'] = explainer_differences(records, rng)
    return out


def explainer_differences(records, rng: np.random.Generator, metric: str = 'jaccard_at_5') -> Dict[str, Any]:
    """Paired instance-level differences between explainers under each stress test."""
    by_test: Dict[str, Dict[str, Dict[int, float]]] = {}
    for (explainer, test), rows in records.items():
        by_test.setdefault(test, {})[explainer] = {
            r['sample_idx']: float(np.nanmean([m[metric] for m in r['metrics']])) for r in rows if r['metrics']}
    out = {}
    for test, per_explainer in by_test.items():
        names = sorted(per_explainer)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = sorted(set(per_explainer[a]) & set(per_explainer[b]))
                d = np.array([per_explainer[a][s] - per_explainer[b][s] for s in shared])
                d = d[~np.isnan(d)]
                boot = d[rng.integers(0, d.size, size=(N_BOOTSTRAP, d.size))].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                out[f'{test}: {a} − {b}'] = {'mean_difference': float(d.mean()), 'ci': [float(lo), float(hi)],
                                             'n_instances': int(d.size), 'distinguishable': bool(lo > 0 or hi < 0)}
    return out


def to_markdown(result: Dict[str, Any]) -> str:
    combos = result['combinations']
    lines = [f"# Measurement choices: {result['experiment']}", "",
             "## 1. Aggregation and interval estimation", "",
             "Comparison-level mean (as reported) with a 95% CI treating comparisons as independent, the cluster "
             "bootstrap CI over instances, and the instance-level mean. Ratio = cluster / naive CI half-width.", "",
             "| Explainer / stress test | Metric | Mean (comparisons) | Naive ± | Cluster 95% CI | Ratio | Mean (instances) ± |",
             "|---|---|---|---|---|---|---|"]
    for name, e in combos.items():
        for k, label in METRIC_LABELS.items():
            s = e['intervals'][k]
            lo, hi = s['cluster_bootstrap_ci']
            lines.append(f"| {name} | {label} | {s['comparison_mean']:.3f} | {s['naive_ci_half_width']:.3f} | "
                         f"[{lo:.3f}, {hi:.3f}] | {s['cluster_ci_half_width'] / s['naive_ci_half_width']:.2f} | "
                         f"{s['instance_mean']:.3f} ± {s['instance_ci_half_width']:.3f} |")
    lines += ["", "## 2. Metric definitions", "",
              "| Explainer / stress test | J@5 absolute | J@5 signed | AD (%) shared words | AD (%) union |",
              "|---|---|---|---|---|"]
    for name, e in combos.items():
        lines.append(f"| {name} | {e['jaccard_at_5_absolute']:.3f} | {e['jaccard_at_5_signed']:.3f} | "
                     f"{e['ad_intersection']:.2f} | {e['ad_union']:.2f} |")
    lines += ["", "## 3. Perturbation strength: surface words vs. model tokens", "",
              "Spearman correlation between strength and stability (negative: stronger perturbations, less stable).", "",
              "| Explainer / stress test | Mean word edit | Mean token edit | word vs J@5 | token vs J@5 | word vs ρ | token vs ρ |",
              "|---|---|---|---|---|---|---|"]
    for name, e in combos.items():
        s = e.get('strength')
        if s:
            lines.append(f"| {name} | {s['mean_word_edit_fraction']:.3f} | {s['mean_token_edit_fraction']:.3f} | "
                         f"{s['word_edit_fraction_vs_jaccard_at_5']:.2f} | {s['token_edit_fraction_vs_jaccard_at_5']:.2f} | "
                         f"{s['word_edit_fraction_vs_spearman_correlation']:.2f} | {s['token_edit_fraction_vs_spearman_correlation']:.2f} |")
    lines += ["", "## 4. Explainer differences (J@5, paired over instances)", "",
              "| Comparison | Mean difference | 95% CI | n | Distinguishable |", "|---|---|---|---|---|"]
    for name, d in result['explainer_differences'].items():
        lines.append(f"| {name} | {d['mean_difference']:+.3f} | [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | "
                     f"{d['n_instances']} | {'yes' if d['distinguishable'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiment', choices=sorted(RUNNER_NAMES))
    args = parser.parse_args(argv)
    result = build(args.experiment)
    out = REPO_ROOT / 'results' / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.experiment}_measurement.json').write_text(json.dumps(result, indent=2))
    (out / f'{args.experiment}_measurement.md').write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == '__main__':
    main()
