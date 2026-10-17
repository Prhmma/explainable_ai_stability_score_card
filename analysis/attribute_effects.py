"""
Effects of single estimand attributes relative to the primary estimand.

The primary estimand (per explainer, model and stress test) uses comparisons with
non-zero token dose, explains every variant for the original's predicted class
(prediction-changing variants re-explained, see analysis.target_class),
normalised word keys, J@5 on absolute attributions, and the mean over instances
of each instance's mean comparison.

Each contrast changes one attribute and reports the paired, instance-level
difference in J@5 with a cluster-bootstrap 95% interval:
- zero_dose_included: zero-dose variants added back;
- own_class_target: prediction-changing variants explained for their own class;
- k3 / k10: J@3 or J@10 instead of J@5;
- signed: top-5 by signed instead of absolute attribution;
- comparison_mean: mean over comparisons instead of over instances;
- misclassified_included: misclassified instances added to the population;
- string_keys, gradient_config, fused_attention: the implementation-sensitivity
  runs (subsets), compared variant by variant with the main runs.
Metric contrasts (k, signed) use prediction-invariant comparisons, where the
target rules coincide.

Usage (from the repository root):
    uv run python -m analysis.attribute_effects experiment_1 [experiment_2 ...]
Writes results/analysis/attribute_effects.csv (all experiments given).
"""

import argparse
import csv
import json
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from analysis.measurement import jaccard_at_k, levenshtein, make_tokenizer
from analysis.report import REPO_ROOT
from analysis.sensitivity import _base_name, _config_dir
from analysis.stability import load_latest_records, load_per_instance

TESTS = ('PreprocessingVariation', 'SemanticParaphrasing')
CONTRASTS = ['zero_dose_included', 'own_class_target', 'string_keys', 'k3', 'k10', 'signed',
             'comparison_mean', 'misclassified_included', 'gradient_config', 'fused_attention']
B = 2000


def _compared(r):
    indices = r.get('compared_variant_indices') or range(1, len(r['variant_explanations']))
    return list(zip(indices, r['metrics']))


def _bootstrap(diffs: List[float], rng) -> Optional[Dict]:
    d = np.array([x for x in diffs if x is not None and not np.isnan(x)])
    if d.size < 5:
        return None
    boot = d[rng.integers(0, d.size, (B, d.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'delta': float(d.mean()), 'lo': float(lo), 'hi': float(hi), 'n': int(d.size)}


class Doser:
    def __init__(self, experiment_key):
        self.tokenize = make_tokenizer(experiment_key)
        self.tokens, self.cache = {}, {}

    def __call__(self, original: str, variant: str) -> float:
        key = (original, variant)
        if key not in self.cache:
            a = self.tokens.setdefault(original, self.tokenize(original))
            b = self.tokens.setdefault(variant, self.tokenize(variant))
            self.cache[key] = levenshtein(a, b) / max(len(a), 1)
        return self.cache[key]


def _target_map(experiment_key, explainer, test):
    path = REPO_ROOT / 'results' / 'target_class' / experiment_key / f'{explainer}_{test}.jsonl'
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                e = json.loads(line)
                out[(e['sample_idx'], e['variant_idx'])] = e['original_target']['jaccard_at_5']
            except json.JSONDecodeError:
                continue
    return out


def _comparisons(r, dose, target):
    """Per comparison: (dose, invariant, own J@5, primary J@5 or NaN, metrics, variant index)."""
    out = []
    for i, m in _compared(r):
        d = dose(r['original_text'], r['variant_texts'][i])
        inv = m['comparison_info']['same_prediction']
        primary = m['jaccard_at_5'] if inv else target.get((r['sample_idx'], i), np.nan)
        out.append((d, inv, m['jaccard_at_5'], primary, m, i))
    return out


def _mean(xs):
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.mean(xs)) if xs else np.nan


def effects_for(experiment_key: str, explainer: str, test: str, rows, dose, rng) -> Dict[str, Dict]:
    target = _target_map(experiment_key, explainer, test)
    per = {r['sample_idx']: (r, _comparisons(r, dose, target)) for r in rows}
    primary = {s: _mean([c[3] for c in cs if c[0] > 0]) for s, (_, cs) in per.items()}
    out = {}

    diffs = []
    for s, (_, cs) in per.items():
        allv = [c[3] if c[0] > 0 else c[2] for c in cs]
        diffs.append(_mean(allv) - primary[s] if not np.isnan(primary[s]) else np.nan)
    out['zero_dose_included'] = _bootstrap(diffs, rng)

    if target:
        out['own_class_target'] = _bootstrap(
            [_mean([c[2] for c in cs if c[0] > 0]) - primary[s] for s, (_, cs) in per.items()], rng)

    for name, field in (('k3', 'jaccard_at_3'), ('k10', 'jaccard_at_10')):
        out[name] = _bootstrap([_mean([c[4][field] for c in cs if c[0] > 0 and c[1]]) -
                                _mean([c[2] for c in cs if c[0] > 0 and c[1]]) for _, (_, cs) in per.items()], rng)

    sd = []
    for _, (r, cs) in per.items():
        o = r['original_explanation']['attributions']
        vals = [jaccard_at_k(o, r['variant_explanations'][c[5]]['attributions'], signed=True) - c[2]
                for c in cs if c[0] > 0 and c[1]]
        sd.append(_mean(vals))
    out['signed'] = _bootstrap(sd, rng)

    # Comparison- vs instance-level mean: resample instances, recompute both
    groups = [np.array([c[3] for c in cs if c[0] > 0 and not np.isnan(c[3])]) for _, (_, cs) in per.items()]
    groups = [g for g in groups if g.size]
    if len(groups) >= 5:
        def stat(gs):
            return np.concatenate(gs).mean() - np.mean([g.mean() for g in gs])
        boots = [stat([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(500)]
        out['comparison_mean'] = {'delta': float(stat(groups)), 'lo': float(np.percentile(boots, 2.5)),
                                  'hi': float(np.percentile(boots, 97.5)), 'n': len(groups)}
    return out


def config_effect(experiment_key, config, explainer, test, main_rows, dose, rng):
    """Variant-by-variant J@5 difference (config run minus main run) on non-zero-dose, invariant comparisons."""
    try:
        cfg = load_per_instance(_config_dir(config, experiment_key))
    except FileNotFoundError:
        return None
    rows = next((v for (e, t), v in cfg.items() if _base_name(e) == explainer and t == test), None)
    if not rows:
        return None
    main = {r['original_text']: r for r in main_rows}
    diffs = []
    for r in rows:
        m0 = main.get(r['original_text'])
        if m0 is None:
            continue
        base = {m0['variant_texts'][i]: m for i, m in _compared(m0)}
        vals = []
        for i, m in _compared(r):
            v = r['variant_texts'][i]
            if v in base and dose(r['original_text'], v) > 0 and m['comparison_info']['same_prediction'] \
                    and base[v]['comparison_info']['same_prediction']:
                vals.append(m['jaccard_at_5'] - base[v]['jaccard_at_5'])
        diffs.append(_mean(vals))
    return _bootstrap(diffs, rng)


def misclassified_effect(experiment_key, explainer, test, main_rows, dose, rng):
    try:
        mis = load_per_instance(_config_dir('misclassified', experiment_key)).get((explainer, test))
    except FileNotFoundError:
        return None
    if not mis:
        return None

    def inst_means(rows):
        out = []
        for r in rows:
            vals = [m['jaccard_at_5'] for i, m in _compared(r)
                    if m['comparison_info']['same_prediction'] and dose(r['original_text'], r['variant_texts'][i]) > 0]
            if vals:
                out.append(np.mean(vals))
        return np.array(out)

    a, b = inst_means(main_rows), inst_means(mis)
    if a.size < 5 or b.size < 5:
        return None
    stat = lambda x, y: np.concatenate([x, y]).mean() - x.mean()
    boots = [stat(a[rng.integers(0, a.size, a.size)], b[rng.integers(0, b.size, b.size)]) for _ in range(B)]
    return {'delta': float(stat(a, b)), 'lo': float(np.percentile(boots, 2.5)),
            'hi': float(np.percentile(boots, 97.5)), 'n': int(a.size + b.size)}


def build(experiment_key: str, rng) -> List[Dict]:
    records = load_latest_records(experiment_key)
    dose = Doser(experiment_key)
    rows_out = []
    for (explainer, test), rows in sorted(records.items()):
        if test not in TESTS:
            continue
        effects = effects_for(experiment_key, explainer, test, rows, dose, rng)
        effects['string_keys'] = config_effect(experiment_key, 'legacy_alignment', explainer, test, rows, dose, rng)
        effects['fused_attention'] = config_effect(experiment_key, 'sdpa_attention', explainer, test, rows, dose, rng)
        if explainer == 'IntegratedGradients':
            effects['gradient_config'] = config_effect(experiment_key, 'legacy_ig_config', explainer, test, rows, dose, rng)
        effects['misclassified_included'] = misclassified_effect(experiment_key, explainer, test, rows, dose, rng)
        for contrast in CONTRASTS:
            e = effects.get(contrast)
            if e:
                rows_out.append({'experiment': experiment_key, 'explainer': explainer, 'test': test,
                                 'contrast': contrast, **e})
    return rows_out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiments', nargs='+')
    args = parser.parse_args(argv)
    rng = np.random.default_rng(0)
    rows = [row for e in args.experiments for row in build(e, rng)]
    out = REPO_ROOT / 'results' / 'analysis' / 'attribute_effects.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'explainer', 'test', 'contrast', 'delta', 'lo', 'hi', 'n'])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == '__main__':
    main()
