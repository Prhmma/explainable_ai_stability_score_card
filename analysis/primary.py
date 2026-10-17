"""
Stability under the primary estimand, with paired differences between methods.

Primary estimand (per model, attribution method and stress test):
- comparisons with non-zero token-level dose (zero-dose variants are identical
  model inputs and are reported separately); seed variation is kept as is;
- every variant explained for the original's predicted class: prediction-
  changing variants use the re-explanations from analysis.target_class;
- normalised word keys, with positional correspondence for substituted words:
  in a single-word substitution the substitute takes the key of the word it
  replaces, so an attribution that moves to the synonym is not counted as lost;
- J@5 on absolute attributions and RBO (p = 0.9), with Spearman rho secondary;
- the mean over instances of each instance's mean comparison, with 95%
  cluster-bootstrap intervals; prediction retention is the share of
  comparisons whose prediction is unchanged.
Paired differences between methods are instance-level, with 95% and 90%
intervals; a difference is negligible when its 90% interval lies within the
smallest effect size of interest (|dJ@5| < 0.02).

Usage (from the repository root):
    uv run python -m analysis.primary experiment_1 experiment_2 experiment_3 experiment_4 [--tex FILE]
Writes results/analysis/primary.{json,md} and, with --tex, a LaTeX table.
"""

import argparse
import json
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

from analysis.attribute_effects import Doser
from analysis.measurement import jaccard_at_k
from analysis.report import REPO_ROOT
from analysis.robustness import ranking, rbo_ext
from analysis.stability import load_latest_records
from stability_framework.explainers.base import BaseExplainer

METHODS = ('IntegratedGradients', 'LIME', 'SHAP')
SHORT = {'IntegratedGradients': 'IG', 'LIME': 'LIME', 'SHAP': 'SHAP'}
TESTS = ('SeedVariation', 'PreprocessingVariation', 'SemanticParaphrasing')
SESOI = 0.02
B = 2000


def substitution(original: str, variant: str):
    """(index, original token, variant token) if the variant replaces exactly one whitespace token."""
    a, b = original.split(), variant.split()
    if len(a) != len(b):
        return None
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return (diff[0], a[diff[0]], b[diff[0]]) if len(diff) == 1 else None


def positional_attributions(original: str, variant: str, words) -> Dict[str, float]:
    """
    Keyed attributions of a variant, with a substituted word keyed as the word it replaces.

    `words` is the explanation's ordered (word, score) list over the variant's
    whitespace tokens (tokens without attribution may be absent).
    """
    words = [tuple(w) for w in words]
    sub = substitution(original, variant)
    if sub is not None:
        index, old, new = sub
        tokens = variant.split()
        p = 0
        for j, token in enumerate(tokens):
            if p < len(words) and words[p][0] == token:
                if j == index:
                    words[p] = (old, words[p][1])
                    break
                p += 1
    return BaseExplainer._key_attributions(words)


def _spearman(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return float('nan')
    x, y = [a[k] for k in common], [b[k] for k in common]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float('nan')
    return float(spearmanr(x, y).correlation)


def _target_entries(experiment_key, explainer, test):
    path = REPO_ROOT / 'results' / 'target_class' / experiment_key / f'{explainer}_{test}.jsonl'
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                e = json.loads(line)
                out[(e['sample_idx'], e['variant_idx'])] = e
            except json.JSONDecodeError:
                continue
    return out


def primary_comparisons(experiment_key, explainer, test, rows, dose) -> Dict[int, List[Dict]]:
    """Per instance: list of {j5, rbo, rho, invariant, variant, dose} under the primary estimand."""
    target = _target_entries(experiment_key, explainer, test)
    out = {}
    for r in rows:
        original = r['original_explanation']['attributions']
        orig_rank = ranking(original)
        indices = r.get('compared_variant_indices') or range(1, len(r['variant_explanations']))
        comps = []
        for i, m in zip(indices, r['metrics']):
            variant = r['variant_texts'][i]
            d = 0.0 if test == 'SeedVariation' else dose(r['original_text'], variant)
            if test != 'SeedVariation' and d == 0:
                continue
            invariant = m['comparison_info']['same_prediction']
            if invariant:
                words = r['variant_explanations'][i]['words']
            else:
                entry = target.get((r['sample_idx'], i))
                if entry is None:
                    continue
                words = entry.get('original_target_words')
                if words is None:  # older re-explanations stored metrics only
                    comps.append({'j5': entry['original_target']['jaccard_at_5'], 'rbo': float('nan'),
                                  'rho': entry['original_target']['spearman_correlation'], 'invariant': False,
                                  'variant': variant, 'dose': d})
                    continue
            a = (positional_attributions(r['original_text'], variant, words) if test == 'SemanticParaphrasing'
                 else BaseExplainer._key_attributions([tuple(w) for w in words]))
            comps.append({'j5': jaccard_at_k(original, a), 'rbo': rbo_ext(orig_rank, ranking(a)),
                          'rho': _spearman(original, a), 'invariant': invariant, 'variant': variant, 'dose': d})
        out[r['sample_idx']] = comps
    return out


def _instance_means(per: Dict[int, List[Dict]], field: str) -> Dict[int, float]:
    out = {}
    for s, comps in per.items():
        vals = [c[field] for c in comps if not np.isnan(c[field])]
        if vals:
            out[s] = float(np.mean(vals))
    return out


def _ci(values: np.ndarray, rng, level=95):
    boot = values[rng.integers(0, values.size, (B, values.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


def summarise(per, rng) -> Dict:
    out = {'n_instances': sum(1 for c in per.values() if c), 'n_comparisons': sum(len(c) for c in per.values())}
    allc = [c for cs in per.values() for c in cs]
    out['retention'] = float(np.mean([c['invariant'] for c in allc])) if allc else float('nan')
    for field in ('j5', 'rbo', 'rho'):
        v = np.array(list(_instance_means(per, field).values()))
        if v.size >= 5:
            lo, hi = _ci(v, rng)
            out[field] = {'mean': float(v.mean()), 'lo': lo, 'hi': hi}
    return out


def paired(per_a, per_b, rng) -> Optional[Dict]:
    a, b = _instance_means(per_a, 'j5'), _instance_means(per_b, 'j5')
    shared = sorted(set(a) & set(b))
    if len(shared) < 5:
        return None
    d = np.array([a[s] - b[s] for s in shared])
    lo95, hi95 = _ci(d, rng, 95)
    lo90, hi90 = _ci(d, rng, 90)
    return {'mean': float(d.mean()), 'ci95': [lo95, hi95], 'ci90': [lo90, hi90], 'n': len(shared),
            'different': bool(lo95 > 0 or hi95 < 0), 'negligible': bool(lo90 > -SESOI and hi90 < SESOI)}


def build(experiments, rng) -> Dict:
    result = {'stability': {}, 'differences': {}}
    for e in experiments:
        records = load_latest_records(e)
        dose = Doser(e)
        per_all = {}
        for test in TESTS:
            for method in METHODS:
                rows = records.get((method, test))
                if not rows:
                    continue
                per = primary_comparisons(e, method, test, rows, dose)
                per_all[(method, test)] = per
                result['stability'][f'{e}|{method}|{test}'] = summarise(per, rng)
            for a, b in combinations(METHODS, 2):
                if (a, test) in per_all and (b, test) in per_all:
                    d = paired(per_all[(a, test)], per_all[(b, test)], rng)
                    if d:
                        result['differences'][f'{e}|{test}|{SHORT[a]}-{SHORT[b]}'] = d
    return result


def _fmt(s, field):
    x = s.get(field)
    return f"{x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]" if x else '–'


def to_markdown(result) -> str:
    lines = ["# Stability under the primary estimand", "",
             "| Experiment | Method | Test | J@5 | RBO | ρ | Retention | Instances | Comparisons |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, s in result['stability'].items():
        e, m, t = key.split('|')
        lines.append(f"| {e} | {SHORT[m]} | {t} | {_fmt(s, 'j5')} | {_fmt(s, 'rbo')} | {_fmt(s, 'rho')} | "
                     f"{s['retention']:.1%} | {s['n_instances']} | {s['n_comparisons']} |")
    lines += ["", f"## Paired differences in J@5 (negligible: 90% CI within ±{SESOI})", "",
              "| Experiment | Test | Pair | Mean | 95% CI | Different | Negligible |", "|---|---|---|---|---|---|---|"]
    for key, d in result['differences'].items():
        e, t, pair = key.split('|')
        lines.append(f"| {e} | {t} | {pair} | {d['mean']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | "
                     f"{'yes' if d['different'] else 'no'} | {'yes' if d['negligible'] else 'no'} |")
    return "\n".join(lines) + "\n"


def to_latex(result) -> str:
    """Rows: experiment x method; columns: J@5 and RBO per stress test, and retention for perturbation tests."""
    names = {'experiment_1': 'E1', 'experiment_2': 'E2', 'experiment_3': 'E3', 'experiment_4': 'E4'}
    cell = lambda s, f: f"{s[f]['mean']:.2f}" if s and s.get(f) else '--'
    lines = [r"\begin{tabular}{@{}l l cc cc cc c@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Seed} & \multicolumn{2}{c}{Preprocessing} & \multicolumn{2}{c}{Paraphrasing} & Retention \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Exp. & Method & J@5 & RBO & J@5 & RBO & J@5 & RBO & (pre./para.) \\", r"\midrule"]
    for e in names:
        for m in METHODS:
            get = lambda t: result['stability'].get(f'{e}|{m}|{t}')
            sd, pp, pa = get('SeedVariation'), get('PreprocessingVariation'), get('SemanticParaphrasing')
            if not (pp or pa):
                continue
            ret = '/'.join(f"{x['retention'] * 100:.0f}" if x else '--' for x in (pp, pa))
            lines.append(f"{names[e]} & {SHORT[m]} & {cell(sd, 'j5')} & {cell(sd, 'rbo')} & {cell(pp, 'j5')} & "
                         f"{cell(pp, 'rbo')} & {cell(pa, 'j5')} & {cell(pa, 'rbo')} & {ret} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiments', nargs='+')
    parser.add_argument('--tex', help='also write the LaTeX table to this path')
    args = parser.parse_args(argv)
    result = build(args.experiments, np.random.default_rng(0))
    out = REPO_ROOT / 'results' / 'analysis'
    (out / 'primary.json').write_text(json.dumps(result, indent=2))
    (out / 'primary.md').write_text(to_markdown(result))
    if args.tex:
        with open(args.tex, 'w') as f:
            f.write(to_latex(result))
    print(to_markdown(result))


if __name__ == '__main__':
    main()
