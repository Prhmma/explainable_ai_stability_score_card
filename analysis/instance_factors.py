"""
Instance-level factors of explanation stability.

Tests candidate claims on the per-instance records; one instance-level value is
the mean over its comparisons:
- input dependence: do the explainers find the same instances unstable
  (Spearman between explainers' instance-level J@5 under the same stress test)?
- model confidence, text length and the top-k margin of the original
  explanation (gap between the 5th and 6th largest |attribution|, relative to
  the largest) as predictors of instance-level J@5, each also as a partial
  correlation controlling for length;
- concentration: share of instances with J@5 < 0.5, and share of all
  instability (1 − J@5) carried by the least stable 10% of instances;
- within vs. cross-method: share of instances whose within-method J@5 exceeds
  the agreement of their original explanations with another explainer;
- stability vs. faithfulness per instance (comprehensiveness; needs the model,
  cached in results/analysis/<experiment>_faithfulness_per_instance.json).

Usage (from the repository root):
    uv run python -m analysis.instance_factors experiment_1 [--device mps] [--no-faithfulness]
"""

import argparse
import json
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr

from analysis.measurement import jaccard_at_k
from analysis.report import REPO_ROOT, RUNNER_NAMES, latest_instance_dir
from analysis.stability import load_latest_records, load_per_instance

TESTS = ('PreprocessingVariation', 'SemanticParaphrasing', 'SeedVariation')


def instance_table(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    """Per instance: mean J@5 and ρ over comparisons, confidence, length, top-5 margin."""
    out = {}
    for r in rows:
        if not r['metrics']:
            continue
        a = np.sort(np.abs(np.array(list(r['original_explanation']['attributions'].values()), dtype=float)))[::-1]
        margin = (a[4] - a[5]) / a[0] if a.size > 5 and a[0] > 0 else float('nan')
        out[r['sample_idx']] = {
            'jaccard_at_5': float(np.nanmean([m['jaccard_at_5'] for m in r['metrics']])),
            'spearman_correlation': float(np.nanmean([m['spearman_correlation'] for m in r['metrics']]))
            if any(not np.isnan(m['spearman_correlation']) for m in r['metrics']) else float('nan'),
            'confidence': r['original_explanation']['confidence'],
            'length': len(r['original_text'].split()),
            'top5_margin': float(margin),
        }
    return out


def _spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    if keep.sum() < 3 or len(set(x[keep])) < 2 or len(set(y[keep])) < 2:
        return float('nan')
    return float(spearmanr(x[keep], y[keep]).correlation)


def _partial(x, y, z) -> float:
    """Spearman correlation of x and y controlling for z."""
    rxy, rxz, ryz = _spearman(x, y), _spearman(x, z), _spearman(y, z)
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / denom) if denom > 0 else float('nan')


def predictors(table: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    rows = list(table.values())
    col = lambda k: [r[k] for r in rows]
    j, length = col('jaccard_at_5'), col('length')
    return {
        'n': len(rows),
        'confidence_vs_j5': _spearman(col('confidence'), j),
        'confidence_vs_j5_given_length': _partial(col('confidence'), j, length),
        'length_vs_j5': _spearman(length, j),
        'margin_vs_j5': _spearman(col('top5_margin'), j),
        'margin_vs_j5_given_length': _partial(col('top5_margin'), j, length),
        'share_j5_below_0.5': float(np.mean(np.array(j) < 0.5)),
        'instability_share_worst_10pct': float(np.sort(1 - np.array(j))[::-1][:max(1, len(j) // 10)].sum()
                                               / max((1 - np.array(j)).sum(), 1e-12)),
    }


def build(experiment_key: str, device: str = 'cpu', with_faithfulness: bool = True) -> Dict[str, Any]:
    records = load_latest_records(experiment_key)
    tables = {key: instance_table(rows) for key, rows in records.items()}
    out = {'experiment': experiment_key, 'predictors': {}, 'input_dependence': {}, 'within_vs_cross': {},
           'stability_vs_faithfulness': {}}

    for (explainer, test), table in tables.items():
        out['predictors'][f'{explainer}/{test}'] = predictors(table)

    for test in TESTS:
        names = sorted(e for e, t in tables if t == test)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = sorted(set(tables[(a, test)]) & set(tables[(b, test)]))
                out['input_dependence'][f'{test}: {a} vs {b}'] = {
                    'n': len(shared),
                    'spearman_j5': _spearman([tables[(a, test)][s]['jaccard_at_5'] for s in shared],
                                             [tables[(b, test)][s]['jaccard_at_5'] for s in shared])}

    originals = {e: {r['sample_idx']: r['original_explanation']['attributions'] for r in rows}
                 for (e, t), rows in records.items() if t == 'PreprocessingVariation'}
    for (explainer, test), table in tables.items():
        if test == 'SeedVariation':
            continue
        for other in sorted(originals):
            if other == explainer:
                continue
            shared = sorted(set(table) & set(originals[other]) & set(originals[explainer]))
            within = np.array([table[s]['jaccard_at_5'] for s in shared])
            cross = np.array([jaccard_at_k(originals[explainer][s], originals[other][s]) for s in shared])
            out['within_vs_cross'][f'{explainer}/{test} vs agreement with {other}'] = {
                'n': len(shared), 'within_mean': float(within.mean()), 'cross_mean': float(cross.mean()),
                'share_within_greater': float(np.mean(within > cross))}

    if with_faithfulness:
        cache = REPO_ROOT / 'results' / 'analysis' / f'{experiment_key}_faithfulness_per_instance.json'
        if cache.exists():
            faith = json.loads(cache.read_text())
        else:
            from analysis.faithfulness import faithfulness, make_predict_proba
            from stability_framework.model_loader import EXPERIMENT_MODELS, ModelLoader
            model, tokenizer = ModelLoader(device=device).load_model_and_tokenizer(
                EXPERIMENT_MODELS[experiment_key]['name'])
            predict = make_predict_proba(model, tokenizer)
            faith = {e: faithfulness(rows, predict, per_instance=True)['per_instance']['comprehensiveness']
                     for (e, t), rows in records.items() if t == 'PreprocessingVariation'}
            cache.write_text(json.dumps(faith))
        for (explainer, test), table in tables.items():
            comp = {int(k): v for k, v in faith.get(explainer, {}).items()}
            shared = sorted(set(table) & set(comp))
            j = [table[s]['jaccard_at_5'] for s in shared]
            c = [comp[s] for s in shared]
            out['stability_vs_faithfulness'][f'{explainer}/{test}'] = {
                'n': len(shared), 'spearman': _spearman(j, c),
                'spearman_given_length': _partial(j, c, [table[s]['length'] for s in shared])}
    return out


def to_markdown(result: Dict[str, Any]) -> str:
    f = lambda v: 'n/a' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:+.2f}'
    lines = [f"# Instance-level factors: {result['experiment']}", "",
             "## 1. Predictors of instance-level J@5 (Spearman; 'given len.' = partial, controlling for length)", "",
             "| Explainer / stress test | n | confidence | given len. | length | top-5 margin | given len. | J@5 < 0.5 | worst 10% share |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, p in result['predictors'].items():
        lines.append(f"| {k} | {p['n']} | {f(p['confidence_vs_j5'])} | {f(p['confidence_vs_j5_given_length'])} | "
                     f"{f(p['length_vs_j5'])} | {f(p['margin_vs_j5'])} | {f(p['margin_vs_j5_given_length'])} | "
                     f"{p['share_j5_below_0.5']:.1%} | {p['instability_share_worst_10pct']:.1%} |")
    lines += ["", "## 2. Input dependence: do explainers find the same instances unstable?", "",
              "| Pair | n | Spearman (instance J@5) |", "|---|---|---|"]
    for k, d in result['input_dependence'].items():
        lines.append(f"| {k} | {d['n']} | {f(d['spearman_j5'])} |")
    lines += ["", "## 3. Within-method stability vs. cross-method agreement, per instance", "",
              "| Comparison | n | Within J@5 | Cross J@5 | Instances with within > cross |", "|---|---|---|---|---|"]
    for k, d in result['within_vs_cross'].items():
        lines.append(f"| {k} | {d['n']} | {d['within_mean']:.3f} | {d['cross_mean']:.3f} | {d['share_within_greater']:.1%} |")
    if result['stability_vs_faithfulness']:
        lines += ["", "## 4. Stability vs. faithfulness per instance (J@5 vs comprehensiveness)", "",
                  "| Explainer / stress test | n | Spearman | given length |", "|---|---|---|---|"]
        for k, d in result['stability_vs_faithfulness'].items():
            lines.append(f"| {k} | {d['n']} | {f(d['spearman'])} | {f(d['spearman_given_length'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiment', choices=sorted(RUNNER_NAMES))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--no-faithfulness', action='store_true')
    args = parser.parse_args(argv)
    result = build(args.experiment, args.device, not args.no_faithfulness)
    out = REPO_ROOT / 'results' / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.experiment}_instance_factors.json').write_text(json.dumps(result, indent=2))
    (out / f'{args.experiment}_instance_factors.md').write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == '__main__':
    main()
