"""
Robustness checks for the claims of the extended analysis.

From the per-instance records:
- negative rank correlations by prediction invariance: how many comparisons
  with ρ < 0 change the predicted class;
- explainer rankings under J@3, J@5 and J@10, with paired instance-level
  intervals;
- rank-biased overlap (RBO, p = 0.9, extrapolated) over the full |attribution|
  ranking, next to J@5: explainer means, and whether the top-5 boundary margin
  still predicts instance-level stability;
- per-comparison perturbation strength (word- and token-level) with stability,
  exported for figures (results/analysis/figdata/<experiment>_strength.csv.gz).

With the model (--device):
- logit margin (top-1 minus top-2 logit) of each original prediction, and the
  LIME seed-variation partial correlations of P4.4 recomputed with it;
- faithfulness with removed words replaced by the tokenizer's mask token
  (unknown token if there is none) instead of deleted.

Usage (from the repository root):
    uv run python -m analysis.robustness experiment_1 [--device mps] [--no-model]
"""

import argparse
import csv
import gzip
import json
from typing import Any, Dict, List, Sequence

import numpy as np

from analysis.instance_factors import _partial, _spearman, instance_table
from analysis.measurement import explainer_differences, levenshtein, make_tokenizer
from analysis.report import REPO_ROOT, RUNNER_NAMES, latest_instance_dir
from analysis.stability import load_latest_records, load_per_instance
from analysis.strength import word_edit_fraction

RBO_P = 0.9


def rbo_ext(a: Sequence[str], b: Sequence[str], p: float = RBO_P) -> float:
    """Extrapolated rank-biased overlap of two rankings of possibly different length (Webber et al., 2010)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    S, L = (a, b) if len(a) <= len(b) else (b, a)
    s, l = len(S), len(L)
    seen_s, seen_l, x = set(), set(), []  # x[d-1] = |S[:min(d, s)] ∩ L[:d]|
    for d in range(1, l + 1):
        if d <= s:
            seen_s.add(S[d - 1])
        seen_l.add(L[d - 1])
        x.append(len(seen_s & seen_l))
    x_s, x_l = x[s - 1], x[l - 1]
    total = sum(x[d - 1] / d * p ** d for d in range(1, l + 1))
    total += sum(x_s * (d - s) / (s * d) * p ** d for d in range(s + 1, l + 1))
    return float((1 - p) / p * total + ((x_l - x_s) / l + x_s / s) * p ** l)


def ranking(attributions: Dict[str, float]) -> List[str]:
    return sorted(attributions, key=lambda w: abs(attributions[w]), reverse=True)


def compared(record):
    indices = record.get('compared_variant_indices') or range(1, len(record['variant_explanations']))
    return [(i, m) for i, m in zip(indices, record['metrics'])]


def negative_rho_by_invariance(records) -> Dict[str, Any]:
    out = {}
    for (explainer, test), rows in records.items():
        ms = [m for r in rows for m in r['metrics'] if not np.isnan(m['spearman_correlation'])]
        neg = [m for m in ms if m['spearman_correlation'] < 0]
        flips = [m for m in ms if not m['comparison_info']['same_prediction']]
        out[f'{explainer}/{test}'] = {
            'comparisons': len(ms), 'negative_rho': len(neg),
            'negative_rho_with_prediction_change': sum(not m['comparison_info']['same_prediction'] for m in neg),
            'share_negative_among_changed': (sum(m['spearman_correlation'] < 0 for m in flips) / len(flips)) if flips else float('nan'),
            'share_negative_among_invariant': (sum(m['spearman_correlation'] < 0 for m in ms if m['comparison_info']['same_prediction'])
                                               / max(len(ms) - len(flips), 1)),
        }
    return out


def rbo_analysis(records) -> Dict[str, Any]:
    out = {}
    for (explainer, test), rows in records.items():
        table = instance_table(rows)
        per_instance = {}
        for r in rows:
            orig = ranking(r['original_explanation']['attributions'])
            vals = [rbo_ext(orig, ranking(r['variant_explanations'][i]['attributions'])) for i, _ in compared(r)]
            if vals:
                per_instance[r['sample_idx']] = float(np.mean(vals))
        shared = sorted(set(per_instance) & set(table))
        margin = [table[s]['top5_margin'] for s in shared]
        length = [table[s]['length'] for s in shared]
        rbo = [per_instance[s] for s in shared]
        j5 = [table[s]['jaccard_at_5'] for s in shared]
        out[f'{explainer}/{test}'] = {
            'rbo_mean': float(np.mean(rbo)), 'j5_mean': float(np.mean(j5)),
            'margin_vs_j5_given_length': _partial(margin, j5, length),
            'margin_vs_rbo_given_length': _partial(margin, rbo, length),
            'rbo_vs_j5': _spearman(rbo, j5),
        }
    return out


def ranking_tables(records, rng) -> Dict[str, Any]:
    out = {}
    for k in (3, 5, 10):
        means = {}
        for (explainer, test), rows in records.items():
            if test == 'SeedVariation':
                continue
            means.setdefault(test, {})[explainer] = float(np.nanmean(
                [m[f'jaccard_at_{k}'] for r in rows for m in r['metrics']]))
        out[f'J@{k}'] = {'means': means, 'differences': {
            name: d for name, d in explainer_differences(records, rng, f'jaccard_at_{k}').items()
            if not name.startswith('SeedVariation')}}
    return out


def strength_rows(experiment_key: str, records) -> List[Dict[str, Any]]:
    tokenize = make_tokenizer(experiment_key)
    tokens, cache, rows = {}, {}, []
    for (explainer, test), recs in records.items():
        if test == 'SeedVariation':
            continue
        for r in recs:
            for i, m in compared(r):
                variant = r['variant_texts'][i]
                key = (r['original_text'], variant)
                if key not in cache:
                    ids = [tokens.setdefault(t, tokenize(t)) for t in key]
                    cache[key] = (word_edit_fraction(*key), levenshtein(*ids) / max(len(ids[0]), 1))
                rows.append({'explainer': explainer, 'test': test, 'sample_idx': r['sample_idx'],
                             'word_edit': cache[key][0], 'token_edit': cache[key][1],
                             'jaccard_at_5': m['jaccard_at_5'], 'spearman': m['spearman_correlation'],
                             'same_prediction': int(m['comparison_info']['same_prediction'])})
    return rows


def model_checks(experiment_key: str, records, device: str) -> Dict[str, Any]:
    from analysis.faithfulness import faithfulness, make_predict_proba
    from stability_framework.model_loader import EXPERIMENT_MODELS, ModelLoader
    model, tokenizer = ModelLoader(device=device).load_model_and_tokenizer(EXPERIMENT_MODELS[experiment_key]['name'])
    out = {}

    rows = records[('LIME', 'SeedVariation')]
    logits = make_predict_proba(model, tokenizer, logits=True)([r['original_text'] for r in rows])
    top2 = np.sort(logits, axis=1)[:, -2:]
    margin = {r['sample_idx']: float(top2[n, 1] - top2[n, 0]) for n, r in enumerate(rows)}
    table = instance_table(rows)
    cache = REPO_ROOT / 'results' / 'analysis' / f'{experiment_key}_faithfulness_per_instance.json'
    comp = {int(k): v for k, v in json.loads(cache.read_text())['LIME'].items()} if cache.exists() else {}
    shared = sorted(set(table) & set(comp) & set(margin))
    j = [table[s]['jaccard_at_5'] for s in shared]
    c = [comp[s] for s in shared]
    lm = [margin[s] for s in shared]
    out['lime_seed_saturation'] = {
        'n': len(shared),
        'j5_vs_logit_margin': _spearman(j, lm),
        'j5_vs_logit_margin_given_comprehensiveness': _partial(j, lm, c),
        'j5_vs_comprehensiveness_given_logit_margin': _partial(j, c, lm),
        'logit_margin_vs_comprehensiveness': _spearman(lm, c),
        'tied_confidence_share': float(np.mean([table[s]['confidence'] >= 0.9999 for s in shared])),
    }

    token = getattr(tokenizer, 'mask_token', None) or getattr(tokenizer, 'unk_token', None)
    predict = make_predict_proba(model, tokenizer)
    out['faithfulness_replacement'] = {'token': token, 'explainers': {
        e: faithfulness(recs, predict, replacement=token)
        for (e, t), recs in records.items() if t == 'PreprocessingVariation'}}
    return out


def build(experiment_key: str, device: str = 'cpu', with_model: bool = True) -> Dict[str, Any]:
    records = load_latest_records(experiment_key)
    rng = np.random.default_rng(0)
    out = {'experiment': experiment_key,
           'negative_rho': negative_rho_by_invariance(records),
           'rankings_by_k': ranking_tables(records, rng),
           'rbo': rbo_analysis(records)}
    figdata = REPO_ROOT / 'results' / 'analysis' / 'figdata'
    figdata.mkdir(parents=True, exist_ok=True)
    rows = strength_rows(experiment_key, records)
    with gzip.open(figdata / f'{experiment_key}_strength.csv.gz', 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if with_model:
        out.update(model_checks(experiment_key, records, device))
    return out


def to_markdown(result: Dict[str, Any]) -> str:
    f = lambda v: 'n/a' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:+.2f}'
    lines = [f"# Robustness checks: {result['experiment']}", "",
             "## 1. Negative rank correlations and prediction changes", "",
             "| Explainer / stress test | Comparisons | ρ < 0 | of which prediction changed | P(ρ<0 \\| changed) | P(ρ<0 \\| invariant) |",
             "|---|---|---|---|---|---|"]
    for k, d in result['negative_rho'].items():
        lines.append(f"| {k} | {d['comparisons']} | {d['negative_rho']} | {d['negative_rho_with_prediction_change']} | "
                     f"{d['share_negative_among_changed']:.1%} | {d['share_negative_among_invariant']:.1%} |")
    lines += ["", "## 2. Explainer differences under J@3, J@5, J@10 (paired, instance level)", "",
              "| k | Comparison | Mean difference | 95% CI | Distinguishable |", "|---|---|---|---|---|"]
    for k, block in result['rankings_by_k'].items():
        for name, d in block['differences'].items():
            lines.append(f"| {k} | {name} | {d['mean_difference']:+.3f} | [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | "
                         f"{'yes' if d['distinguishable'] else 'no'} |")
    lines += ["", f"## 3. Rank-biased overlap (p = {RBO_P}) vs J@5", "",
              "| Explainer / stress test | RBO | J@5 | RBO vs J@5 (instances) | margin → J@5 (given len.) | margin → RBO (given len.) |",
              "|---|---|---|---|---|---|"]
    for k, d in result['rbo'].items():
        lines.append(f"| {k} | {d['rbo_mean']:.3f} | {d['j5_mean']:.3f} | {f(d['rbo_vs_j5'])} | "
                     f"{f(d['margin_vs_j5_given_length'])} | {f(d['margin_vs_rbo_given_length'])} |")
    if 'lime_seed_saturation' in result:
        d = result['lime_seed_saturation']
        lines += ["", "## 4. LIME seed variation and saturation (logit margin)", "",
                  f"n = {d['n']}; share of instances with confidence ≥ 0.9999 (ties in probability): {d['tied_confidence_share']:.0%}", "",
                  "| Quantity | Spearman |", "|---|---|",
                  f"| J@5 vs logit margin | {f(d['j5_vs_logit_margin'])} |",
                  f"| J@5 vs logit margin, given comprehensiveness | {f(d['j5_vs_logit_margin_given_comprehensiveness'])} |",
                  f"| J@5 vs comprehensiveness, given logit margin | {f(d['j5_vs_comprehensiveness_given_logit_margin'])} |",
                  f"| logit margin vs comprehensiveness | {f(d['logit_margin_vs_comprehensiveness'])} |"]
    if 'faithfulness_replacement' in result:
        fr = result['faithfulness_replacement']
        lines += ["", f"## 5. Faithfulness with removed words replaced by `{fr['token']}`", "",
                  "| Explainer | Comprehensiveness | (random) | Sufficiency | (random) | n |", "|---|---|---|---|---|---|"]
        for e, d in fr['explainers'].items():
            lines.append(f"| {e} | {d['explainer']['comprehensiveness']:.3f} | {d['random']['comprehensiveness']:.3f} | "
                         f"{d['explainer']['sufficiency']:.3f} | {d['random']['sufficiency']:.3f} | {d['explainer']['n']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiment', choices=sorted(RUNNER_NAMES))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--no-model', action='store_true')
    args = parser.parse_args(argv)
    result = build(args.experiment, args.device, not args.no_model)
    out = REPO_ROOT / 'results' / 'analysis'
    (out / f'{args.experiment}_robustness.json').write_text(json.dumps(result, indent=2))
    (out / f'{args.experiment}_robustness.md').write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == '__main__':
    main()
