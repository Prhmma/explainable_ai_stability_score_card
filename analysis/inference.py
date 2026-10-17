"""
Statistical inference for the extended analysis.

- Cluster-bootstrap 95% intervals (resampling instances) for the Spearman
  correlation between token-level perturbation strength and stability.
- The share of the prediction-invariant ρ gain recovered by explaining
  prediction-changing variants for the original class, with cluster-bootstrap
  intervals.
- Equivalence tests (TOST via 90% bootstrap intervals) against a smallest
  effect size of interest: |ΔJ@5| < 0.02 for explainer differences, and
  |ρ| < 0.1 for the per-instance association of stability with faithfulness.
- Chance-corrected J@5: (J − E[J]) / (1 − E[J]), where E[J] is the expected
  overlap of two random top-5 sets drawn from the explanation's n words.
- LIME seed reuse: share of preprocessing comparisons whose LIME explanation is
  bit-identical to the original, and LIME stability with those excluded.

Usage (from the repository root):
    uv run python -m analysis.inference experiment_1
    uv run python -m analysis.inference matched     # RoBERTa vs DistilBERT on shared AG News texts
"""

import argparse
import json
from math import comb
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr

from analysis.measurement import levenshtein, make_tokenizer
from analysis.report import REPO_ROOT, RUNNER_NAMES, latest_instance_dir
from analysis.stability import load_per_instance

B = 2000
SESOI_J5 = 0.02
SESOI_RHO = 0.1


def _compared(r):
    indices = r.get('compared_variant_indices') or range(1, len(r['variant_explanations']))
    return list(zip(indices, r['metrics']))


def _cluster_boot(groups: List[np.ndarray], stat, rng, b: int = B) -> List[float]:
    m = len(groups)
    vals = []
    for _ in range(b):
        idx = rng.integers(0, m, m)
        vals.append(stat([groups[i] for i in idx]))
    return vals


def expected_jaccard(n: int, k: int = 5) -> float:
    """E[|A∩B| / |A∪B|] for independent uniformly random k-subsets of n items."""
    k = min(k, n)
    if n == 0 or k == 0:
        return 1.0
    total = comb(n, k)
    return sum(comb(k, x) * comb(n - k, k - x) / total * x / (2 * k - x) for x in range(0, k + 1))


def strength_correlations(experiment_key: str, records, rng) -> Dict[str, Any]:
    tokenize = make_tokenizer(experiment_key)
    tokens, out = {}, {}
    for (explainer, test), rows in records.items():
        if test == 'SeedVariation':
            continue
        groups = []
        for r in rows:
            ids0 = tokens.setdefault(r['original_text'], tokenize(r['original_text']))
            g = []
            for i, m in _compared(r):
                v = r['variant_texts'][i]
                ids1 = tokens.setdefault(v, tokenize(v))
                g.append((levenshtein(ids0, ids1) / max(len(ids0), 1), m['jaccard_at_5'], m['spearman_correlation']))
            if g:
                groups.append(np.array(g, dtype=float))

        def corr(gs, col):
            a = np.concatenate(gs)
            keep = ~np.isnan(a[:, col])
            return spearmanr(a[keep, 0], a[keep, col]).correlation

        entry = {}
        for col, name in ((1, 'jaccard_at_5'), (2, 'spearman_correlation')):
            boots = _cluster_boot(groups, lambda gs: corr(gs, col), rng, b=500)
            entry[name] = {'rho': float(corr(groups, col)), 'ci': [float(np.nanpercentile(boots, 2.5)),
                                                                   float(np.nanpercentile(boots, 97.5))]}
        out[f'{explainer}/{test}'] = entry
    return out


def target_recovery(experiment_key: str, records, rng) -> Dict[str, Any]:
    out = {}
    base = REPO_ROOT / 'results' / 'target_class' / experiment_key
    for (explainer, test), rows in records.items():
        path = base / f'{explainer}_{test}.jsonl'
        if test == 'SeedVariation' or not path.exists():
            continue
        fixed = {}
        for line in path.read_text().splitlines():
            try:
                e = json.loads(line)
                fixed[(e['sample_idx'], e['variant_idx'])] = e['original_target']['spearman_correlation']
            except json.JSONDecodeError:
                continue
        groups = []  # per instance: rows of (own ρ, fixed ρ, invariant flag)
        for r in rows:
            g = []
            for i, m in _compared(r):
                own = m['spearman_correlation']
                inv = m['comparison_info']['same_prediction']
                g.append((own, own if inv else fixed.get((r['sample_idx'], i), np.nan), float(inv)))
            if g:
                groups.append(np.array(g, dtype=float))

        def share(gs):
            a = np.concatenate(gs)
            own, fix, inv = np.nanmean(a[:, 0]), np.nanmean(a[:, 1]), np.nanmean(a[a[:, 2] == 1, 0])
            return (fix - own) / (inv - own) if inv - own > 1e-9 else np.nan

        boots = _cluster_boot(groups, share, rng, b=1000)
        out[f'{explainer}/{test}'] = {'share': float(share(groups)),
                                      'ci': [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]}
    return out


def equivalence(experiment_key: str, records, rng) -> Dict[str, Any]:
    out = {'explainer_differences': {}, 'stability_faithfulness': {}}
    by_test: Dict[str, Dict[str, Dict[int, float]]] = {}
    for (explainer, test), rows in records.items():
        by_test.setdefault(test, {})[explainer] = {
            r['sample_idx']: float(np.nanmean([m['jaccard_at_5'] for m in r['metrics']])) for r in rows if r['metrics']}
    for test, per in by_test.items():
        names = sorted(per)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = sorted(set(per[a]) & set(per[b]))
                d = np.array([per[a][s] - per[b][s] for s in shared])
                boot = d[rng.integers(0, d.size, (B, d.size))].mean(axis=1)
                lo, hi = np.percentile(boot, [5, 95])
                out['explainer_differences'][f'{test}: {a} − {b}'] = {
                    'mean': float(d.mean()), 'ci90': [float(lo), float(hi)],
                    'equivalent': bool(lo > -SESOI_J5 and hi < SESOI_J5),
                    'different': bool(np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0)}
    cache = REPO_ROOT / 'results' / 'analysis' / f'{experiment_key}_faithfulness_per_instance.json'
    if cache.exists():
        faith = json.loads(cache.read_text())
        for (explainer, test), rows in records.items():
            comp = {int(k): v for k, v in faith.get(explainer, {}).items()}
            pairs = np.array([(float(np.nanmean([m['jaccard_at_5'] for m in r['metrics']])), comp[r['sample_idx']])
                              for r in rows if r['metrics'] and r['sample_idx'] in comp])
            rho = spearmanr(pairs[:, 0], pairs[:, 1]).correlation
            boots = [spearmanr(*pairs[rng.integers(0, len(pairs), len(pairs))].T).correlation for _ in range(500)]
            lo, hi = np.nanpercentile(boots, [5, 95])
            out['stability_faithfulness'][f'{explainer}/{test}'] = {
                'rho': float(rho), 'ci90': [float(lo), float(hi)],
                'equivalent_to_zero': bool(lo > -SESOI_RHO and hi < SESOI_RHO)}
    return out


def chance_corrected(records) -> Dict[str, Any]:
    out = {}
    cache: Dict[int, float] = {}
    for (explainer, test), rows in records.items():
        raw, corrected = [], []
        for r in rows:
            n = len(r['original_explanation']['attributions'])
            if n not in cache:
                cache[n] = expected_jaccard(n)
            e = cache[n]
            for _, m in _compared(r):
                raw.append(m['jaccard_at_5'])
                corrected.append((m['jaccard_at_5'] - e) / (1 - e) if e < 1 else np.nan)
        out[f'{explainer}/{test}'] = {'j5': float(np.mean(raw)), 'j5_chance_corrected': float(np.nanmean(corrected)),
                                      'mean_expected': float(np.mean([cache[len(r['original_explanation']['attributions'])] for r in rows]))}
    return out


def lime_seed_reuse(records) -> Dict[str, Any]:
    out = {}
    for test in ('PreprocessingVariation', 'SemanticParaphrasing'):
        rows = records.get(('LIME', test), [])
        same, kept = 0, []
        total = 0
        for r in rows:
            o = r['original_explanation']['attributions']
            for i, m in _compared(r):
                total += 1
                if r['variant_explanations'][i]['attributions'] == o:
                    same += 1
                else:
                    kept.append(m['jaccard_at_5'])
        seed = [m['jaccard_at_5'] for r in records.get(('LIME', 'SeedVariation'), []) for m in r['metrics']]
        out[test] = {'identical_share': same / total if total else float('nan'),
                     'j5_all': float(np.mean([m['jaccard_at_5'] for r in rows for m in r['metrics']])) if rows else float('nan'),
                     'j5_excluding_identical': float(np.mean(kept)) if kept else float('nan'),
                     'seed_floor_j5': float(np.mean(seed)) if seed else float('nan')}
    return out


def matched_agnews(rng) -> Dict[str, Any]:
    """RoBERTa (E3) vs DistilBERT (E4) on the shared texts and identical variants."""
    e3 = load_per_instance(latest_instance_dir('experiment_3'))
    e4 = load_per_instance(latest_instance_dir('experiment_4'))
    tok3, tok4 = make_tokenizer('experiment_3'), make_tokenizer('experiment_4')
    out = {}
    for explainer in ('IntegratedGradients', 'LIME', 'SHAP'):
        for test in ('PreprocessingVariation', 'SemanticParaphrasing'):
            a = {r['original_text']: r for r in e3[(explainer, test)]}
            b = {r['original_text']: r for r in e4[(explainer, test)]}
            rows = []  # (token edit E3, token edit E4, J5 E3, J5 E4)
            for text in set(a) & set(b):
                ra, rb = a[text], b[text]
                va = {ra['variant_texts'][i]: m['jaccard_at_5'] for i, m in _compared(ra)}
                vb = {rb['variant_texts'][i]: m['jaccard_at_5'] for i, m in _compared(rb)}
                ids3, ids4 = tok3(text), tok4(text)
                for v in set(va) & set(vb):
                    rows.append((levenshtein(ids3, tok3(v)) / max(len(ids3), 1),
                                 levenshtein(ids4, tok4(v)) / max(len(ids4), 1), va[v], vb[v]))
            arr = np.array(rows)
            # Stability as a function of each model's own token edit, on common bins
            edges = np.array([0, 1e-9, 0.05, 0.15, 0.3, 0.6, 10])
            bins = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m3 = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
                m4 = (arr[:, 1] >= lo) & (arr[:, 1] < hi)
                bins.append({'range': [float(lo), float(hi)], 'n_e3': int(m3.sum()), 'n_e4': int(m4.sum()),
                             'j5_e3': float(arr[m3, 2].mean()) if m3.any() else None,
                             'j5_e4': float(arr[m4, 3].mean()) if m4.any() else None})
            # Raw gap vs gap within the same token-edit bin (weighted by the smaller count)
            raw_gap = float(arr[:, 3].mean() - arr[:, 2].mean())
            w = [(min(x['n_e3'], x['n_e4']), x['j5_e4'] - x['j5_e3']) for x in bins
                 if x['j5_e3'] is not None and x['j5_e4'] is not None and min(x['n_e3'], x['n_e4']) >= 30]
            matched_gap = float(sum(n * g for n, g in w) / sum(n for n, _ in w)) if w else float('nan')
            out[f'{explainer}/{test}'] = {'comparisons': len(arr), 'mean_token_edit_e3': float(arr[:, 0].mean()),
                                          'mean_token_edit_e4': float(arr[:, 1].mean()), 'raw_gap_e4_minus_e3': raw_gap,
                                          'gap_within_token_edit_bins': matched_gap, 'bins': bins}
    return out


def to_markdown(result: Dict[str, Any]) -> str:
    if result.get('kind') == 'matched':
        lines = ["# RoBERTa (E3) vs DistilBERT (E4) on shared AG News texts and identical variants", "",
                 "Gap = E4 − E3 in mean J@5; within-bin gap compares the models at the same token-level edit "
                 "under each model's own tokenizer (bins with ≥ 30 comparisons per model).", "",
                 "| Explainer / stress test | Comparisons | Token edit E3 | Token edit E4 | Raw gap | Gap within token-edit bins |",
                 "|---|---|---|---|---|---|"]
        for k, d in result['combinations'].items():
            lines.append(f"| {k} | {d['comparisons']} | {d['mean_token_edit_e3']:.3f} | {d['mean_token_edit_e4']:.3f} | "
                         f"{d['raw_gap_e4_minus_e3']:+.3f} | {d['gap_within_token_edit_bins']:+.3f} |")
        return "\n".join(lines) + "\n"
    lines = [f"# Inference: {result['experiment']}", "", "## 1. Token-level strength vs stability (cluster-bootstrap 95% CI)", "",
             "| Explainer / stress test | vs J@5 | vs ρ |", "|---|---|---|"]
    for k, d in result['strength'].items():
        lines.append(f"| {k} | {d['jaccard_at_5']['rho']:+.2f} [{d['jaccard_at_5']['ci'][0]:+.2f}, {d['jaccard_at_5']['ci'][1]:+.2f}] | "
                     f"{d['spearman_correlation']['rho']:+.2f} [{d['spearman_correlation']['ci'][0]:+.2f}, {d['spearman_correlation']['ci'][1]:+.2f}] |")
    lines += ["", "## 2. Share of the invariant ρ gain recovered by fixing the target class (95% CI)", "",
              "| Explainer / stress test | Share | 95% CI |", "|---|---|---|"]
    for k, d in result['target_recovery'].items():
        lines.append(f"| {k} | {d['share']:.0%} | [{d['ci'][0]:.0%}, {d['ci'][1]:.0%}] |")
    lines += ["", f"## 3. Equivalence (SESOI |ΔJ@5| < {SESOI_J5}; 90% CI)", "",
              "| Comparison | Mean | 90% CI | Different from 0 | Equivalent |", "|---|---|---|---|---|"]
    for k, d in result['equivalence']['explainer_differences'].items():
        lines.append(f"| {k} | {d['mean']:+.3f} | [{d['ci90'][0]:+.3f}, {d['ci90'][1]:+.3f}] | "
                     f"{'yes' if d['different'] else 'no'} | {'yes' if d['equivalent'] else 'no'} |")
    lines += ["", f"Per-instance J@5 vs comprehensiveness (SESOI |ρ| < {SESOI_RHO}; 90% CI):", "",
              "| Explainer / stress test | ρ | 90% CI | Equivalent to 0 |", "|---|---|---|---|"]
    for k, d in result['equivalence']['stability_faithfulness'].items():
        lines.append(f"| {k} | {d['rho']:+.2f} | [{d['ci90'][0]:+.2f}, {d['ci90'][1]:+.2f}] | {'yes' if d['equivalent_to_zero'] else 'no'} |")
    lines += ["", "## 4. Chance-corrected J@5", "", "| Explainer / stress test | J@5 | E[J@5] by chance | Chance-corrected |",
              "|---|---|---|---|"]
    for k, d in result['chance_corrected'].items():
        lines.append(f"| {k} | {d['j5']:.3f} | {d['mean_expected']:.3f} | {d['j5_chance_corrected']:.3f} |")
    lines += ["", "## 5. LIME seed reuse", "", "| Stress test | Identical LIME explanations | J@5 all | J@5 excluding identical | Seed-variation J@5 |",
              "|---|---|---|---|---|"]
    for k, d in result['lime_seed_reuse'].items():
        lines.append(f"| {k} | {d['identical_share']:.1%} | {d['j5_all']:.3f} | {d['j5_excluding_identical']:.3f} | {d['seed_floor_j5']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('target', choices=sorted(RUNNER_NAMES) + ['matched'])
    args = parser.parse_args(argv)
    rng = np.random.default_rng(0)
    if args.target == 'matched':
        result = {'kind': 'matched', 'combinations': matched_agnews(rng)}
    else:
        records = load_per_instance(latest_instance_dir(args.target))
        result = {'experiment': args.target,
                  'strength': strength_correlations(args.target, records, rng),
                  'target_recovery': target_recovery(args.target, records, rng),
                  'equivalence': equivalence(args.target, records, rng),
                  'chance_corrected': chance_corrected(records),
                  'lime_seed_reuse': lime_seed_reuse(records)}
    out = REPO_ROOT / 'results' / 'analysis'
    name = 'matched_agnews_inference' if args.target == 'matched' else f'{args.target}_inference'
    (out / f'{name}.json').write_text(json.dumps(result, indent=2))
    (out / f'{name}.md').write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == '__main__':
    main()
