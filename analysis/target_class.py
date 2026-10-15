"""
Target class of variant explanations: the variant's own prediction vs. the original's.

The framework explains every input for its own predicted class. When a
perturbation changes the prediction, the original and the variant explanation
then describe different classes. This re-explains exactly those variants for
the original's predicted class (all other comparisons are unaffected) and
compares, per explainer and stress test:
- the prediction-changing comparisons under both targets (J@5, ρ, AD, share of ρ < 0);
- stability over all comparisons under each protocol, next to the
  prediction-invariant subset.

Re-explanations are stored per comparison in results/target_class/<experiment>/
and reused on reruns.

Usage (from the repository root):
    uv run python -m analysis.target_class experiment_1 [--device mps] [--summary-only]
"""

import argparse
import json
from typing import Any, Dict, List

import numpy as np

from analysis.report import REPO_ROOT, RUNNER_NAMES, latest_instance_dir
from analysis.stability import load_per_instance

FACTORY_NAMES = {'IntegratedGradients': 'integrated_gradients', 'LIME': 'lime', 'SHAP': 'shap'}
KEPT = ('jaccard_at_5', 'spearman_correlation', 'attr_diff_relative_difference_percentage')
LABELS = {'jaccard_at_5': 'J@5', 'spearman_correlation': 'ρ', 'attr_diff_relative_difference_percentage': 'AD (%)'}


def _changed(record):
    indices = record.get('compared_variant_indices') or range(1, len(record['variant_explanations']))
    return [(i, m) for i, m in zip(indices, record['metrics']) if not m['comparison_info']['same_prediction']]


def run(experiment_key: str, records, device: str) -> None:
    import torch
    from experiment_runners.common import EXPLAINER_PARAMS, RANDOM_SEED
    from stability_framework.explainers import create_explainer
    from stability_framework.metrics import StabilityMetrics
    from stability_framework.model_loader import EXPERIMENT_MODELS, ModelLoader

    out_dir = REPO_ROOT / 'results' / 'target_class' / experiment_key
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = ModelLoader(device=device).load_model_and_tokenizer(EXPERIMENT_MODELS[experiment_key]['name'])
    metrics = StabilityMetrics()
    for (explainer_name, test), rows in records.items():
        if test == 'SeedVariation':
            continue
        path = out_dir / f'{explainer_name}_{test}.jsonl'
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    done.add((r['sample_idx'], r['variant_idx']))
                except json.JSONDecodeError:
                    pass  # a line cut off by an interrupted run is redone
        factory = FACTORY_NAMES[explainer_name]
        explainer = create_explainer(factory, random_seed=RANDOM_SEED, **EXPLAINER_PARAMS[factory])
        with open(path, 'a', encoding='utf-8') as f:
            for r in rows:
                original = r['original_explanation']
                for i, m in _changed(r):
                    if (r['sample_idx'], i) in done:
                        continue
                    exp = explainer.explain(model, tokenizer, r['variant_texts'][i], target_class=original['predicted_class'])
                    fixed = metrics.calculate_all_metrics(original, exp)
                    f.write(json.dumps({
                        'sample_idx': r['sample_idx'], 'variant_idx': i,
                        'own_target': {k: m[k] for k in KEPT},
                        'original_target': {k: fixed[k] for k in KEPT},
                        'explained_class': exp.get('target_class'),
                    }) + '\n')
                    f.flush()
                if device == 'mps':
                    torch.mps.empty_cache()


def _mean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float('nan')


def summarise(experiment_key: str, records) -> Dict[str, Any]:
    out_dir = REPO_ROOT / 'results' / 'target_class' / experiment_key
    out = {'experiment': experiment_key, 'combinations': {}}
    for (explainer_name, test), rows in records.items():
        path = out_dir / f'{explainer_name}_{test}.jsonl'
        if test == 'SeedVariation' or not path.exists():
            continue
        fixed = {}
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            fixed[(r['sample_idx'], r['variant_idx'])] = r
        own_all, fixed_all, invariant, changed_own, changed_fixed = [], [], [], [], []
        missing = 0
        for r in rows:
            indices = r.get('compared_variant_indices') or range(1, len(r['variant_explanations']))
            for i, m in zip(indices, r['metrics']):
                own_all.append(m)
                if m['comparison_info']['same_prediction']:
                    fixed_all.append(m)
                    invariant.append(m)
                elif (r['sample_idx'], i) in fixed:
                    entry = fixed[(r['sample_idx'], i)]
                    fixed_all.append(entry['original_target'])
                    changed_own.append(entry['own_target'])
                    changed_fixed.append(entry['original_target'])
                else:
                    missing += 1
        block = {'n_comparisons': len(own_all), 'n_changed': len(changed_own), 'missing': missing}
        for k in KEPT:
            block[k] = {
                'changed_own_target': _mean(c[k] for c in changed_own),
                'changed_original_target': _mean(c[k] for c in changed_fixed),
                'all_own_target': _mean(c[k] for c in own_all),
                'all_original_target': _mean(c[k] for c in fixed_all),
                'invariant_only': _mean(c[k] for c in invariant),
            }
        block['negative_rho_share_changed_own'] = _mean(c['spearman_correlation'] < 0 for c in changed_own
                                                        if not np.isnan(c['spearman_correlation']))
        block['negative_rho_share_changed_original'] = _mean(c['spearman_correlation'] < 0 for c in changed_fixed
                                                             if not np.isnan(c['spearman_correlation']))
        out['combinations'][f'{explainer_name}/{test}'] = block
    return out


def to_markdown(result: Dict[str, Any]) -> str:
    lines = [f"# Target class of variant explanations: {result['experiment']}", "",
             "## 1. Prediction-changing comparisons: variant explained for its own vs. the original's class", "",
             "| Explainer / stress test | n changed | " + " | ".join(f"{LABELS[k]} own → original" for k in KEPT)
             + " | ρ < 0 own → original |", "|---|---|" + "---|" * (len(KEPT) + 1)]
    for name, b in result['combinations'].items():
        cells = " | ".join(f"{b[k]['changed_own_target']:.3f} → {b[k]['changed_original_target']:.3f}" for k in KEPT)
        lines.append(f"| {name} | {b['n_changed']} | {cells} | "
                     f"{b['negative_rho_share_changed_own']:.1%} → {b['negative_rho_share_changed_original']:.1%} |")
    lines += ["", "## 2. All comparisons under each protocol, and the prediction-invariant subset", "",
              "| Explainer / stress test | Metric | Own target (current) | Original target | Invariant only |",
              "|---|---|---|---|---|"]
    for name, b in result['combinations'].items():
        for k in KEPT:
            lines.append(f"| {name} | {LABELS[k]} | {b[k]['all_own_target']:.3f} | {b[k]['all_original_target']:.3f} | "
                         f"{b[k]['invariant_only']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiment', choices=sorted(RUNNER_NAMES))
    parser.add_argument('--device', default='mps')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args(argv)
    records = load_per_instance(latest_instance_dir(args.experiment))
    if not args.summary_only:
        run(args.experiment, records, args.device)
    result = summarise(args.experiment, records)
    out = REPO_ROOT / 'results' / 'analysis'
    (out / f'{args.experiment}_target_class.json').write_text(json.dumps(result, indent=2))
    (out / f'{args.experiment}_target_class.md').write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == '__main__':
    main()
