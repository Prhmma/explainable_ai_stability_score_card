"""
Analysis report for one experiment's per-instance results.

Usage (from the repository root):
    uv run python -m analysis.report experiment_1 [--instance-dir DIR] [--device mps]

Writes results/analysis/<experiment>.json and .md with:
  1. reproduced results next to the paper's published tables,
  2. stability over all vs. prediction-invariant comparisons,
  3. agreement between explainers on the same inputs,
  4. deletion-based faithfulness vs. a random baseline.
"""

import argparse
import glob
import json
import os
from pathlib import Path

from analysis.faithfulness import faithfulness, make_predict_proba
from analysis.paper_tables import PAPER_RESULTS
from analysis.stability import METRICS, cross_method_agreement, load_per_instance, stability_summary

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_NAMES = {
    'experiment_1': 'experiment_1_distilbert_sst2',
    'experiment_2': 'experiment_2_t5_sst2',
    'experiment_3': 'experiment_3_roberta_agnews',
    'experiment_4': 'experiment_4_distilbert_agnews',
}
LABELS = {'jaccard_at_5': 'J@5', 'spearman_correlation': 'ρ', 'attr_diff_relative_difference_percentage': 'AD (%)'}
TEST_ORDER = ['SeedVariation', 'PreprocessingVariation', 'SemanticParaphrasing']


def latest_instance_dir(experiment_key: str) -> str:
    dirs = glob.glob(str(REPO_ROOT / 'results' / 'per_instance' / f'{RUNNER_NAMES[experiment_key]}_*'))
    if not dirs:
        raise FileNotFoundError(f"No per-instance results for {experiment_key}")
    return max(dirs, key=os.path.getmtime)


def _fmt(s):
    return "N/A" if s is None or s.get('n', 1) == 0 else f"{s['mean']:.3f} ± {s['std']:.3f}"


def build_report(experiment_key: str, instance_dir: str, device: str = None, with_faithfulness: bool = True):
    records = load_per_instance(instance_dir)
    keys = sorted(records, key=lambda k: (k[0], TEST_ORDER.index(k[1]) if k[1] in TEST_ORDER else 9))
    report = {'experiment': experiment_key, 'instance_dir': instance_dir, 'combinations': {}}
    for explainer, test in keys:
        summary = stability_summary(records[(explainer, test)])
        paper = PAPER_RESULTS.get((experiment_key, explainer, test))
        summary['paper'] = {m: {'mean': v[0], 'std': v[1]} for m, v in paper.items()} if paper else None
        report['combinations'][f'{explainer}/{test}'] = summary
    report['cross_method_agreement'] = cross_method_agreement(records)

    if with_faithfulness:
        from stability_framework.model_loader import EXPERIMENT_MODELS, ModelLoader
        model, tokenizer = ModelLoader(device=device).load_model_and_tokenizer(EXPERIMENT_MODELS[experiment_key]['name'])
        predict = make_predict_proba(model, tokenizer)
        report['faithfulness'] = {
            explainer: faithfulness(rows, predict)
            for (explainer, test), rows in records.items() if test == 'PreprocessingVariation'
        }
    return report


def to_markdown(report) -> str:
    lines = [f"# Analysis: {report['experiment']}", "", f"Per-instance data: `{report['instance_dir']}`", ""]
    lines += ["## 1. Reproduced vs. published (all comparisons)", "",
              "| Explainer | Stress test | " + " | ".join(f"{LABELS[m]} (now)" for m in METRICS)
              + " | " + " | ".join(f"{LABELS[m]} (paper)" for m in METRICS) + " | n inst. |",
              "|" + "---|" * (3 + 2 * len(METRICS))]
    for key, s in report['combinations'].items():
        explainer, test = key.split('/')
        now = " | ".join(_fmt(s['all'][m]) for m in METRICS)
        paper = " | ".join(_fmt(s['paper'][m]) if s['paper'] else "N/A" for m in METRICS)
        lines.append(f"| {explainer} | {test} | {now} | {paper} | {s['n_instances']} |")
    lines += ["", "## 2. All vs. prediction-invariant comparisons", "",
              "| Explainer | Stress test | Retention | " + " | ".join(f"{LABELS[m]} (all)" for m in METRICS)
              + " | " + " | ".join(f"{LABELS[m]} (invariant)" for m in METRICS) + " |",
              "|" + "---|" * (3 + 2 * len(METRICS))]
    for key, s in report['combinations'].items():
        explainer, test = key.split('/')
        lines.append(f"| {explainer} | {test} | {s['prediction_retention']:.1%} | "
                     + " | ".join(_fmt(s['all'][m]) for m in METRICS) + " | "
                     + " | ".join(_fmt(s['prediction_invariant'][m]) for m in METRICS) + " |")
    lines += ["", "## 3. Agreement between explainers (unperturbed inputs)", "", "| Pair | J@5 | ρ |", "|---|---|---|"]
    for pair, s in report['cross_method_agreement'].items():
        lines.append(f"| {pair} | {_fmt(s['jaccard_at_5'])} | {_fmt(s['spearman_correlation'])} |")
    if 'faithfulness' in report:
        lines += ["", "## 4. Deletion-based faithfulness (AOPC over 10/20/50% of words)", "",
                  "Higher comprehensiveness and lower sufficiency are better; compare each with the random ranking.", "",
                  "| Explainer | Comprehensiveness | (random) | Sufficiency | (random) | n |", "|---|---|---|---|---|---|"]
        for explainer, f in report['faithfulness'].items():
            lines.append(f"| {explainer} | {f['explainer']['comprehensiveness']:.3f} | {f['random']['comprehensiveness']:.3f} | "
                         f"{f['explainer']['sufficiency']:.3f} | {f['random']['sufficiency']:.3f} | {f['explainer']['n']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiment', choices=sorted(RUNNER_NAMES))
    parser.add_argument('--instance-dir', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--no-faithfulness', action='store_true')
    args = parser.parse_args(argv)
    instance_dir = args.instance_dir or latest_instance_dir(args.experiment)
    report = build_report(args.experiment, instance_dir, args.device, not args.no_faithfulness)
    out = REPO_ROOT / 'results' / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.experiment}.json').write_text(json.dumps(report, indent=2, default=str))
    (out / f'{args.experiment}.md').write_text(to_markdown(report))
    print(to_markdown(report))


if __name__ == '__main__':
    main()
