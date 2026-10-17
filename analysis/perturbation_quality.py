"""
Perturbation quality (RQ2): the released paraphrase generator (v1) vs. the current one.

For the same texts (the v1 runs' 300-instance subsets, matched by original_text
to the main runs), per explainer:
- variants per instance, mean edit fraction and semantic distance;
- share of single-word substitutions that replace a stopword (v1 substituted
  function words, e.g. "a" -> "adenine");
- stability (J@5, rho, AD) under each generator.

Usage (from the repository root):
    uv run python -m analysis.perturbation_quality experiment_1
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np

from analysis.report import REPO_ROOT, RUNNER_NAMES, latest_instance_dir
from analysis.stability import load_latest_records, METRICS, load_per_instance, stability_summary
from analysis.strength import annotate_strength, make_embedder

LABELS = {'jaccard_at_5': 'J@5', 'spearman_correlation': 'ρ', 'attr_diff_relative_difference_percentage': 'AD (%)'}


def _v1_dir(experiment_key: str) -> str:
    dirs = glob.glob(str(REPO_ROOT / 'results' / 'perturbation_quality' / 'per_instance' / f'{experiment_key}_paraphrasing_v1_*'))
    if not dirs:
        raise FileNotFoundError(f"No v1 paraphrasing results for {experiment_key}")
    return max(dirs, key=os.path.getmtime)


def _stopword_substitution_share(records: List[Dict[str, Any]]) -> float:
    from stability_framework.stress_tests import SemanticParaphrasingTest
    stop = SemanticParaphrasingTest._stopwords()
    singles = stop_hits = 0
    for r in records:
        original = r['original_text'].split()
        for variant in r['variant_texts'][1:]:
            words = variant.split()
            if len(words) != len(original):
                continue
            changed = [a for a, b in zip(original, words) if a != b]
            if len(changed) == 1:
                singles += 1
                stop_hits += changed[0].strip('.,!?;:').lower() in stop
    return stop_hits / singles if singles else float('nan')


def _generator_profile(records: List[Dict[str, Any]], embed) -> Dict[str, Any]:
    rows = annotate_strength(records, embed)
    return {
        'variants_per_instance': float(np.mean([len(r['variant_texts']) - 1 for r in records])),
        'edit_fraction': float(np.mean([x['edit_fraction'] for x in rows])) if rows else float('nan'),
        'semantic_distance': float(np.mean([x['semantic_distance'] for x in rows])) if rows else float('nan'),
        'stopword_substitution_share': _stopword_substitution_share(records),
        'stability': stability_summary(records),
    }


def build(experiment_key: str) -> Dict[str, Any]:
    main = load_latest_records(experiment_key)
    v1 = load_per_instance(_v1_dir(experiment_key))
    embed = make_embedder('cpu')
    out = {'experiment': experiment_key, 'explainers': {}}
    for (explainer, test), rows in v1.items():
        texts = {r['original_text'] for r in rows}
        current = [r for r in main.get((explainer, 'SemanticParaphrasing'), []) if r['original_text'] in texts]
        out['explainers'][explainer] = {'n_instances': len(rows),
                                        'v1': _generator_profile(rows, embed),
                                        'current': _generator_profile(current, embed)}
    return out


def to_markdown(result: Dict[str, Any]) -> str:
    lines = [f"# Perturbation quality: {result['experiment']}", "",
             "Released (v1) vs. current paraphrase generator on the same instances.", "",
             "| Explainer | Generator | n | Variants/inst. | Edit fraction | Semantic dist. | Stopword subst. | "
             + " | ".join(LABELS[m] for m in METRICS) + " |", "|" + "---|" * (7 + len(METRICS))]
    for explainer, s in result['explainers'].items():
        for gen in ('v1', 'current'):
            g = s[gen]
            stab = " | ".join(f"{g['stability']['all'][m]['mean']:.3f}" for m in METRICS)
            lines.append(f"| {explainer} | {gen} | {s['n_instances']} | {g['variants_per_instance']:.2f} | {g['edit_fraction']:.3f} | "
                         f"{g['semantic_distance']:.3f} | {g['stopword_substitution_share']:.1%} | {stab} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiment', choices=sorted(RUNNER_NAMES))
    args = parser.parse_args(argv)
    result = build(args.experiment)
    out = REPO_ROOT / 'results' / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.experiment}_perturbation_quality.json').write_text(json.dumps(result, indent=2, default=str))
    (out / f'{args.experiment}_perturbation_quality.md').write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == '__main__':
    main()
