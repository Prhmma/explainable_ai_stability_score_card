"""
Implementation sensitivity (RQ1): the effect of reverting single choices.

Each configuration's per-instance results are compared with the main runs on
the same texts (matched on original_text). Two further choices are analysed
post hoc from the main runs:
- undefined_as_zero: undefined Spearman correlation and attribution difference
  scored 0.0 (as released) instead of being excluded;
- shared_random_stream: preprocessing variants drawn from one global random
  stream shared by all explainers (as released) instead of per text; reported
  as the share of instances whose variants differ between explainers.

Usage (from the repository root):
    uv run python -m analysis.sensitivity experiment_1
"""

import argparse
import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from analysis.report import REPO_ROOT, RUNNER_NAMES, latest_instance_dir
from analysis.stability import METRICS, load_latest_records, load_per_instance, stability_summary

CONFIG_SUFFIXES = ('LegacyConfig', 'LegacyAlignment')
LABELS = {'jaccard_at_5': 'J@5', 'spearman_correlation': 'ρ', 'attr_diff_relative_difference_percentage': 'AD (%)'}


def _base_name(explainer: str) -> str:
    for suffix in CONFIG_SUFFIXES:
        if explainer.endswith(suffix):
            return explainer[:-len(suffix)]
    return explainer


def _config_dir(config: str, experiment_key: str) -> str:
    dirs = glob.glob(str(REPO_ROOT / 'results' / 'implementation_sensitivity' / config / 'per_instance' / f'{experiment_key}_{config}_*'))
    if not dirs:
        raise FileNotFoundError(f"No results for {config} / {experiment_key}")
    return max(dirs, key=os.path.getmtime)


def compare_config(main: Dict, variant: Dict) -> Dict[str, Any]:
    """Current vs. reverted summaries on the same texts, per explainer and stress test."""
    out = {}
    for (explainer, test), rows in variant.items():
        base = (_base_name(explainer), test)
        if base not in main:
            continue
        texts = {r['original_text'] for r in rows}
        current = [r for r in main[base] if r['original_text'] in texts]
        out[f'{base[0]}/{test}'] = {'current': stability_summary(current), 'reverted': stability_summary(rows),
                                    'n_instances': len(rows)}
    return out


def undefined_as_zero(main: Dict) -> Dict[str, Any]:
    """Main-run summaries with undefined ρ and AD scored 0.0 instead of excluded."""
    out = {}
    for (explainer, test), rows in main.items():
        zeroed = [{**r, 'metrics': [{**m, **{k: (0.0 if isinstance(m[k], float) and math.isnan(m[k]) else m[k])
                                             for k in ('spearman_correlation', 'attr_diff_relative_difference_percentage')}}
                                    for m in r['metrics']]} for r in rows]
        out[f'{explainer}/{test}'] = {'current': stability_summary(rows), 'reverted': stability_summary(zeroed),
                                      'n_instances': len(rows)}
    return out


def shared_random_stream(texts: List[str], seed: int = 42, n_explainers: int = 3) -> Dict[str, float]:
    """Share of instances whose preprocessing variants differ between explainers under one global stream."""
    from stability_framework.stress_tests import PreprocessingVariationTest

    class GlobalStream(PreprocessingVariationTest):
        def _rng(self, text):
            return random  # the module-level stream, as released

        def apply(self, text):
            # Released selection: categories in order, first 8 unique. It keeps the
            # random case variant; the current round-robin selection never reaches
            # the random variants, so under it the stream would have no effect.
            variants = [text] + self._generate_case_variants(text) + \
                self._generate_punctuation_variants(text) + self._generate_spacing_variants(text)
            return list(dict.fromkeys(variants))[:8]

    test = GlobalStream(random_seed=seed)
    random.seed(seed)
    passes = [[tuple(test.apply(t)) for t in texts] for _ in range(n_explainers)]
    differs = sum(len({p[i] for p in passes}) > 1 for i in range(len(texts)))
    return {'instances': len(texts), 'instances_with_differing_variants': differs, 'share': differs / len(texts)}


def ranking(summaries: Dict[str, Any], which: str, metric: str = 'jaccard_at_5') -> Dict[str, List[str]]:
    """Explainers ordered by mean metric (most stable first) per stress test."""
    by_test: Dict[str, List] = {}
    for key, s in summaries.items():
        explainer, test = key.split('/')
        by_test.setdefault(test, []).append((s[which]['all'][metric]['mean'], explainer))
    return {test: [e for _, e in sorted(v, reverse=True)] for test, v in by_test.items()}


def build(experiment_key: str) -> Dict[str, Any]:
    main = load_latest_records(experiment_key)
    result = {'experiment': experiment_key, 'main_dir': 'latest records (merged across runs)', 'configs': {}}
    for config in ('legacy_ig_config', 'legacy_alignment', 'sdpa_attention'):
        try:
            result['configs'][config] = compare_config(main, load_per_instance(_config_dir(config, experiment_key)))
        except FileNotFoundError as e:
            result['configs'][config] = {'missing': str(e)}
    try:
        mis = load_per_instance(_config_dir('misclassified', experiment_key))
        result['configs']['misclassified'] = {
            f'{e}/{t}': {'current': stability_summary(main[(e, t)]),
                         'reverted': stability_summary(main[(e, t)] + rows),
                         'misclassified_only': stability_summary(rows), 'n_instances': len(rows)}
            for (e, t), rows in mis.items() if (e, t) in main}
    except FileNotFoundError as e:
        result['configs']['misclassified'] = {'missing': str(e)}
    result['configs']['undefined_as_zero'] = undefined_as_zero(main)
    texts = [r['original_text'] for r in next(iter(main.values()))]
    result['shared_random_stream'] = shared_random_stream(texts)
    return result


def to_markdown(result: Dict[str, Any]) -> str:
    lines = [f"# Implementation sensitivity: {result['experiment']}", "",
             "Mean over all comparisons; Δ = reverted − current, on the same instances.", ""]
    for config, combos in result['configs'].items():
        lines += [f"## {config}", ""]
        if 'missing' in combos:
            lines += [f"_{combos['missing']}_", ""]
            continue
        lines += ["| Explainer / stress test | n | " + " | ".join(f"{LABELS[m]} current → reverted (Δ)" for m in METRICS) + " |",
                  "|" + "---|" * (2 + len(METRICS))]
        for key, s in combos.items():
            cells = []
            for m in METRICS:
                a, b = s['current']['all'][m]['mean'], s['reverted']['all'][m]['mean']
                cells.append(f"{a:.3f} → {b:.3f} ({b - a:+.3f})")
            lines.append(f"| {key} | {s['n_instances']} | " + " | ".join(cells) + " |")
        cur, rev = ranking(combos, 'current'), ranking(combos, 'reverted')
        changed = [f"{t}: {' > '.join(cur[t])} → {' > '.join(rev[t])}" for t in cur if len(cur[t]) > 1 and cur[t] != rev.get(t)]
        lines += ["", "Ranking by J@5 changes: " + ("; ".join(changed) if changed else "none"), ""]
    s = result['shared_random_stream']
    lines += ["## shared_random_stream", "",
              f"With one global random stream shared by three explainers, {s['instances_with_differing_variants']} of "
              f"{s['instances']} instances ({s['share']:.1%}) receive different preprocessing variants per explainer.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiment', choices=sorted(RUNNER_NAMES))
    args = parser.parse_args(argv)
    result = build(args.experiment)
    out = REPO_ROOT / 'results' / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.experiment}_sensitivity.json').write_text(json.dumps(result, indent=2, default=str))
    (out / f'{args.experiment}_sensitivity.md').write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == '__main__':
    main()
