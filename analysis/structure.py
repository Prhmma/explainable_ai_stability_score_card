"""
Variance structure, reliability and dose-response of stability under the primary estimand.

variance:    linear mixed model of J@5 per experiment and stress test, with the
             attribution method as a fixed effect and random components for
             instance, instance x method and variant (within instance, shared by
             all methods); the residual is variant x method. Shares are reported
             relative to the total of the random components plus the variance of
             the method means.
reliability: one-way random-effects intraclass correlation ICC(1) of J@5 with
             instances as groups and variants as replicates, per method and
             stress test, and the Spearman-Brown projection: the reliability of
             an instance's mean stability over n variants, and the number of
             variants needed for a reliability of 0.8.
dose:        mixed model J@5 ~ log2(dose) with random intercepts and slopes per
             instance (dose > 0), per experiment, method and stress test; the slope
             is the change in J@5 per doubling of the token-level dose.

Usage (from the repository root):
    uv run python -m analysis.structure experiment_1 [experiment_2 ...]
Writes results/analysis/structure.{json,md}.
"""

import argparse
import json
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.attribute_effects import Doser
from analysis.primary import METHODS, SHORT, primary_comparisons
from analysis.report import REPO_ROOT
from analysis.stability import load_latest_records

PERTURBATION_TESTS = ('PreprocessingVariation', 'SemanticParaphrasing')


def icc1(groups: List[np.ndarray]) -> Dict[str, float]:
    """ICC(1) from a one-way random-effects ANOVA with unequal group sizes."""
    groups = [g for g in groups if g.size >= 1]
    k, n_i = len(groups), np.array([g.size for g in groups])
    N = n_i.sum()
    grand = np.concatenate(groups).mean()
    msb = sum(g.size * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else float('nan')
    mean_n = float(n_i.mean())
    rel = mean_n * icc / (1 + (mean_n - 1) * icc) if icc > 0 else float('nan')
    n80 = 0.8 * (1 - icc) / (0.2 * icc) if icc > 0 else float('inf')
    return {'icc1': float(icc), 'mean_variants': mean_n, 'reliability_at_mean_n': float(rel), 'variants_for_0.8': float(n80)}


def _frame(experiment, records, dose) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for test in PERTURBATION_TESTS:
            recs = records.get((method, test))
            if not recs:
                continue
            for s, comps in primary_comparisons(experiment, method, test, recs, dose).items():
                rows += [{'method': method, 'test': test, 'instance': s, 'variant': c['variant'],
                          'dose': c['dose'], 'j5': c['j5']} for c in comps]
    return pd.DataFrame(rows)


def variance_components(df: pd.DataFrame) -> Dict[str, float]:
    import statsmodels.formula.api as smf
    full = df.groupby(['instance', 'variant']).filter(lambda g: g['method'].nunique() == len(METHODS))
    full = full.assign(vid=full.groupby('instance')['variant'].transform(lambda x: pd.factorize(x)[0]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = smf.mixedlm('j5 ~ C(method)', full, groups='instance', re_formula='1',
                            vc_formula={'instance_method': '0 + C(method)', 'variant': '0 + C(vid)'})
        fit = model.fit(reml=True, method='lbfgs')
    vc = dict(zip(model.exog_vc.names, fit.vcomp))
    method_means = full.groupby('method')['j5'].mean()
    comps = {'instance': float(fit.cov_re.iloc[0, 0]), 'instance_method': float(vc.get('instance_method', np.nan)),
             'variant': float(vc.get('variant', np.nan)), 'residual': float(fit.scale),
             'method_means': float(method_means.var(ddof=0))}
    total = sum(v for v in comps.values() if not np.isnan(v))
    return {'n_rows': int(len(full)), 'n_instances': int(full['instance'].nunique()),
            'shares': {k: v / total for k, v in comps.items()}, 'components': comps}


def dose_slope(df: pd.DataFrame) -> Dict[str, float]:
    import statsmodels.formula.api as smf
    d = df[df['dose'] > 0].assign(ld=lambda x: np.log2(x['dose']) - np.log2(0.05))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            fit = smf.mixedlm('j5 ~ ld', d, groups='instance', re_formula='~ld').fit(reml=True, method='lbfgs')
        except Exception:
            fit = smf.mixedlm('j5 ~ ld', d, groups='instance').fit(reml=True, method='lbfgs')
    lo, hi = fit.conf_int().loc['ld']
    return {'slope_per_doubling': float(fit.params['ld']), 'lo': float(lo), 'hi': float(hi), 'n': int(len(d))}


def build(experiments) -> Dict:
    result = {'variance': {}, 'reliability': {}, 'dose': {}}
    for e in experiments:
        records = load_latest_records(e)
        dose = Doser(e)
        df = _frame(e, records, dose)
        for test in PERTURBATION_TESTS:
            sub = df[df['test'] == test]
            if sub.empty:
                continue
            result['variance'][f'{e}|{test}'] = variance_components(sub)
            for method in METHODS:
                m = sub[sub['method'] == method]
                if m.empty:
                    continue
                groups = [g['j5'].to_numpy() for _, g in m.groupby('instance')]
                result['reliability'][f'{e}|{method}|{test}'] = icc1(groups)
                result['dose'][f'{e}|{method}|{test}'] = dose_slope(m)
    return result


def to_markdown(result) -> str:
    lines = ["# Variance structure, reliability and dose-response (primary estimand, J@5)", "",
             "## Variance components (share of total)", "",
             "| Experiment | Test | Instance | Instance x method | Variant | Variant x method (residual) | Method means | Rows |",
             "|---|---|---|---|---|---|---|---|"]
    for key, v in result['variance'].items():
        e, t = key.split('|')
        s = v['shares']
        lines.append(f"| {e} | {t} | {s['instance']:.1%} | {s['instance_method']:.1%} | {s['variant']:.1%} | "
                     f"{s['residual']:.1%} | {s['method_means']:.1%} | {v['n_rows']} |")
    lines += ["", "## Reliability of instance-level stability", "",
              "| Experiment | Method | Test | ICC(1) | Mean variants | Reliability at mean | Variants for 0.8 |", "|---|---|---|---|---|---|---|"]
    for key, r in result['reliability'].items():
        e, m, t = key.split('|')
        lines.append(f"| {e} | {SHORT[m]} | {t} | {r['icc1']:.2f} | {r['mean_variants']:.1f} | "
                     f"{r['reliability_at_mean_n']:.2f} | {r['variants_for_0.8']:.0f} |")
    lines += ["", "## Dose-response: change in J@5 per doubling of token-level dose", "",
              "| Experiment | Method | Test | Slope | 95% CI | Comparisons |", "|---|---|---|---|---|---|"]
    for key, d in result['dose'].items():
        e, m, t = key.split('|')
        lines.append(f"| {e} | {SHORT[m]} | {t} | {d['slope_per_doubling']:+.3f} | [{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['n']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiments', nargs='+')
    args = parser.parse_args(argv)
    result = build(args.experiments)
    out = REPO_ROOT / 'results' / 'analysis'
    (out / 'structure.json').write_text(json.dumps(result, indent=2))
    (out / 'structure.md').write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == '__main__':
    main()
