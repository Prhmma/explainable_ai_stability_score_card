"""
Figures for the journal extension.

dose:    stability (J@5, primary estimand) against token-level dose, one panel per
         attribution method and stress test, one line per model. Colour marks the
         dataset and line style the tokenizer (solid uncased, dashed cased).
effects: attribute effects relative to the primary estimand (from
         results/analysis/attribute_effects.csv), one panel per experiment, one
         row per contrast, one marker per method, with 95% cluster-bootstrap
         intervals.

Colours: validated categorical slots (blue, orange, aqua; all pairs pass the
colour-vision checks); aqua is below 3:1 contrast on white, so methods also differ
in marker shape and are named in the legend.

Usage (from the repository root):
    uv run python -m analysis.figures dose --out DIR [--tests PreprocessingVariation SemanticParaphrasing]
    uv run python -m analysis.figures effects --out DIR [--tests ...]
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

from analysis.report import REPO_ROOT  # noqa: E402

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
METHODS = [('IntegratedGradients', 'IG', BLUE, 'o'), ('LIME', 'LIME', ORANGE, 's'), ('SHAP', 'SHAP', AQUA, '^')]
MODELS = {  # experiment: (label, dataset colour, line style)
    'experiment_1': ('DistilBERT, SST-2', BLUE, '-'),
    'experiment_2': ('FLAN-T5, SST-2', BLUE, '--'),
    'experiment_4': ('DistilBERT, AG News', ORANGE, '-'),
    'experiment_3': ('RoBERTa, AG News', ORANGE, '--'),
}
TEST_LABELS = {'PreprocessingVariation': 'Preprocessing', 'SemanticParaphrasing': 'Paraphrasing'}
CONTRAST_LABELS = {
    'zero_dose_included': 'Zero-dose variants included',
    'own_class_target': 'Own-class target',
    'string_keys': 'Surface-string keys',
    'k3': 'J@3 instead of J@5',
    'k10': 'J@10 instead of J@5',
    'signed': 'Signed attributions',
    'comparison_mean': 'Mean over comparisons',
    'misclassified_included': 'Misclassified instances included',
    'gradient_config': 'Alternative IG configuration',
    'fused_attention': 'Fused attention kernel',
}
EXPERIMENT_TITLES = {'experiment_1': 'E1 DistilBERT\nSST-2', 'experiment_2': 'E2 FLAN-T5\nSST-2',
                     'experiment_3': 'E3 RoBERTa\nAG News', 'experiment_4': 'E4 DistilBERT\nAG News'}


def _style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42,
    })


def dose_figure(out_dir: str, tests):
    from analysis.attribute_effects import Doser, _comparisons, _target_map
    from analysis.stability import load_latest_records
    edges = 2.0 ** np.arange(-7, 2.5, 1.0)  # 1/128 ... 4 (token edit fraction)
    centres = np.sqrt(edges[:-1] * edges[1:])
    data = {}  # (experiment, method, test) -> per-bin (mean, se)
    for experiment in MODELS:
        records = load_latest_records(experiment)
        dose = Doser(experiment)
        for method, *_ in METHODS:
            for test in tests:
                rows = records.get((method, test))
                if not rows:
                    continue
                target = _target_map(experiment, method, test)
                per_bin = defaultdict(lambda: defaultdict(list))  # bin -> instance -> values
                for r in rows:
                    for d, _, _, primary, _, _ in _comparisons(r, dose, target):
                        if d <= 0 or np.isnan(primary):
                            continue
                        b = np.searchsorted(edges, d) - 1
                        if 0 <= b < len(centres):
                            per_bin[b][r['sample_idx']].append(primary)
                series = []
                for b in range(len(centres)):
                    inst = [np.mean(v) for v in per_bin[b].values()]
                    n_comp = sum(len(v) for v in per_bin[b].values())
                    if n_comp >= 30 and len(inst) >= 10:
                        series.append((centres[b], float(np.mean(inst)), float(np.std(inst, ddof=1) / np.sqrt(len(inst)))))
                data[(experiment, method, test)] = series
    _style()
    fig, axes = plt.subplots(len(tests), 3, figsize=(6.3, 2.1 * len(tests) + 0.4), sharey=True, squeeze=False)
    for row, test in enumerate(tests):
        for col, (method, label, _, _) in enumerate(METHODS):
            ax = axes[row][col]
            for experiment, (mlabel, colour, ls) in MODELS.items():
                s = data.get((experiment, method, test))
                if not s:
                    continue
                x, y, se = map(np.array, zip(*s))
                ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=colour, alpha=0.12, linewidth=0)
                ax.plot(x, y, ls, color=colour, linewidth=1.5, marker='o', markersize=2.5, label=mlabel)
            ax.set_xscale('log', base=2)
            ax.set_ylim(0, 1.02)
            ax.grid(True, color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_title(f"{label}, {TEST_LABELS[test].lower()}", color=INK)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:g}" if v >= 1 else f"1/{round(1 / v):d}"))
            if col == 0:
                ax.set_ylabel('J@5')
    fig.supxlabel('Token-level dose (edit distance / original length, log scale)', fontsize=8, y=0.07)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'dose_response.pdf')
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print('wrote', path)


def effects_figure(out_dir: str, tests, contrasts=None, name='attribute_effects'):
    path_csv = REPO_ROOT / 'results' / 'analysis' / 'attribute_effects.csv'
    rows = [r for r in csv.DictReader(open(path_csv)) if r['test'] in tests]
    contrasts = [c for c in (contrasts or CONTRAST_LABELS) if any(r['contrast'] == c for r in rows)]
    rows = [r for r in rows if r['contrast'] in contrasts]
    experiments = list(EXPERIMENT_TITLES)
    _style()
    fig, axes = plt.subplots(1, len(experiments), figsize=(6.3, 0.26 * len(contrasts) * len(tests) + 1.2),
                             sharey=True, sharex=True)
    ylabels, ypos = [], {}
    y = 0
    for test in tests:
        for c in contrasts:
            ypos[(test, c)] = y
            ylabels.append(CONTRAST_LABELS[c] + (f" ({TEST_LABELS[test].lower()})" if len(tests) > 1 else ''))
            y += 1
    offsets = {'IntegratedGradients': -0.22, 'LIME': 0.0, 'SHAP': 0.22}
    for ax, experiment in zip(axes, experiments):
        ax.axvline(0, color=MUTED, linewidth=0.6)
        for method, label, colour, marker in METHODS:
            for r in rows:
                if r['experiment'] != experiment or r['explainer'] != method:
                    continue
                yy = ypos[(r['test'], r['contrast'])] + offsets[method]
                d, lo, hi = float(r['delta']), float(r['lo']), float(r['hi'])
                ax.plot([lo, hi], [yy, yy], color=colour, linewidth=1.2, solid_capstyle='round')
                ax.plot(d, yy, marker, color=colour, markersize=4, markeredgecolor='white', markeredgewidth=0.5,
                        label=label)
        ax.set_title(EXPERIMENT_TITLES[experiment], color=INK)
        ax.grid(True, axis='x', color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel(r'$\Delta$J@5')
        ax.set_ylim(len(ylabels) - 0.5, -0.5)
    axes[0].set_yticks(range(len(ylabels)))
    axes[0].set_yticklabels(ylabels)
    handles, labels, seen = [], [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.6, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{name}.pdf')
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print('wrote', path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('figure', choices=['dose', 'effects'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--tests', nargs='+', default=['PreprocessingVariation', 'SemanticParaphrasing'])
    parser.add_argument('--contrasts', nargs='+', choices=list(CONTRAST_LABELS), help='effects: rows to show (default: all)')
    parser.add_argument('--name', default='attribute_effects', help='effects: output file name')
    args = parser.parse_args(argv)
    if args.figure == 'dose':
        dose_figure(args.out, args.tests)
    else:
        effects_figure(args.out, args.tests, args.contrasts, args.name)


if __name__ == '__main__':
    main()
