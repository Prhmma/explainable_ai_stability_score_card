"""
Implementation-sensitivity runs for the journal extension (RQ1).

Each configuration reverts one implementation choice to its behaviour in the
code released with the paper, with everything else at the current protocol,
on seeded subsets of Experiments 1 and 3. Results are compared with the main
runs' per-instance results for the same texts (analysis/sensitivity.py).

Configurations:
  legacy_ig_config        IG in train mode, classifier-only forward (DistilBERT), token-id-0 baseline
  legacy_alignment        released attribution keys for IG, LIME and SHAP
  sdpa_attention          fused SDPA attention on Apple Silicon (MPS only)
  misclassified           the instances the correct-prediction filter excludes

Undefined-metric handling and the shared random stream for preprocessing are
analysed post hoc and need no runs.

Usage (from the repository root):
    uv run python experiment_runners/implementation_sensitivity.py [--configs ...] [--experiments ...]
"""

import argparse
import logging

from common import EXPLAINER_PARAMS, RANDOM_SEED, REPO_ROOT
from stability_framework.explainers import create_explainer
from stability_framework.legacy.explainers import (
    LegacyAlignmentIG, LegacyAlignmentLIME, LegacyAlignmentSHAP, LegacyConfigIG)
from stability_framework.model_loader import EXPERIMENT_MODELS, ModelLoader
from stability_framework.stability_assessor import StabilityAssessor

EXPERIMENTS = {
    'experiment_1': {'dataset': {'config_key': 'sst2', 'sample_size': None, 'stratify': False}, 'subset': 300},
    'experiment_3': {'dataset': {'config_key': 'ag_news', 'sample_size': 2000, 'stratify': True}, 'subset': 150},
}
ALL_TESTS = ['seed_variation', 'preprocessing_variation', 'semantic_paraphrasing']
STRESS_TEST_PARAMS = {'seed_variation': {'seeds': [42, 43, 44, 45, 46]}, 'semantic_paraphrasing': {'similarity_threshold': 0.8}}


def standard_explainers():
    return [create_explainer(n, random_seed=RANDOM_SEED, **EXPLAINER_PARAMS[n]) for n in ('integrated_gradients', 'lime', 'shap')]


CONFIGS = {
    'legacy_ig_config': lambda: ([LegacyConfigIG(RANDOM_SEED, **EXPLAINER_PARAMS['integrated_gradients'])],
                                 ['preprocessing_variation', 'semantic_paraphrasing'], {}),
    'legacy_alignment': lambda: ([LegacyAlignmentIG(RANDOM_SEED, **EXPLAINER_PARAMS['integrated_gradients']),
                                  LegacyAlignmentLIME(RANDOM_SEED, **EXPLAINER_PARAMS['lime']),
                                  LegacyAlignmentSHAP(RANDOM_SEED)], ALL_TESTS, {}),
    'sdpa_attention': lambda: (standard_explainers(), ALL_TESTS, {'attn_implementation': 'sdpa'}),
    'misclassified': lambda: (standard_explainers(), ALL_TESTS, {'prediction_filter': 'incorrect'}),
}


class SensitivityAssessor(StabilityAssessor):
    """Assessor that takes explainer instances and can select misclassified instances."""

    def __init__(self, explainers, prediction_filter='correct', attn_implementation=None, **kwargs):
        super().__init__(**kwargs)
        self._explainers = explainers
        self._prediction_filter = prediction_filter
        if attn_implementation is not None:
            self.model_loader = ModelLoader(device=kwargs.get('device'), attn_implementation=attn_implementation)

    def _initialize_explainers(self, explainer_names, params):
        return self._explainers

    def _filter_correct_predictions(self, model, tokenizer, texts, labels, batch_size=32):
        kept, kept_labels, info = super()._filter_correct_predictions(model, tokenizer, texts, labels, batch_size)
        if self._prediction_filter != 'incorrect':
            return kept, kept_labels, info
        kept_set = set(zip(kept, kept_labels))
        wrong = [(t, y) for t, y in zip(texts, labels) if (t, y) not in kept_set]
        info = {**info, 'selection': 'misclassified', 'kept': len(wrong)}
        return [t for t, _ in wrong], [y for _, y in wrong], info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS))
    parser.add_argument('--experiments', nargs='+', default=list(EXPERIMENTS))
    parser.add_argument('--subset', type=int, default=None, help='Override subset size (e.g. for a smoke test)')
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)

    root = REPO_ROOT / 'results' / 'implementation_sensitivity'
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(root / 'implementation_sensitivity.log'), logging.StreamHandler()])

    for config in args.configs:
        for key in args.experiments:
            explainers, tests, options = CONFIGS[config]()
            subset = None if config == 'misclassified' else (args.subset or EXPERIMENTS[key]['subset'])
            assessor = SensitivityAssessor(
                explainers, results_dir=str(root / config), random_seed=RANDOM_SEED, device=args.device,
                max_samples=subset if args.subset is None or config != 'misclassified' else args.subset, **options)
            assessor.run_experiment(
                model_name=EXPERIMENT_MODELS[key]['name'], dataset_config=EXPERIMENTS[key]['dataset'],
                explainer_names=[e.name for e in explainers], stress_test_names=tests,
                experiment_name=f'{key}_{config}',
                explainer_params={e.name: e.get_params() for e in explainers},
                stress_test_params=STRESS_TEST_PARAMS)


if __name__ == '__main__':
    main()
