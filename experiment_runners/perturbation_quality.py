"""
Perturbation-quality comparison for the journal extension.

Re-runs the paraphrasing stress test with the generator released with the
paper (SemanticParaphrasingV1) on a seeded subset of each experiment's
correctly classified instances, with the paper's explainer settings. Compare
against the corrected generator's results for the same texts from the main
runs (match on original_text in the per-instance files).

The released generator depends on Python's hash seed; this script refuses to
run unless PYTHONHASHSEED is set, so the comparison is reproducible.

Usage (from the repository root):
    PYTHONHASHSEED=0 uv run python experiment_runners/perturbation_quality.py [--experiments experiment_1 ...] [--max-samples 300]
"""

import argparse
import logging
import os

from common import EXPLAINERS, EXPLAINER_PARAMS, RANDOM_SEED, REPO_ROOT
from stability_framework.model_loader import EXPERIMENT_MODELS
from stability_framework.stability_assessor import StabilityAssessor

DATASETS = {
    'experiment_1': {'config_key': 'sst2', 'sample_size': None, 'stratify': False},
    'experiment_2': {'config_key': 'sst2', 'sample_size': None, 'stratify': False},
    'experiment_3': {'config_key': 'ag_news', 'sample_size': 2000, 'stratify': True},
    'experiment_4': {'config_key': 'ag_news', 'sample_size': 2000, 'stratify': True},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--experiments', nargs='+', default=sorted(DATASETS))
    parser.add_argument('--max-samples', type=int, default=300)
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)
    if 'PYTHONHASHSEED' not in os.environ:
        parser.error("set PYTHONHASHSEED (e.g. PYTHONHASHSEED=0): the released generator depends on it")

    out = REPO_ROOT / 'results' / 'perturbation_quality'
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(out / 'perturbation_quality.log'), logging.StreamHandler()])

    for key in args.experiments:
        assessor = StabilityAssessor(results_dir=str(out), random_seed=RANDOM_SEED, device=args.device,
                                     max_samples=args.max_samples, correct_predictions_only=True)
        assessor.run_experiment(
            model_name=EXPERIMENT_MODELS[key]['name'],
            dataset_config=DATASETS[key],
            explainer_names=EXPLAINERS,
            stress_test_names=['semantic_paraphrasing_v1'],
            experiment_name=f'{key}_paraphrasing_v1',
            explainer_params=EXPLAINER_PARAMS,
            stress_test_params={'semantic_paraphrasing_v1': {'similarity_threshold': 0.8}},
        )


if __name__ == '__main__':
    main()
