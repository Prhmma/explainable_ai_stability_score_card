"""
Shared configuration and entry point for the paper's four experiments.

The settings below are those reported in the paper (Section IV.A): IG with 50
steps from a zero baseline, LIME with 1000 samples, SHAP Partition explainer,
explainer seeds 42-46 (LIME only), paraphrases filtered at cosine >= 0.8, and
stability evaluated only on correctly classified instances.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from stability_framework.model_loader import EXPERIMENT_MODELS
from stability_framework.stability_assessor import StabilityAssessor

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

EXPLAINERS = ['integrated_gradients', 'lime', 'shap']
STRESS_TESTS = ['seed_variation', 'preprocessing_variation', 'semantic_paraphrasing']

EXPLAINER_PARAMS = {
    'integrated_gradients': {'n_steps': 50},
    'lime': {'num_samples': 1000},
    'shap': {}
}

STRESS_TEST_PARAMS = {
    'seed_variation': {'seeds': [42, 43, 44, 45, 46]},
    'preprocessing_variation': {},
    'semantic_paraphrasing': {'similarity_threshold': 0.8}
}

RANDOM_SEED = 42


def run_experiment(
    experiment_key: str,
    experiment_name: str,
    title: str,
    dataset_config: Dict[str, Any],
    argv: Optional[list] = None
) -> Dict[str, Any]:
    """Run one of the paper's experiments and write its results and stability card."""
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Evaluate at most this many (correctly classified) instances, e.g. for a smoke test')
    parser.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: CUDA if available)")
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR,
                        help=f'Output directory (default: {DEFAULT_RESULTS_DIR})')
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.results_dir / f'{experiment_name}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(experiment_name)

    model_name = EXPERIMENT_MODELS[experiment_key]['name']
    logger.info("Starting %s (model: %s)", title, model_name)

    assessor = StabilityAssessor(
        results_dir=str(args.results_dir),
        random_seed=RANDOM_SEED,
        device=args.device,
        max_samples=args.max_samples,
        correct_predictions_only=True
    )
    results = assessor.run_experiment(
        model_name=model_name,
        dataset_config=dataset_config,
        explainer_names=EXPLAINERS,
        stress_test_names=STRESS_TESTS,
        experiment_name=experiment_name,
        explainer_params=EXPLAINER_PARAMS,
        stress_test_params=STRESS_TEST_PARAMS
    )

    card_path = args.results_dir / f'{experiment_name}_stability_card.md'
    assessor.generate_stability_card(results, str(card_path))

    dataset_info = results['dataset_info']
    print("\n" + "=" * 50)
    print(title.upper())
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_info.get('dataset_name')} ({dataset_info.get('total_samples')} loaded, "
          f"{dataset_info.get('evaluated_samples')} evaluated)")
    print(f"Duration: {results['duration_seconds']:.1f} seconds")
    print(f"Stability card: {card_path}")
    print("=" * 50)
    return results
