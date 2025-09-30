"""
Experiment 1: DistilBERT on SST-2 Validation Set

Evaluates stability of explanation methods for DistilBERT fine-tuned on SST-2
sentiment classification using the full validation set (872 samples).
"""

import logging
from stability_framework.stability_assessor import StabilityAssessor
from stability_framework.model_loader import EXPERIMENT_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_1.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run Experiment 1: DistilBERT on SST-2."""
    logger.info("Starting Experiment 1: DistilBERT on SST-2")
    
    # Experiment configuration
    model_name = EXPERIMENT_MODELS['experiment_1']['name']
    
    # Build dataset config directly (no EXPERIMENT_DATASETS registry)
    dataset_config = {
        'config_key': 'sst2',
        'sample_size': None,  # Full validation set (872 samples)
        'stratify': False
    }
    explainer_names = ['integrated_gradients', 'lime', 'shap']
    stress_test_names = ['seed_variation', 'preprocessing_variation', 'semantic_paraphrasing']
    experiment_name = 'experiment_1_distilbert_sst2'
    
    # Initialize assessor
    assessor = StabilityAssessor(
        results_dir='../results',
        random_seed=42,
        max_samples=None  # Full experiment - all 872 samples
    )
    
    # Explainer parameters (production settings)
    explainer_params = {
        'integrated_gradients': {'n_steps': 50},
        'lime': {'num_samples': 1000},
        'shap': {}
    }
    
    # Stress test parameters (full production settings)
    stress_test_params = {
        'seed_variation': {'seeds': [42, 43, 44, 45, 46]},
        'preprocessing_variation': {},
        'semantic_paraphrasing': {'similarity_threshold': 0.8}
    }
    
    try:
        # Run the experiment
        results = assessor.run_experiment(
            model_name=model_name,
            dataset_config=dataset_config,
            explainer_names=explainer_names,
            stress_test_names=stress_test_names,
            experiment_name=experiment_name,
            explainer_params=explainer_params,
            stress_test_params=stress_test_params
        )
        
        # Generate stability card
        card_path = f'../results/{experiment_name}_stability_card.md'
        assessor.generate_stability_card(results, card_path)
        
        logger.info("Experiment 1 completed successfully!")
        logger.info("Results saved to: %s", assessor.results_dir)
        logger.info("Stability card: %s", card_path)
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT 1 SUMMARY")
        print("="*50)
        print("Model: %s" % model_name)
        print("Dataset: SST-2 validation")
        print("Explainers: %d" % len(explainer_names))
        print("Stress tests: %d" % len(stress_test_names))
        print("Duration: %.1f seconds" % results['duration_seconds'])
        print("="*50)
        
    except Exception as e:
        logger.error("Experiment 1 failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()