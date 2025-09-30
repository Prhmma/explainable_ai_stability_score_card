"""
Experiment 2: T5-small on SST-2 Validation Set

Evaluates stability of explanation methods for T5-small adapted for SST-2
sentiment classification using the full validation set (872 samples).
Note: This requires custom T5 fine-tuning for classification.
"""

import logging
from stability_framework.stability_assessor import StabilityAssessor
from stability_framework.model_loader import EXPERIMENT_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_2.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run Experiment 2: T5-small on SST-2."""
    logger.info("Starting Experiment 2: T5-small on SST-2")
    
    # Note: T5 requires special handling for classification
    logger.warning("T5 classification requires custom fine-tuning setup")
    logger.warning("This experiment may need additional model preparation")
    
    # Experiment configuration
    model_name = EXPERIMENT_MODELS['experiment_2']['name']
    
    # Build dataset config directly (no EXPERIMENT_DATASETS registry)
    dataset_config = {
        'config_key': 'sst2',
        'sample_size': None,  # Full validation set (872 samples)
        'stratify': False
    }
    explainer_names = ['integrated_gradients', 'lime', 'shap']
    stress_test_names = ['seed_variation', 'preprocessing_variation', 'semantic_paraphrasing']
    experiment_name = 'experiment_2_t5_sst2'
    
    # Initialize assessor
    assessor = StabilityAssessor(
        results_dir='../results',
        random_seed=42,
        max_samples=200  # Limited for T5 due to custom setup needed
    )
    
    # Explainer parameters (may need adjustment for T5)
    explainer_params = {
        'integrated_gradients': {'n_steps': 25},  # Reduced for T5
        'lime': {'num_samples': 500},  # Reduced for efficiency
        'shap': {}
    }
    
    # Stress test parameters
    stress_test_params = {
        'seed_variation': {'seeds': [42, 43, 44]},  # Reduced for T5
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
        card_path = '../results/%s_stability_card.md' % experiment_name
        assessor.generate_stability_card(results, card_path)
        
        logger.info("Experiment 2 completed successfully!")
        logger.info("Results saved to: %s", assessor.results_dir)
        logger.info("Stability card: %s", card_path)
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT 2 SUMMARY")
        print("="*50)
        print("Model: %s" % model_name)
        print("Dataset: SST-2 validation (limited)")
        print("Explainers: %d" % len(explainer_names))
        print("Stress tests: %d" % len(stress_test_names))
        print("Duration: %.1f seconds" % results['duration_seconds'])
        print("="*50)
        
    except Exception as e:
        logger.error("Experiment 2 failed: %s", str(e))
        logger.error("Note: T5 may require custom model preparation for classification")
        raise


if __name__ == "__main__":
    main()