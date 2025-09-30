"""
Experiment 4: DistilBERT on AG-News Test Set

Evaluates stability of explanation methods for DistilBERT fine-tuned on AG-News
topic classification using stratified sampling (2000 samples, 500 per class).
Note: This requires custom DistilBERT fine-tuning for AG-News.
"""

import logging
from stability_framework.stability_assessor import StabilityAssessor
from stability_framework.model_loader import EXPERIMENT_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_4.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run Experiment 4: DistilBERT on AG-News."""
    logger.info("Starting Experiment 4: DistilBERT on AG-News")
    
    # Note: This requires DistilBERT fine-tuned specifically for AG-News
    logger.warning("DistilBERT for AG-News requires custom fine-tuning setup")
    logger.warning("Using base DistilBERT - may need model adaptation")
    
    # Experiment configuration
    model_name = EXPERIMENT_MODELS['experiment_4']['name']
    
    # Build dataset config directly (stratified sampling)
    dataset_config = {
        'config_key': 'ag_news',
        'sample_size': 2000,  # 500 per class
        'stratify': True
    }
    explainer_names = ['integrated_gradients', 'lime', 'shap']
    stress_test_names = ['seed_variation', 'preprocessing_variation', 'semantic_paraphrasing']
    experiment_name = 'experiment_4_distilbert_agnews'
    
    # Initialize assessor
    assessor = StabilityAssessor(
        results_dir='../results',
        random_seed=42,
        max_samples=500  # Limited until fine-tuned DistilBERT-AG model available
    )
    
    # Explainer parameters
    explainer_params = {
        'integrated_gradients': {'n_steps': 50},
        'lime': {'num_samples': 1000},
        'shap': {}
    }
    
    # Stress test parameters
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
        card_path = '../results/%s_stability_card.md' % experiment_name
        assessor.generate_stability_card(results, card_path)
        
        logger.info("Experiment 4 completed successfully!")
        logger.info("Results saved to: %s", assessor.results_dir)
        logger.info("Stability card: %s", card_path)
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT 4 SUMMARY")
        print("="*50)
        print("Model: %s" % model_name)
        print("Dataset: AG-News test stratified")
        print("Explainers: %d" % len(explainer_names))
        print("Stress tests: %d" % len(stress_test_names))
        print("Duration: %.1f seconds" % results['duration_seconds'])
        print("="*50)
        
    except Exception as e:
        logger.error("Experiment 4 failed: %s", str(e))
        logger.error("Note: May require fine-tuned DistilBERT model for AG-News")
        raise


if __name__ == "__main__":
    main()