"""
Main Stability Assessor Module for Stability Assessment Framework

Orchestrates the entire stability assessment process, coordinating model loading,
dataset processing, explanation generation, stress testing, and metric calculation.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import torch
import numpy as np

try:
    from .model_loader import ModelLoader
    from .dataset_loader import DatasetLoader
    from .explainers import create_explainer, EXPLAINER_CONFIGS
    from .stress_tests import create_stress_test, AVAILABLE_STRESS_TESTS
    from .metrics import StabilityMetrics
except ImportError:
    # Fallback to absolute imports when running as script
    from model_loader import ModelLoader
    from dataset_loader import DatasetLoader
    from explainers import create_explainer, EXPLAINER_CONFIGS
    from stress_tests import create_stress_test, AVAILABLE_STRESS_TESTS
    from metrics import StabilityMetrics

logger = logging.getLogger(__name__)


class StabilityAssessor:
    """
    Main class for conducting stability assessments of explanation methods.
    """
    
    def __init__(
        self, 
        results_dir: str = "results",
        random_seed: int = 42,
        device: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the stability assessor.
        
        Args:
            results_dir: Directory to save results
            random_seed: Random seed for reproducibility
            device: Target device ('cuda', 'cpu', or None for auto)
            max_samples: Maximum number of samples to process (None for all)
        """
        self.results_dir = results_dir
        self.random_seed = random_seed
        self.device = device
        self.max_samples = max_samples
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.model_loader = ModelLoader(device=device)
        self.dataset_loader = DatasetLoader(random_seed=random_seed)
        self.metrics_calculator = StabilityMetrics()
        
        # State tracking
        self.current_experiment = None
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.loaded_dataset = None
        
        logger.info(f"StabilityAssessor initialized with seed {random_seed}")
    
    def run_experiment(
        self,
        model_name: str,
        dataset_config: Dict[str, Any],
        explainer_names: List[str],
        stress_test_names: List[str],
        experiment_name: str,
        explainer_params: Optional[Dict[str, Any]] = None,
        stress_test_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete stability experiment.
        
        Args:
            model_name: HuggingFace model identifier
            dataset_config: Dataset loading configuration
            explainer_names: List of explanation methods to use
            stress_test_names: List of stress tests to apply
            experiment_name: Name for this experiment
            explainer_params: Parameters for explainers
            stress_test_params: Parameters for stress tests
            run_only_combinations: Optional list of (explainer, stress_test) tuples to run only specific combinations
                                   Example: [("lime", "seed_variation"), ("lime", "preprocessing_variation")]
                                   If None, runs all combinations
            
        Returns:
            Complete experiment results
        """
        start_time = time.time()
        self.current_experiment = experiment_name
        
        logger.info(f"Starting experiment: {experiment_name}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Explainers: {explainer_names}")
        logger.info(f"Stress tests: {stress_test_names}")
        
        try:
            # Step 1: Load model and tokenizer
            logger.info("Step 1: Loading model and tokenizer")
            model, tokenizer = self.model_loader.load_model_and_tokenizer(model_name)
            self.loaded_model = model
            self.loaded_tokenizer = tokenizer
            
            model_info = self.model_loader.get_model_info(model, tokenizer)
            logger.info(f"Model loaded: {model_info['model_type']} with {model_info['num_parameters']:,} parameters")
            
            # Step 2: Load dataset
            logger.info("Step 2: Loading dataset")
            texts, labels, dataset_metadata = self._load_dataset(dataset_config)
            self.loaded_dataset = (texts, labels, dataset_metadata)
            
            # Limit samples if specified
            if self.max_samples and len(texts) > self.max_samples:
                indices = np.random.choice(len(texts), self.max_samples, replace=False)
                texts = [texts[i] for i in indices]
                labels = [labels[i] for i in indices]
                logger.info(f"Limited to {self.max_samples} samples")
            
            logger.info(f"Dataset loaded: {len(texts)} samples")
            
            # Step 3: Initialize explainers and stress tests
            logger.info("Step 3: Initializing explainers and stress tests")
            explainers = self._initialize_explainers(explainer_names, explainer_params or {})
            stress_tests = self._initialize_stress_tests(stress_test_names, stress_test_params or {})
            
            # Step 4: Run stability assessment
            logger.info("Step 4: Running stability assessment")
            assessment_results = self._run_stability_assessment(
                texts, labels, explainers, stress_tests, experiment_name
            )
            
            # Step 5: Compile final results
            end_time = time.time()
            duration = end_time - start_time
            
            final_results = {
                'experiment_name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'model_info': model_info,
                'dataset_info': dataset_metadata,
                'configuration': {
                    'model_name': model_name,
                    'explainer_names': explainer_names,
                    'stress_test_names': stress_test_names,
                    'random_seed': self.random_seed,
                    'max_samples': self.max_samples,
                    'explainer_params': explainer_params,
                    'stress_test_params': stress_test_params
                },
                'results': assessment_results
            }
            
            # Step 6: Save results
            result_file = self._save_results(final_results)
            logger.info(f"Experiment completed in {duration:.1f}s")
            logger.info(f"Results saved to: {result_file}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise
        finally:
            # Clean up GPU memory
            self.model_loader.clear_memory()
    
    def _load_dataset(self, dataset_config: Dict[str, Any]) -> Tuple[List[str], List[int], Dict]:
        """Load dataset based on configuration."""
        from .dataset_loader import DATASET_CONFIGS
        
        config_key = dataset_config['config_key']
        sample_size = dataset_config.get('sample_size')
        stratify = dataset_config.get('stratify', False)
        
        config = DATASET_CONFIGS[config_key]
        return self.dataset_loader.load_dataset(config, sample_size=sample_size, stratify=stratify)
    
    def _initialize_explainers(
        self, 
        explainer_names: List[str], 
        params: Dict[str, Any]
    ) -> List[Any]:
        """Initialize explanation methods."""
        explainers = []
        
        for name in explainer_names:
            if name not in EXPLAINER_CONFIGS:
                logger.warning(f"Unknown explainer: {name}")
                continue
                
            config = EXPLAINER_CONFIGS[name]
            if not config.available:
                logger.warning(f"Explainer {name} is not available (missing dependencies)")
                continue
            
            # Get parameters for this explainer (merge defaults with user params)
            explainer_params = {**config.default_params, **params.get(name, {})}
            
            explainer = create_explainer(name, random_seed=self.random_seed, **explainer_params)
            explainers.append(explainer)
            logger.info(f"Initialized explainer: {name}")
        
        if not explainers:
            raise ValueError("No valid explainers could be initialized")
            
        return explainers
    
    def _initialize_stress_tests(
        self, 
        stress_test_names: List[str], 
        params: Dict[str, Any]
    ) -> List[Any]:
        """Initialize stress tests."""
        stress_tests = []
        
        for name in stress_test_names:
            if name not in AVAILABLE_STRESS_TESTS:
                logger.warning(f"Unknown stress test: {name}")
                continue
                
            if not AVAILABLE_STRESS_TESTS[name].get('available', True):
                logger.warning(f"Stress test {name} is not available (missing dependencies)")
                continue
            
            # Get parameters for this stress test
            test_params = params.get(name, {})
            test_params.update(AVAILABLE_STRESS_TESTS[name].get('parameters', {}))
            
            stress_test = create_stress_test(name, random_seed=self.random_seed, **test_params)
            stress_tests.append(stress_test)
            logger.info(f"Initialized stress test: {name}")
        
        if not stress_tests:
            raise ValueError("No valid stress tests could be initialized")
            
        return stress_tests
    
    def _run_stability_assessment(
        self,
        texts: List[str],
        labels: List[int], 
        explainers: List[Any],
        stress_tests: List[Any],
        experiment_name: str
    ) -> Dict[str, Any]:
        """Run the main stability assessment."""
        results = {
            'by_explainer': {},
            'by_stress_test': {},
            'summary': {},
            'individual_results': []
        }
        
        total_combinations = len(explainers) * len(stress_tests) * len(texts)
        logger.info(f"Processing {total_combinations} combinations...")
        
        # Process each explainer
        for explainer in explainers:
            explainer_name = explainer.name
            results['by_explainer'][explainer_name] = {}
            
            # Process each stress test
            for stress_test in stress_tests:
                stress_test_name = stress_test.name
                
                # Skip seed variation for non-LIME explainers (seed variation only applies to LIME)
                if stress_test_name == 'SeedVariation' and explainer_name.upper() != 'LIME':
                    logger.info(f"Skipping {stress_test_name} for {explainer_name} (only applicable to LIME)")
                    continue
                
                logger.info(f"Running {explainer_name} with {stress_test_name}")
                
                # Process samples
                sample_results = []
                
                # Create progress bar
                pbar = tqdm(
                    enumerate(texts), 
                    total=len(texts),
                    desc=f"{explainer_name}-{stress_test_name}"
                )
                
                for sample_idx, text in pbar:
                    try:
                        sample_result = self._process_sample(
                            text, labels[sample_idx], explainer, stress_test, sample_idx
                        )
                        sample_results.append(sample_result)
                        
                        # Update progress bar with current metrics
                        if sample_result and 'metrics' in sample_result:
                            avg_jaccard = np.mean([
                                m.get('jaccard_at_5', 0) 
                                for m in sample_result['metrics']
                            ])
                            pbar.set_postfix({'avg_jaccard_5': f'{avg_jaccard:.3f}'})
                            
                    except Exception as e:
                        logger.error(f"Error processing sample {sample_idx}: {str(e)}")
                        continue
                        
                    # Memory management: clear cache periodically
                    if sample_idx % 50 == 0:
                        self.model_loader.clear_memory()
                
                pbar.close()
                
                # Aggregate results for this explainer-stress test combination
                aggregated = self._aggregate_sample_results(sample_results)
                results['by_explainer'][explainer_name][stress_test_name] = aggregated
                
                # Store in by_stress_test format too
                if stress_test_name not in results['by_stress_test']:
                    results['by_stress_test'][stress_test_name] = {}
                results['by_stress_test'][stress_test_name][explainer_name] = aggregated
                
                # Save intermediate results after each combination
                self._save_intermediate_results(results, experiment_name, explainer_name, stress_test_name)
                
                logger.info(f"Completed {explainer_name} with {stress_test_name} - intermediate results saved")
        
        # Generate summary across all combinations
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _process_sample(
        self,
        text: str,
        label: int,
        explainer: Any,
        stress_test: Any,
        sample_idx: int
    ) -> Optional[Dict[str, Any]]:
        """Process a single text sample through the stability assessment."""
        try:
            # Generate original explanation
            original_explanation = explainer.explain(
                self.loaded_model, self.loaded_tokenizer, text
            )
            
            # Apply stress test to generate variants
            if stress_test.name == 'SeedVariation':
                # Special handling for seed variation - same text, different seeds
                seeds = stress_test.get_seeds()
                variant_texts = [text] * len(seeds)  # Same text repeated for each seed
            else:
                variant_texts = stress_test.apply(text)
                seeds = [self.random_seed] * len(variant_texts)
            
            # Generate explanations for variants
            variant_explanations = []
            for i, (variant_text, seed) in enumerate(zip(variant_texts, seeds)):
                if stress_test.name == 'SeedVariation':
                    # For seed variation, create new explainer instance with different seed
                    # Get original explainer parameters
                    explainer_params = {}
                    if hasattr(explainer, 'n_steps'):
                        explainer_params['n_steps'] = explainer.n_steps
                    if hasattr(explainer, 'num_samples'):
                        explainer_params['num_samples'] = explainer.num_samples
                    
                    # Create explainer with different seed
                    variant_explainer = create_explainer(
                        explainer.name.lower(), 
                        random_seed=seed,
                        **explainer_params
                    )
                    variant_exp = variant_explainer.explain(
                        self.loaded_model, self.loaded_tokenizer, variant_text
                    )
                else:
                    # Use same explainer for non-seed variations
                    variant_exp = explainer.explain(
                        self.loaded_model, self.loaded_tokenizer, variant_text
                    )
                variant_explanations.append(variant_exp)
            
            # Calculate stability metrics
            metrics = []
            for variant_exp in variant_explanations[1:]:  # Skip first (original)
                metric_result = self.metrics_calculator.calculate_all_metrics(
                    original_explanation, variant_exp
                )
                metrics.append(metric_result)
            
            return {
                'sample_idx': sample_idx,
                'original_text': text,
                'true_label': label,
                'stress_test': stress_test.name,
                'explainer': explainer.name,
                'original_explanation': original_explanation,
                'variant_explanations': variant_explanations,
                'variant_texts': variant_texts,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in _process_sample for sample {sample_idx}: {str(e)}")
            return None
    
    def _aggregate_sample_results(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple samples."""
        if not sample_results:
            return {'error': 'No valid sample results'}
        
        # Extract all metrics
        all_metrics = []
        for sample_result in sample_results:
            if sample_result and 'metrics' in sample_result:
                all_metrics.extend(sample_result['metrics'])
        
        if not all_metrics:
            return {'error': 'No valid metrics computed'}
        
        # Aggregate metrics
        aggregated = self.metrics_calculator.aggregate_metrics(all_metrics)
        
        # Add sample-level statistics
        aggregated['sample_statistics'] = {
            'total_samples': len(sample_results),
            'successful_samples': len([r for r in sample_results if r is not None]),
            'total_comparisons': len(all_metrics)
        }
        
        return aggregated
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level summary of all results."""
        summary = {
            'explainers_tested': list(results['by_explainer'].keys()),
            'stress_tests_applied': list(results['by_stress_test'].keys()),
            'key_findings': {}
        }
        
        # Extract key metrics for summary
        key_metrics = ['jaccard_at_5', 'spearman_correlation', 'attr_diff_mean_absolute_difference']
        
        for explainer in summary['explainers_tested']:
            explainer_summary = {}
            
            for stress_test in summary['stress_tests_applied']:
                if stress_test in results['by_explainer'][explainer]:
                    test_data = results['by_explainer'][explainer][stress_test]
                    
                    test_summary = {}
                    for metric in key_metrics:
                        if metric in test_data:
                            test_summary[metric] = {
                                'mean': test_data[metric]['mean'],
                                'std': test_data[metric]['std']
                            }
                    
                    explainer_summary[stress_test] = test_summary
            
            summary['key_findings'][explainer] = explainer_summary
        
        return summary
    
    def _normalize_stress_test_name(self, name: str) -> str:
        """
        Normalize stress test names to handle different naming conventions.
        Maps class names like 'SeedVariation' to user-friendly names like 'seed_variation'.
        """
        name_lower = name.lower()
        
        # Map class names to standardized names
        name_mapping = {
            'seedvariation': 'seed_variation',
            'preprocessingvariation': 'preprocessing_variation', 
            'semanticparaphrasing': 'semantic_paraphrasing'
        }
        
        return name_mapping.get(name_lower, name_lower)
    
    def _save_intermediate_results(
        self, 
        results: Dict[str, Any], 
        experiment_name: str, 
        explainer_name: str, 
        stress_test_name: str
    ) -> str:
        """Save intermediate results after each explainer-stress test combination."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{explainer_name}_{stress_test_name}_{timestamp}_intermediate.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Create a focused intermediate result structure
        intermediate_result = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'explainer': explainer_name,
            'stress_test': stress_test_name,
            'status': 'intermediate',
            'completed_combination': f"{explainer_name} + {stress_test_name}",
            'partial_results': {
                'by_explainer': {
                    explainer_name: {
                        stress_test_name: results['by_explainer'].get(explainer_name, {}).get(stress_test_name, {})
                    }
                },
                'by_stress_test': {
                    stress_test_name: {
                        explainer_name: results['by_stress_test'].get(stress_test_name, {}).get(explainer_name, {})
                    }
                }
            }
        }
        
        # Convert numpy types for JSON serialization
        intermediate_serializable = self._make_serializable(intermediate_result)
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(intermediate_serializable, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Intermediate results saved: {filename}")
        return filepath
    
    def _save_results(self, results: Dict[str, Any]) -> str:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results['experiment_name']}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert numpy types to Python native types for JSON serialization
        results_serializable = self._make_serializable(results)
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        else:
            return obj
    
    def generate_stability_card(
        self, 
        results: Dict[str, Any], 
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate markdown stability card from results.
        
        Args:
            results: Experiment results dictionary
            output_path: Optional path to save the card
            
        Returns:
            Markdown content as string
        """
        try:
            from .stability_card_generator import StabilityCardGenerator
        except ImportError:
            from stability_card_generator import StabilityCardGenerator
        
        generator = StabilityCardGenerator()
        card_content = generator.generate_card(results)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(card_content)
            logger.info(f"Stability card saved to: {output_path}")
        
        return card_content