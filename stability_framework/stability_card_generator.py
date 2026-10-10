"""
Stability Card Generator for Stability Assessment Framework

Generates publication-ready markdown stability cards with metrics,
interpretation guidelines, and statistical summaries.
"""

from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class StabilityCardGenerator:
    """Generator for markdown stability assessment cards."""
    
    def __init__(self):
        """Initialize the stability card generator."""
        pass
    
    def generate_card(self, results: Dict[str, Any]) -> str:
        """
        Generate a complete stability card from experiment results.
        
        Args:
            results: Complete experiment results dictionary
            
        Returns:
            Markdown-formatted stability card as string
        """
        card_sections = [
            self._generate_header(results),
            self._generate_metrics_summary_table(results),
            self._generate_class_breakdown(results),
            self._generate_detailed_metrics(results),
            self._generate_interpretation_guidelines(results),
            self._generate_statistical_summary(results),
            self._generate_technical_details(results),
            self._generate_footer()
        ]
        
        return '\n\n'.join(card_sections)
    
    def _generate_header(self, results: Dict[str, Any]) -> str:
        """Generate the header section."""
        config = results.get('configuration', {})
        dataset_info = results.get('dataset_info', {})

        model_name = config.get('model_name', 'Unknown Model')
        dataset_name = dataset_info.get('dataset_name', 'Unknown Dataset')
        loaded = dataset_info.get('total_samples', 'N/A')
        evaluated = dataset_info.get('evaluated_samples', loaded)
        explainer_names = ', '.join(config.get('explainer_names', []))
        timestamp = results.get('timestamp', 'Unknown')

        filter_info = dataset_info.get('correct_prediction_filter')
        if filter_info:
            population = (
                f"{evaluated} evaluated: {filter_info['kept']} of {filter_info['before']} loaded instances "
                f"were correctly classified (accuracy {filter_info['accuracy']:.3f}); "
                f"only correctly classified instances are assessed"
            )
        else:
            population = f"{evaluated} evaluated of {loaded} loaded (no correct-prediction filter)"

        return f"""# Stability Assessment Card

**Model**: {model_name}  
**Dataset**: {dataset_name}  
**Instances**: {population}  
**Explainers**: {explainer_names}  
**Generated**: {timestamp}  

## Overview

This stability card evaluates the consistency of explanation methods for the specified model-dataset combination under three categories of stress tests: explainer seed variation (stochastic explainers only), preprocessing variation, and semantic paraphrasing."""

    # (result key, column header) for the paper's three metrics
    SUMMARY_METRICS = [
        ('jaccard_at_5', 'J@5'),
        ('spearman_correlation', 'ρ'),
        ('attr_diff_relative_difference_percentage', 'AD (%)'),
    ]
    STRESS_TEST_ORDER = ['SeedVariation', 'PreprocessingVariation', 'SemanticParaphrasing']

    def _format_mean_std(self, metrics_data: Dict[str, Any], key: str) -> str:
        stats = metrics_data.get(key)
        if not stats:
            return "N/A"
        return f"{stats['mean']:.3f} ± {stats['std']:.3f}"

    def _generate_metrics_summary_table(self, results: Dict[str, Any]) -> str:
        """Generate the main summary table: one row per explainer and stress test (as in the paper's tables)."""
        header_cells = " | ".join(label for _, label in self.SUMMARY_METRICS)
        table = f"""## Stability Metrics Summary

Mean ± standard deviation over all pairwise comparisons between each instance's original and perturbed explanations.

| Explainer | Stress Test | {header_cells} | Flip-rate | Comparisons |
|-----------|-------------|{"|".join("---" for _ in self.SUMMARY_METRICS)}|-----------|-------------|"""

        by_explainer = results.get('results', {}).get('by_explainer', {})
        for explainer_name, explainer_data in by_explainer.items():
            for stress_test_name in self.STRESS_TEST_ORDER:
                metrics_data = explainer_data.get(stress_test_name)
                if metrics_data is None:
                    continue
                if 'error' in metrics_data:
                    table += f"\n| {explainer_name} | {self._format_stress_test_name(stress_test_name)} | {metrics_data['error']} |"
                    continue
                cells = " | ".join(self._format_mean_std(metrics_data, key) for key, _ in self.SUMMARY_METRICS)
                flip = metrics_data.get('flip_rate', {})
                flip_cell = f"{flip['flip_rate_percentage']:.1f}%" if 'flip_rate_percentage' in flip else "N/A"
                n = metrics_data.get('sample_statistics', {}).get('total_comparisons', 'N/A')
                table += f"\n| {explainer_name} | {self._format_stress_test_name(stress_test_name)} | {cells} | {flip_cell} | {n} |"

        table += """

*AD (%)*: attribution difference, Σ|a₁ᵢ − a₂ᵢ| / (n · max(‖a₁‖∞, ‖a₂‖∞)) × 100 over common words.
*Flip-rate*: share of comparisons where the prediction is stable (same class, confidence change ≤ 0.1) but the explanation is not (J@5 < 0.5 or ρ < 0.7)."""
        return table

    def _generate_class_breakdown(self, results: Dict[str, Any]) -> str:
        """Per-predicted-class results for multi-class tasks (paper, Section III.D)."""
        if results.get('model_info', {}).get('num_labels', 2) <= 2:
            return ""
        header_cells = " | ".join(label for _, label in self.SUMMARY_METRICS)
        table = f"""## Results by Predicted Class

| Explainer | Stress Test | Class | {header_cells} |
|-----------|-------------|-------|{"|".join("---" for _ in self.SUMMARY_METRICS)}|"""
        by_explainer = results.get('results', {}).get('by_explainer', {})
        for explainer_name, explainer_data in by_explainer.items():
            for stress_test_name in self.STRESS_TEST_ORDER:
                metrics_data = explainer_data.get(stress_test_name) or {}
                for label, class_data in metrics_data.get('by_predicted_class', {}).items():
                    cells = " | ".join(self._format_mean_std(class_data, key) for key, _ in self.SUMMARY_METRICS)
                    table += f"\n| {explainer_name} | {self._format_stress_test_name(stress_test_name)} | {label} | {cells} |"
        return table

    def _generate_detailed_metrics(self, results: Dict[str, Any]) -> str:
        """Generate detailed metrics breakdown."""
        detailed_section = "## Detailed Results\n"
        
        by_explainer = results.get('results', {}).get('by_explainer', {})
        
        for explainer_name, explainer_data in by_explainer.items():
            detailed_section += f"\n### {explainer_name}\n"
            
            for stress_test_name, metrics_data in explainer_data.items():
                if 'error' in metrics_data:
                    continue
                    
                detailed_section += f"\n**{self._format_stress_test_name(stress_test_name)}**\n"
                
                # Key metrics
                key_metrics = [
                    ('jaccard_at_5', 'Jaccard@5 Similarity'),
                    ('jaccard_at_3', 'Jaccard@3 Similarity'),
                    ('spearman_correlation', 'Spearman Correlation'),
                    ('attr_diff_mean_absolute_difference', 'Mean Attribution Difference'),
                    ('attr_diff_relative_difference_percentage', 'Relative Difference %')
                ]
                
                for metric_key, metric_name in key_metrics:
                    if metric_key in metrics_data:
                        stats = metrics_data[metric_key]
                        mean_val = stats['mean']
                        std_val = stats['std']
                        ci_low, ci_high = stats.get('ci_95', (mean_val, mean_val))
                        
                        detailed_section += f"- {metric_name}: {mean_val:.3f} ± {std_val:.3f} (CI: {ci_low:.3f}, {ci_high:.3f})\n"
                
                # Sample statistics
                if 'sample_statistics' in metrics_data:
                    stats = metrics_data['sample_statistics']
                    detailed_section += f"- Samples processed: {stats['successful_samples']}/{stats['total_samples']}\n"
                    detailed_section += f"- Total comparisons: {stats['total_comparisons']}\n"
        
        return detailed_section
    
    def _generate_interpretation_guidelines(self, results: Dict[str, Any]) -> str:
        """Generate interpretation guidelines section."""
        guidelines = """## Interpretation Guidelines

### Stability Metrics Interpretation

The bands below are heuristic reading aids, not empirically calibrated thresholds.

**Jaccard@K Similarity (0.0 - 1.0)**
- > 0.8: Highly stable explanations
- 0.6 - 0.8: Moderately stable explanations  
- 0.4 - 0.6: Low stability, some consistency
- < 0.4: Highly unstable explanations

**Spearman Rank Correlation (-1.0 - 1.0)**
- > 0.8: Strong ranking consistency
- 0.6 - 0.8: Moderate ranking consistency
- 0.3 - 0.6: Weak ranking consistency
- < 0.3: Poor ranking consistency

**Attribution Difference (AD, %)**
- Lower values indicate more stable attribution magnitudes
- Normalized by word count and the larger maximum |attribution|, so comparable across inputs and methods

### Stress Test Interpretation

**Seed Variation**: Tests fundamental reproducibility of the explanation method. High instability suggests algorithmic non-determinism.

**Preprocessing Variation**: Tests robustness to common text preprocessing. High instability suggests sensitivity to superficial text changes.

**Semantic Paraphrasing**: Tests consistency with meaning-preserving changes. High instability suggests over-sensitivity to surface form."""
        
        # Add specific recommendations based on results
        recommendations = self._generate_recommendations(results)
        if recommendations:
            guidelines += f"\n\n### Recommendations\n\n{recommendations}"
        
        return guidelines
    
    def _generate_statistical_summary(self, results: Dict[str, Any]) -> str:
        """Generate statistical summary section."""
        config = results.get('configuration', {})
        dataset_info = results.get('dataset_info', {})
        
        total_samples = dataset_info.get('evaluated_samples', dataset_info.get('total_samples', 'N/A'))
        duration = results.get('duration_seconds', 0)
        avg_time_per_sample = duration / max(1, int(total_samples)) if isinstance(total_samples, int) else 0
        
        summary = f"""## Statistical Summary

**Total samples analyzed**: {total_samples}  
**Processing time**: {duration:.1f} seconds  
**Mean processing time**: {avg_time_per_sample:.1f} seconds per sample  
**Random seed**: {config.get('random_seed', 'N/A')}  
**Confidence intervals**: 95% (t-distribution)  

### Experimental Setup
- **Explainers**: {len(config.get('explainer_names', []))} methods tested
- **Stress tests**: {len(config.get('stress_test_names', []))} categories applied
- **Device**: {results.get('model_info', {}).get('device', 'N/A')}
- **Model parameters**: {results.get('model_info', {}).get('num_parameters', 'N/A'):,}"""
        
        return summary
    
    def _generate_technical_details(self, results: Dict[str, Any]) -> str:
        """Generate technical details section."""
        model_info = results.get('model_info', {})
        config = results.get('configuration', {})
        
        details = f"""## Technical Details

### Model Configuration
- **Architecture**: {model_info.get('model_type', 'Unknown')}
- **Parameters**: {model_info.get('num_parameters', 'N/A'):,}
- **Vocabulary size**: {model_info.get('vocab_size', 'N/A'):,}
- **Max sequence length**: {model_info.get('max_length', 'N/A')}
- **Device**: {model_info.get('device', 'N/A')}

### Experiment Parameters
- **Explainer parameters**: {config.get('explainer_params', {})}
- **Stress test parameters**: {config.get('stress_test_params', {})}

### Limitations
- Results are specific to the tested model and dataset
- Semantic similarity filtering may not capture all nuances
- Processing time depends on hardware configuration
- Statistical significance may vary with sample size"""
        
        return details
    
    def _generate_footer(self) -> str:
        """Generate footer section."""
        footer = f"""---

*Generated by Stability Assessment Framework v1.0*  
*Report timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  

Mohammadalizadeh, P., Mohammadalizadeh, P., and Mahmoudian, A. "A Framework for Systematic Stability Assessment of Post-hoc Explanations in Text Classification." 16th International Conference on Information and Knowledge Technology (IKT), 2025."""
        
        return footer
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate specific recommendations based on results."""
        recommendations = []
        
        # Analyze results to generate recommendations
        by_explainer = results.get('results', {}).get('by_explainer', {})
        
        for explainer_name, explainer_data in by_explainer.items():
            explainer_issues = []
            
            for stress_test_name, metrics_data in explainer_data.items():
                if 'jaccard_at_5' in metrics_data:
                    jaccard_mean = metrics_data['jaccard_at_5']['mean']
                    if jaccard_mean < 0.4:
                        explainer_issues.append(f"low stability under {stress_test_name}")
                    elif jaccard_mean < 0.6:
                        explainer_issues.append(f"moderate instability under {stress_test_name}")
            
            if explainer_issues:
                recommendations.append(f"**{explainer_name}**: Shows {', '.join(explainer_issues)}. Consider ensemble methods or parameter tuning.")
            else:
                recommendations.append(f"**{explainer_name}**: Shows good stability across tested conditions.")
        
        return '\n'.join(recommendations) if recommendations else ""
    
    def _format_stress_test_name(self, name: str) -> str:
        """Format stress test names for display."""
        mapping = {
            'SeedVariation': 'Seed Variation',
            'PreprocessingVariation': 'Preprocessing',
            'SemanticParaphrasing': 'Paraphrasing'
        }
        return mapping.get(name, name)
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5