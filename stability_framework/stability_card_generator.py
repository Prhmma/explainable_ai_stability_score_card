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
        model_info = results.get('model_info', {})
        dataset_info = results.get('dataset_info', {})
        
        model_name = config.get('model_name', 'Unknown Model')
        dataset_name = dataset_info.get('dataset_name', 'Unknown Dataset')
        total_samples = dataset_info.get('total_samples', 'N/A')
        explainer_names = ', '.join(config.get('explainer_names', []))
        timestamp = results.get('timestamp', 'Unknown')
        
        header = f"""# Stability Assessment Card

**Model**: {model_name}  
**Dataset**: {dataset_name} ({total_samples} examples)  
**Explainers**: {explainer_names}  
**Generated**: {timestamp}  

## Overview

This stability card evaluates the reliability and consistency of explanation methods for the specified model-dataset combination. The assessment uses three categories of stress tests to measure explanation stability across different perturbation scenarios."""
        
        return header
    
    def _generate_metrics_summary_table(self, results: Dict[str, Any]) -> str:
        """Generate the main metrics summary table."""
        summary_table = """## Stability Metrics Summary

| Stress Test | Jaccard@5 | Spearman | Flip-Rate | Attr-Diff |
|-------------|-----------|----------|-----------|-----------|"""
        
        # Get results by stress test
        by_stress_test = results.get('results', {}).get('by_stress_test', {})
        
        for stress_test_name in ['SeedVariation', 'PreprocessingVariation', 'SemanticParaphrasing']:
            if stress_test_name not in by_stress_test:
                continue
                
            # Aggregate across all explainers for this stress test
            jaccard_values = []
            spearman_values = []
            attr_diff_values = []
            
            for explainer_name, explainer_data in by_stress_test[stress_test_name].items():
                if 'jaccard_at_5' in explainer_data:
                    jaccard_values.append(explainer_data['jaccard_at_5']['mean'])
                if 'spearman_correlation' in explainer_data:
                    spearman_values.append(explainer_data['spearman_correlation']['mean'])
                if 'attr_diff_mean_absolute_difference' in explainer_data:
                    attr_diff_values.append(explainer_data['attr_diff_mean_absolute_difference']['mean'])
            
            # Calculate means and standard deviations
            jaccard_mean = sum(jaccard_values) / len(jaccard_values) if jaccard_values else 0
            jaccard_std = self._calculate_std(jaccard_values) if len(jaccard_values) > 1 else 0
            
            spearman_mean = sum(spearman_values) / len(spearman_values) if spearman_values else 0
            spearman_std = self._calculate_std(spearman_values) if len(spearman_values) > 1 else 0
            
            attr_diff_mean = sum(attr_diff_values) / len(attr_diff_values) if attr_diff_values else 0
            
            # Format row
            display_name = self._format_stress_test_name(stress_test_name)
            jaccard_cell = f"{jaccard_mean:.2f} ± {jaccard_std:.2f}"
            spearman_cell = f"{spearman_mean:.2f} ± {spearman_std:.2f}"
            flip_rate_cell = "N/A"  # TODO: Calculate flip rate
            attr_diff_cell = f"{attr_diff_mean:.3f}"
            
            summary_table += f"""
| {display_name} | {jaccard_cell} | {spearman_cell} | {flip_rate_cell} | {attr_diff_cell} |"""
        
        return summary_table
    
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

**Attribution Difference**
- Lower values indicate more stable attributions
- Compare relative differences across methods

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
        
        total_samples = dataset_info.get('total_samples', 'N/A')
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

For academic use, please cite the associated publication and methodology."""
        
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