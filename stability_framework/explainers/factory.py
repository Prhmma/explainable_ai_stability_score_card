"""
Explainer Factory - Create explainer instances from configuration
"""

from .base import BaseExplainer
from .config import EXPLAINER_CONFIGS
from .integrated_gradients import IntegratedGradientsExplainer
from .lime_explainer import LIMEExplainer
from .shap_explainer import SHAPExplainer


def create_explainer(method_name: str, random_seed: int = 42, **kwargs) -> BaseExplainer:
    """
    Factory function to create explainers using configuration.
    
    Args:
        method_name: Name of the explainer ('integrated_gradients', 'lime', 'shap')
        random_seed: Random seed for reproducibility
        **kwargs: Override default parameters
    """
    method_name = method_name.lower().replace(' ', '_')
    
    # Map aliases to canonical names
    if method_name in ['ig', 'integratedgradients']:
        method_name = 'integrated_gradients'
    
    if method_name not in EXPLAINER_CONFIGS:
        raise ValueError(f"Unknown explainer: {method_name}. Available: {list(EXPLAINER_CONFIGS.keys())}")
    
    config = EXPLAINER_CONFIGS[method_name]
    
    if not config.available:
        raise ImportError(f"{config.name} is not available. Install required dependencies.")
    
    # Merge default params with user overrides
    params = {**config.default_params, **kwargs}
    
    if method_name == 'integrated_gradients':
        return IntegratedGradientsExplainer(random_seed=random_seed, **params)
    elif method_name == 'lime':
        return LIMEExplainer(random_seed=random_seed, **params)
    elif method_name == 'shap':
        return SHAPExplainer(random_seed=random_seed, **params)
