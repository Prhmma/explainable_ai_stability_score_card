"""
Explainer Configuration - Centralized configuration for all explainers
"""

from typing import Dict, Any, NamedTuple

# Explanation method imports with availability checks
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ExplainerConfig(NamedTuple):
    """Configuration for an explainer - no hardcoded values!"""
    name: str
    available: bool
    default_params: Dict[str, Any]


# Explainer configurations - centralized
EXPLAINER_CONFIGS = {
    'integrated_gradients': ExplainerConfig(
        name='IntegratedGradients',
        available=CAPTUM_AVAILABLE,
        default_params={'n_steps': 50}
    ),
    'lime': ExplainerConfig(
        name='LIME',
        available=LIME_AVAILABLE,
        default_params={'num_samples': 1000}
    ),
    'shap': ExplainerConfig(
        name='SHAP',
        available=SHAP_AVAILABLE,
        default_params={}
    )
}
