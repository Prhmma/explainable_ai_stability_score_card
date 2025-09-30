"""
Stability Assessment Framework for NLP Explanations

This package provides tools for evaluating the stability of explanation methods
for text classification models, generating standardized stability cards for
academic research.
"""

from .model_loader import ModelLoader
from .dataset_loader import DatasetLoader
from .explainers import IntegratedGradientsExplainer, LIMEExplainer, SHAPExplainer
from .stress_tests import SeedVariationTest, PreprocessingVariationTest, SemanticParaphrasingTest
from .metrics import StabilityMetrics
from .stability_assessor import StabilityAssessor

__version__ = "1.0.0"
__author__ = "Academic Research Team"