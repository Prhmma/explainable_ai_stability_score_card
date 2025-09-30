# Stability Assessment Framework for NLP Explanations

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)

Evaluate the stability of explanation methods for NLP models with reproducible experiments and publication-ready stability cards. The framework orchestrates stress tests across seeds, preprocessing, and semantic perturbations while tracking standard metrics such as Jaccard@K, Spearman, flip-rate, and attribution difference.

## Position in Explainable AI

Explainability research has largely focused on interpreting a single explanation in isolation. This framework emphasizes **stability**—how consistent an explainer remains when the underlying data or model is perturbed—which is a critical yet underreported dimension of trustworthy AI. By benchmarking multiple explainers and models under controlled stress tests, the project helps researchers:

- Diagnose brittle behavior that might otherwise go unnoticed when explanations fluctuate across runs or pre-processing tweaks.
- Produce reproducible, quantitative evidence that complements qualitative explanation analysis.
- Compare explanation techniques with a common set of metrics, improving transparency in academic reporting.

In practice, the stability cards produced here support responsible AI efforts by surfacing when explanations align with model behavior and when they diverge, allowing practitioners to set reliability thresholds before deploying models in sensitive domains.

## Highlights

- Supports popular text classifiers out of the box: DistilBERT, RoBERTa, BERT, and T5.
- Compares LIME, SHAP, and Integrated Gradients explanations under multiple stress regimes.
- Produces consistent markdown stability cards for academic reporting.
- Ships with experiment runners, metrics, and explainers that can be extended for custom use cases.
- Uses the [UV](https://github.com/astral-sh/uv) package manager for isolated, reproducible environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Run Your First Experiment](#run-your-first-experiment)
- [Available Experiments](#available-experiments)
- [Understanding the Metrics](#understanding-the-metrics)
- [Example Stability Card](#example-stability-card)
- [Customize Experiments](#customize-experiments)
- [Project Structure](#project-structure)
- [Development Tasks](#development-tasks)
- [License](#license)
- [Citation](#citation)

## Quick Start

### Prerequisites

- Python 3.12 or later
- CUDA-capable GPU (RTX 3070 or better recommended)
- 16 GB RAM and ~3 GB of free disk space

### Installation Steps

1. **Install UV (Python package manager).**

   ```powershell
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository and enter the project folder.**

   ```powershell
   git clone https://github.com/Prhmma/explainable_ai_stability_score_card.git
   cd explainable_ai_stability_score_card
   ```

3. **Sync dependencies (creates a virtual environment automatically).**

   ```powershell
   uv sync
   ```

4. **Download the required NLTK resources for paraphrasing stress tests.**

   ```powershell
   uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   ```

5. **Run the smoke tests to confirm the environment is ready.**

   ```powershell
   uv run pytest tests/test_simple.py -v
   ```

## Run Your First Experiment

Execute the DistilBERT + SST-2 benchmark runner to generate a stability card.

```powershell
uv run python experiment_runners/experiment_1_distilbert_sst2.py
```

- Estimated runtime: 2–4 hours on an RTX 3070 GPU
- Output: `results/experiment_1_distilbert_sst2_stability_card.md`

## Available Experiments

| # | Model | Dataset | Samples | Status | Runtime | Command |
|---|-------|---------|---------|--------|---------|---------|
| 1 | DistilBERT | SST-2 | 872 | ✅ Ready | 2–4 h | `experiment_1_distilbert_sst2.py` |
| 2 | T5-small | SST-2 | 872 | ⚠️ Custom fine-tuning | 3–5 h | `experiment_2_t5_sst2.py` |
| 3 | RoBERTa | AG News | 2000 | ✅ Ready | 3–5 h | `experiment_3_roberta_agnews.py` |
| 4 | DistilBERT | AG News | 2000 | ⚠️ Custom fine-tuning | 3–5 h | `experiment_4_distilbert_agnews.py` |

✅ **Ready** experiments ship with pre-finetuned weights. ⚠️ **Custom** experiments expect you to provide or train compatible checkpoints.

## Understanding the Metrics

| Metric | Range | Interpretation | Stable When |
|--------|-------|----------------|-------------|
| **Jaccard@K** | 0 – 1 | Overlap of top-K important tokens | > 0.5 |
| **Spearman** | –1 – 1 | Correlation of attribution rankings | > 0.7 |
| **Flip-Rate** | 0 – 100 % | Prediction unchanged but explanation altered | < 10 % |
| **Attribution Difference** | 0 – ∞ | Mean absolute change in token attribution | < 0.1 |

## Example Stability Card

```markdown
# Stability Assessment Card

**Model**: distilbert-base-uncased-finetuned-sst-2-english  
**Dataset**: SST-2 (872 examples)  
**Explainers**: IntegratedGradients, LIME, SHAP

| Stress Test      | Jaccard@5 | Spearman | Flip-Rate | Attr-Diff |
|------------------|-----------|----------|-----------|-----------|
| Seed Variation   | 0.82 ± 0.05 | 0.91 ± 0.03 | 2.1%  | 0.023 |
| Preprocessing    | 0.67 ± 0.12 | 0.78 ± 0.08 | 8.4%  | 0.087 |
| Paraphrasing     | 0.54 ± 0.15 | 0.69 ± 0.11 | 15.2% | 0.134 |
```

## Customize Experiments

### Add a New Model

```python
from stability_framework.stability_assessor import StabilityAssessor
from stability_framework.model_loader import EXPERIMENT_MODELS

EXPERIMENT_MODELS["my_model"] = {
    "name": "your-hf-model-name",
    "type": "bert",
    "task": "sentiment",
    "num_labels": 2,
}

assessor = StabilityAssessor(results_dir="./results", random_seed=42)

assessor.run_experiment(
    model_name="my_model",
    dataset_config={"config_key": "sst2", "sample_size": 100},
    explainer_names=["integrated_gradients", "lime", "shap"],
    stress_test_names=["seed_variation", "preprocessing_variation"],
    experiment_name="my_experiment",
)
```

### Register a Custom Dataset

```python
from stability_framework.dataset_loader import DATASET_CONFIGS, DatasetConfig

DATASET_CONFIGS["my_dataset"] = DatasetConfig(
    dataset_name="your_dataset_name",
    split="test",
    text_column="text",
    label_column="label",
    display_name="My Dataset",
)
```

## Project Structure

```text
stability_framework/
├── explainers/            # LIME, SHAP, Integrated Gradients implementations
├── metrics/               # Metric calculators and aggregation utilities
├── dataset_loader.py      # Dataset registration and sampling helpers
├── model_loader.py        # Pre-configured Hugging Face models
├── stability_assessor.py  # Experiment orchestration entry point
├── stability_card_generator.py  # Markdown report builder
└── stress_tests.py        # Perturbation strategies (seed, preprocessing, paraphrasing)

experiment_runners/
├── experiment_1_distilbert_sst2.py
├── experiment_2_t5_sst2.py
├── experiment_3_roberta_agnews.py
└── experiment_4_distilbert_agnews.py

tests/
├── test_simple.py
└── test_integration.py
```

## Development Tasks

```powershell
# Fast unit tests (~10 seconds)
uv run pytest tests/test_simple.py -v

# Full test suite with coverage report
uv run pytest --cov=stability_framework --cov-report=html

# Static analysis
uv run pylint stability_framework/ --max-line-length=100

# Manage dependencies
uv add <package-name>
uv add --dev <dev-package>
uv sync --upgrade
```
Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this framework in your research, please cite it using the entry in [CITATION.cff](CITATION.cff).
