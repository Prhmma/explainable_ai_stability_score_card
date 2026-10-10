# Stability Assessment Framework for NLP Explanations

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Implementation of **"A Framework for Systematic Stability Assessment of Post-hoc Explanations in Text Classification"** (P. Mohammadalizadeh, P. Mohammadalizadeh, A. Mahmoudian; IKT 2025).

The framework measures how consistent post-hoc explanations (Integrated Gradients, LIME, SHAP) remain when the input is perturbed in ways that should not matter: preprocessing changes, meaning-preserving paraphrases, and, for stochastic explainers, a different random seed. Results are summarised with Jaccard@k, Spearman rank correlation and a normalised attribution difference, and reported as markdown **stability cards**.

> **Note on reproducing the paper.** Version 1.1.0 fixes defects in the code first released with the paper and aligns it with the protocol the paper describes. Some fixes change the numbers the code produces, so results are not expected to match Tables I and II exactly. See [CHANGELOG.md](CHANGELOG.md) for each change and the results it affects.

## Table of Contents

- [Quick Start](#quick-start)
- [Reproducing the Paper's Experiments](#reproducing-the-papers-experiments)
- [Assessment Protocol](#assessment-protocol)
- [Metrics](#metrics)
- [Stability Cards](#stability-cards)
- [Known Limitations](#known-limitations)
- [Customize Experiments](#customize-experiments)
- [Project Structure](#project-structure)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

## Quick Start

### Prerequisites

- Python 3.12 or 3.13
- [uv](https://github.com/astral-sh/uv) package manager
- A CUDA GPU is strongly recommended for full runs; CPU works for smoke tests

### Installation

```bash
# Install uv (macOS / Linux; see the uv docs for Windows)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/Prhmma/explainable_ai_stability_score_card.git
cd explainable_ai_stability_score_card

# Creates .venv with the exact library versions in uv.lock
uv sync
```

Dependencies are resolved as they existed on 2025-09-30 (`exclude-newer` in `pyproject.toml`) and pinned in `uv.lock`, so `uv sync` reproduces the same environment (PyTorch 2.8, Transformers 4.56, SHAP 0.48, Captum 0.8, LIME 0.2). NLTK data (WordNet, POS tagger, stopwords) and the sentence-similarity model are downloaded on first use.

### Verify the installation

```bash
uv run pytest -m "not slow"      # fast tests, no model downloads (~10 s)
uv run pytest                    # everything, including model-backed tests
```

## Reproducing the Paper's Experiments

| # | Model (Hugging Face checkpoint) | Dataset | Instances loaded | Accuracy | Runner |
|---|---------------------------------|---------|------------------|----------|--------|
| 1 | DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) | SST-2 validation | 872 | 91.1% | `experiment_1_distilbert_sst2.py` |
| 2 | FLAN-T5-base (`tanganke/flan-t5-base_glue-sst2`), conditional generation | SST-2 validation | 872 | 93.6% | `experiment_2_t5_sst2.py` |
| 3 | RoBERTa-base (`textattack/roberta-base-ag-news`) | AG News test, stratified | 2000 (500/class) | 93.3% | `experiment_3_roberta_agnews.py` |
| 4 | DistilBERT (`textattack/distilbert-base-uncased-ag-news`) | AG News test, stratified | 2000 (500/class) | 93.3% | `experiment_4_distilbert_agnews.py` |

Accuracy is measured on the loaded instances. Only correctly classified instances are assessed (see [protocol](#assessment-protocol)), so 794, 816, 1867 and 1867 instances respectively; each card states this count.

Run an experiment from the repository root:

```bash
uv run python experiment_runners/experiment_1_distilbert_sst2.py
```

Options:

- `--max-samples N` assesses at most N (correctly classified) instances, useful as a smoke test.
- `--device cuda|cpu` selects the device (default: CUDA if available, otherwise CPU).
- `--results-dir PATH` sets the output directory (default: `results/` in the repository).

Each run writes to `results/`:

- `<experiment>_<timestamp>.json`: full results, configuration, dataset filter summary and per-instance explanations
- `<experiment>_stability_card.md`: the stability card
- `<experiment>_..._intermediate.json`: results saved after each explainer × stress-test combination
- `<experiment>.log`

Full runs take hours on a GPU. LIME (1000 samples per explanation) dominates the cost, especially for T5.

## Assessment Protocol

The runners use the settings reported in the paper (Section IV.A), defined in [`experiment_runners/common.py`](experiment_runners/common.py):

1. **Instances.** Load the dataset and keep only the instances the model classifies correctly (Section III.D).
2. **Stress tests.** For each instance, generate perturbed variants:
   - **Preprocessing variation**: case (upper, lower, title, random), punctuation (removal, replacement, addition) and spacing changes; 7 variants per instance.
   - **Semantic paraphrasing**: synonym substitution (content words only, same part of speech), clause reordering, and simple lexical transformations. Variants are kept only if their cosine similarity to the original is ≥ 0.8 (`all-MiniLM-L6-v2`). Up to 4 variants per instance (3.7 on average for SST-2).
   - **Explainer seed variation**: seeds 42–46, applied to LIME only. IG and SHAP Partition are deterministic.
3. **Explanations.** Explain the original and every variant with identical settings:
   - **Integrated Gradients**: Captum, 50 steps, zero-vector baseline in embedding space, target = predicted class.
   - **LIME**: 1000 samples, default kernel width.
   - **SHAP**: Partition explainer with a text masker.
4. **Comparison.** Map token attributions onto the whitespace-delimited words of the input (subword scores are summed). Compare each variant's explanation with the original's. Words are matched case-insensitively and per occurrence, so repeated words are compared separately.
5. **Aggregation.** Report the mean, standard deviation and 95% t-distribution confidence interval of each metric, per explainer and stress test, plus per predicted class for multi-class tasks.

## Metrics

| Metric | Range | Definition |
|--------|-------|------------|
| **J@5** | 0 – 1 | Jaccard overlap of the 5 words with the largest absolute attribution |
| **ρ** | −1 – 1 | Spearman rank correlation of attributions over the words common to both explanations |
| **AD (%)** | 0 – 100 | Σ\|a₁ᵢ − a₂ᵢ\| / (n · max(‖a₁‖∞, ‖a₂‖∞)) × 100 over common words |
| **Flip-rate** | 0 – 100% | Share of comparisons where the prediction is stable (same class, confidence change ≤ 0.1) but the explanation is not (J@5 < 0.5 or ρ < 0.7) |

Higher J@5 and ρ, and lower AD and flip-rate, indicate more stable explanations. The flip-rate thresholds, and the interpretation bands printed on stability cards, are heuristic conventions, not empirically calibrated cut-offs.

## Stability Cards

The summary table follows the layout of the paper's tables: one row per explainer and stress test, values as mean ± standard deviation over all original-vs-variant comparisons. Card format (the values below are from a 2-instance smoke run and carry no statistical meaning):

```markdown
**Instances**: 2 evaluated: 794 of 872 loaded instances were correctly classified (accuracy 0.911); only correctly classified instances are assessed

| Explainer | Stress Test | J@5 | ρ | AD (%) | Flip-rate | Comparisons |
|-----------|-------------|---|---|---|-----------|-------------|
| IntegratedGradients | Preprocessing | 0.821 ± 0.289 | 0.889 ± 0.172 | 10.246 ± 11.333 | 14.3% | 14 |
| LIME | Seed Variation | 0.804 ± 0.210 | 0.825 ± 0.075 | 10.646 ± 7.287 | 12.5% | 8 |
| SHAP | Paraphrasing | 0.607 ± 0.103 | 0.972 ± 0.037 | 7.314 ± 6.057 | 25.0% | 8 |
```

Cards also include per-class results (multi-class tasks), detailed metrics with confidence intervals, and the experiment configuration.

## Known Limitations

- **Paraphrases are rule-based.** WordNet synonyms are not always natural in context ("almost" → "most"), and the similarity filter is only a proxy for meaning preservation. Paraphrase results also include any change in the model's prediction; prediction-invariance filtering is not applied (as stated in the paper).
- **LIME and SHAP are case-sensitive internally.** LIME treats "Not" and "not" as different features, and SHAP's Partition explainer uses a case-sensitive list of connectors ("and", "or", "but") to build its token hierarchy. Case variants can therefore change their explanations even for an uncased model. This is genuine explainer behaviour, and the preprocessing stress test is meant to capture it.
- **IG integration error.** With the zero baseline, 50 steps leaves a completeness gap of a few percent of F(x) − F(baseline) for some inputs. Every IG explanation records its `convergence_delta`.
- **T5 instrumentation.** T5 is explained through the encoder input embeddings of the sentence tokens, with the class score taken from the decoder's first-step logit for each label word. The task prompt is held fixed and receives no attribution.

## Customize Experiments

### Run your own configuration

```python
from stability_framework.stability_assessor import StabilityAssessor

assessor = StabilityAssessor(results_dir="results", random_seed=42, max_samples=100)
results = assessor.run_experiment(
    model_name="distilbert-base-uncased-finetuned-sst-2-english",   # any HF sequence classifier
    dataset_config={"config_key": "sst2", "sample_size": None, "stratify": False},
    explainer_names=["integrated_gradients", "lime", "shap"],
    stress_test_names=["seed_variation", "preprocessing_variation", "semantic_paraphrasing"],
    experiment_name="my_experiment",
    explainer_params={"lime": {"num_samples": 1000}, "integrated_gradients": {"n_steps": 50}},
)
assessor.generate_stability_card(results, "results/my_experiment_stability_card.md")
```

Set `correct_predictions_only=False` on `StabilityAssessor` to assess all instances.

### Register a custom dataset

```python
from stability_framework.dataset_loader import DATASET_CONFIGS, DatasetConfig

DATASET_CONFIGS["my_dataset"] = DatasetConfig(
    dataset_name="your_hf_dataset",
    split="test",
    text_column="text",
    label_column="label",
    display_name="My Dataset",
)
```

The dataset's label ids must match the model's output indices.

## Project Structure

```text
stability_framework/
├── explainers/                  # Integrated Gradients, LIME, SHAP and shared word alignment
├── metrics/                     # Jaccard@k, Spearman, attribution difference, flip-rate, aggregation
├── dataset_loader.py            # Dataset registry and seeded (stratified) sampling
├── model_loader.py              # Model loading, incl. the T5 label-scoring classifier
├── stability_assessor.py        # Experiment orchestration and assessment protocol
├── stability_card_generator.py  # Markdown stability cards
└── stress_tests.py              # Seed, preprocessing and paraphrasing stress tests

experiment_runners/
├── common.py                    # Paper configuration shared by all runners
└── experiment_{1..4}_*.py       # The paper's four experiments

tests/
├── test_simple.py               # Fast logic tests
├── test_paper_fixes.py          # Regression tests for the protocol fixes (some marked slow)
└── test_integration.py          # Model- and dataset-backed tests
```

## Development

```bash
uv run pytest -m "not slow"   # fast tests
uv run pytest                 # full suite (downloads models on first run)
uv run ruff check .           # lint
```

Contributions are welcome via issues and pull requests. Please include tests for behaviour changes and note in [CHANGELOG.md](CHANGELOG.md) any change that affects reported numbers.

## Citation

If you use this framework, please cite the paper (see also [CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{mohammadalizadeh2025stability,
  title     = {A Framework for Systematic Stability Assessment of Post-hoc Explanations in Text Classification},
  author    = {Mohammadalizadeh, Parman and Mohammadalizadeh, Parham and Mahmoudian, Ayda},
  booktitle = {2025 16th International Conference on Information and Knowledge Technology (IKT)},
  year      = {2025},
  publisher = {IEEE}
}
```

## License

MIT. See [LICENSE](LICENSE).
