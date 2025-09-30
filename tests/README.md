# Test Suite for Stability Framework

This directory contains **simple, fast unit tests** for the stability framework.

## Quick Start

### Run Quick Tests (< 1 second!)
```bash
# No pytest needed - just pure Python
uv run python tests/test_simple.py
```

This will test:
- ✓ Metric calculations (Jaccard, Spearman)
- ✓ Stress test logic
- ✓ Model registry structure
- ✓ Dataset registry structure

### Run Full Test Suite (~8 seconds)
```bash
# Run all tests with pytest
uv run pytest tests/test_simple.py -v

# Run with less output
uv run pytest tests/test_simple.py
```

## Test Philosophy

**Keep it simple!** These tests:
- ❌ NO model downloads
- ❌ NO heavy computation
- ❌ NO network calls
- ✅ Pure logic and math
- ✅ Fast feedback (<10 seconds)
- ✅ Perfect for refactoring

## Test Structure

```
tests/
├── test_simple.py          # Main test suite (10 tests)
├── conftest.py             # pytest configuration
├── README.md              # This file
└── test_results/          # Generated during tests (gitignored)
```

## Tests

### TestMetrics (4 tests)
- `test_jaccard_similarity` - Jaccard@K with overlapping sets
- `test_jaccard_identical` - Jaccard with identical attributions (=1.0)
- `test_jaccard_no_overlap` - Jaccard with no overlap (=0.0)
- `test_spearman_correlation` - Spearman rank correlation

### TestStressTests (2 tests)
- `test_seed_variation_creates_variants` - Seed variation creates N variants
- `test_preprocessing_variation_creates_variants` - Preprocessing variations

### TestModelConfig (2 tests)
- `test_experiment_models_exist` - Model registry exists
- `test_model_config_structure` - All configs have required fields

### TestDatasetConfig (2 tests)
- `test_experiment_datasets_exist` - Dataset registry exists
- `test_dataset_config_structure` - All configs have required fields

## Using Tests During Refactoring

**BEFORE refactoring:**
```bash
uv run python tests/test_simple.py
```
Should print "ALL QUICK TESTS PASSED! ✓"

**DURING refactoring:**
Run frequently to catch breaks:
```bash
uv run pytest tests/test_simple.py
```

**AFTER refactoring:**
All tests should still pass:
```bash
uv run pytest tests/test_simple.py -v
```

## Adding New Tests

Keep them simple and fast:

```python
def test_my_new_feature(self):
    """Test description"""
    from stability_framework.my_module import my_function
    
    result = my_function(simple_input)
    assert result == expected_output
```

**Rules:**
1. No model loading
2. No dataset downloading  
3. Test pure logic only
4. Should run in milliseconds

## CI/CD Usage

```bash
# Stop on first failure
uv run pytest tests/test_simple.py --maxfail=1

# Generate coverage report
uv run pytest tests/test_simple.py --cov=stability_framework
```


## Tests

### TestBasicPipeline
- `test_distilbert_sst2_lime_seed_variation` - Complete pipeline with minimal data (5 samples)

### TestModelLoading
- `test_load_distilbert` - Load DistilBERT model
- `test_load_roberta` - Load RoBERTa model

### TestDatasetLoading
- `test_load_sst2` - Load SST-2 dataset
- `test_load_agnews` - Load AG News dataset

### TestExplainers
- `test_lime_explainer` - LIME functionality
- `test_ig_explainer` - Integrated Gradients functionality

### TestStressTests
- `test_seed_variation` - Seed variation stress test
- `test_preprocessing_variation` - Preprocessing variation stress test

### TestMetrics
- `test_jaccard_similarity` - Jaccard@K metric
- `test_spearman_correlation` - Spearman correlation metric

### TestEndToEnd (marked as slow)
- `test_full_experiment_minimal` - Full experiment with multiple configs

## Using Tests During Refactoring

1. **Before refactoring:**
   ```bash
   python tests/test_integration.py
   ```
   Should print "ALL SMOKE TESTS PASSED! ✓"

2. **During refactoring:**
   Run tests frequently:
   ```bash
   pytest tests/test_integration.py -v
   ```

3. **After refactoring:**
   Run full test suite:
   ```bash
   pytest tests/
   ```

## Test Configuration

Tests use minimal resources:
- **CPU only** (no GPU required)
- **Small sample sizes** (2-5 samples)
- **Reduced explainer parameters** (fewer samples for LIME)
- **Fewer seeds** (2-3 instead of 5)

This makes tests fast (~2-10 minutes) while still validating functionality.

## Adding New Tests

When adding new models or datasets:

```python
def test_my_new_model(self):
    """Test my new model"""
    from model_loader import ModelLoader
    
    loader = ModelLoader(device="cpu")
    model, tokenizer = loader.load_model_and_tokenizer("my/model")
    
    assert model is not None
    assert tokenizer is not None
```

## Continuous Integration

For CI/CD, use:
```bash
pytest tests/ --tb=short --maxfail=1
```

This stops on first failure and shows minimal traceback.
