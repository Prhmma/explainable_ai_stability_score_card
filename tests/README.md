# Test Suite

| File | What it covers | Needs models? |
|------|----------------|---------------|
| `test_simple.py` | Metric maths, stress-test logic, model and dataset registries | No |
| `test_paper_fixes.py` | Regression tests for the fixes in [CHANGELOG.md](../CHANGELOG.md): word alignment, paraphrase generation, configuration precedence, stability-card format. Tests marked `slow` check Integrated Gradients, SHAP alignment, the correct-prediction filter and the T5 classifier against real models | Only the `slow` tests |
| `test_integration.py` | Model loading, dataset loading, explainers and a minimal end-to-end experiment | Yes |

## Running

```bash
# Fast tests only (no model downloads, ~10 s)
uv run pytest -m "not slow"

# Everything (downloads models and datasets from the Hugging Face Hub on first run)
uv run pytest
```

Model-backed tests run on CPU. After the first run, models and datasets are read from the local Hugging Face cache.
