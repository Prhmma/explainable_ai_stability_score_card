"""
Integration tests for the Stability Framework.

These load real models and datasets from the Hugging Face Hub (cached after the
first run). All tests in this module are marked `slow`.
"""

import pytest

pytestmark = pytest.mark.slow


class TestModelLoading:
    """Test that all registered models can load"""
    
    def test_load_distilbert(self):
        """Test loading DistilBERT model"""
        from stability_framework.model_loader import ModelLoader
        
        loader = ModelLoader(device="cpu")
        model, tokenizer = loader.load_model_and_tokenizer(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        assert model is not None
        assert tokenizer is not None
        assert hasattr(model, 'eval')
        
        # Test prediction
        inputs = tokenizer("This is a test", return_tensors="pt")
        outputs = model(**inputs)
        assert hasattr(outputs, 'logits')
        assert outputs.logits.shape[-1] == 2  # Binary classification
        
        print("✓ DistilBERT loads successfully")
    
    def test_load_roberta(self):
        """Test loading RoBERTa model"""
        from stability_framework.model_loader import ModelLoader
        
        loader = ModelLoader(device="cpu")
        model, tokenizer = loader.load_model_and_tokenizer(
            "textattack/roberta-base-ag-news"
        )
        
        assert model is not None
        assert tokenizer is not None
        
        # Test prediction
        inputs = tokenizer("This is a test", return_tensors="pt")
        outputs = model(**inputs)
        assert hasattr(outputs, 'logits')
        assert outputs.logits.shape[-1] == 4  # 4 classes for AG News
        
        print("✓ RoBERTa loads successfully")


class TestDatasetLoading:
    """Test that all datasets can load"""
    
    def test_load_sst2(self):
        """Test SST-2 loading"""
        from stability_framework.dataset_loader import DatasetLoader, DATASET_CONFIGS
        
        loader = DatasetLoader(random_seed=42)
        texts, labels, metadata = loader.load_dataset(DATASET_CONFIGS["sst2"], sample_size=10)
        
        assert len(texts) == 10
        assert len(labels) == 10
        assert metadata["num_classes"] == 2
        assert all(isinstance(text, str) for text in texts)
        assert all(label in [0, 1] for label in labels)
        
        print(f"✓ SST-2 loads successfully: {len(texts)} samples")
    
    def test_load_agnews(self):
        """Test AG News loading"""
        from stability_framework.dataset_loader import DatasetLoader, DATASET_CONFIGS
        
        loader = DatasetLoader(random_seed=42)
        texts, labels, metadata = loader.load_dataset(DATASET_CONFIGS["ag_news"], sample_size=20, stratify=True)
        
        assert len(texts) == 20
        assert len(labels) == 20
        assert sorted(metadata["class_distribution"].values()) == [5, 5, 5, 5]
        assert metadata["num_classes"] == 4
        assert all(isinstance(text, str) for text in texts)
        assert all(label in [0, 1, 2, 3] for label in labels)
        
        print(f"✓ AG News loads successfully: {len(texts)} samples")


class TestExplainers:
    """Test that explainers work"""
    
    @pytest.fixture
    def model_and_tokenizer(self):
        """Load a simple model for testing"""
        from stability_framework.model_loader import ModelLoader
        loader = ModelLoader(device="cpu")
        return loader.load_model_and_tokenizer(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
    
    def test_lime_explainer(self, model_and_tokenizer):
        """Test LIME explainer"""
        from stability_framework.explainers import create_explainer
        
        model, tokenizer = model_and_tokenizer
        explainer = create_explainer("lime", random_seed=42, num_samples=100)
        
        result = explainer.explain(
            model, tokenizer,
            text="This movie is great!",
            target_class=1
        )
        
        assert result["method"] == "LIME"
        assert "attributions" in result
        assert len(result["attributions"]) > 0
        assert "confidence" in result
        
        print(f"✓ LIME explainer works")
        print(f"  Top words: {list(result['attributions'].keys())[:3]}")
    
    def test_ig_explainer(self, model_and_tokenizer):
        """Test Integrated Gradients explainer"""
        from stability_framework.explainers import create_explainer
        
        model, tokenizer = model_and_tokenizer
        explainer = create_explainer("integrated_gradients", random_seed=42, n_steps=10)
        
        result = explainer.explain(
            model, tokenizer,
            text="This movie is great!",
            target_class=1
        )
        
        assert result["method"] == "IntegratedGradients"
        assert "attributions" in result
        assert len(result["attributions"]) > 0
        
        print(f"✓ Integrated Gradients explainer works")


class TestStressTests:
    """Test that stress tests work"""
    
    def test_seed_variation(self):
        """Test seed variation stress test"""
        from stability_framework.stress_tests import create_stress_test
        
        test = create_stress_test("seed_variation", seeds=[42, 43, 44])
        variants = test.apply("This is a test sentence.")
        
        assert len(variants) == 3
        assert all(v == "This is a test sentence." for v in variants)
        assert test.get_seeds() == [42, 43, 44]
        
        print("✓ Seed variation stress test works")
    
    def test_preprocessing_variation(self):
        """Test preprocessing variation stress test"""
        from stability_framework.stress_tests import create_stress_test
        
        test = create_stress_test("preprocessing_variation")
        variants = test.apply("Hello, World!")
        
        assert len(variants) > 0
        assert any("hello" in v.lower() for v in variants)
        
        print(f"✓ Preprocessing variation works: {len(variants)} variants")


class TestMetrics:
    """Test metric calculations"""
    
    def test_jaccard_similarity(self):
        """Test Jaccard@K metric"""
        from stability_framework.metrics import StabilityMetrics
        
        metrics = StabilityMetrics()
        
        original = {"word1": 0.9, "word2": 0.8, "word3": 0.7, "word4": 0.6, "word5": 0.5}
        perturbed = {"word1": 0.85, "word2": 0.75, "word3": 0.65, "word4": 0.55, "word6": 0.5}
        
        jaccard = metrics.jaccard_similarity_at_k(original, perturbed, k=5)
        
        assert 0.0 <= jaccard <= 1.0
        assert jaccard > 0.5  # Should have reasonable overlap
        
        print(f"✓ Jaccard@5 calculation works: {jaccard:.3f}")
    
    def test_spearman_correlation(self):
        """Test Spearman correlation"""
        from stability_framework.metrics import StabilityMetrics
        
        metrics = StabilityMetrics()
        
        original = {"word1": 0.9, "word2": 0.8, "word3": 0.7}
        perturbed = {"word1": 0.85, "word2": 0.75, "word3": 0.65}
        
        spearman = metrics.spearman_rank_correlation(original, perturbed)
        
        assert -1.0 <= spearman <= 1.0
        
        print(f"✓ Spearman correlation works: {spearman:.3f}")


class TestEndToEnd:
    """End-to-end tests with multiple configurations"""
    
    @pytest.mark.slow
    def test_full_experiment_minimal(self):
        """
        Full experiment with minimal data
        Tests: 2 models × 1 dataset × 2 explainers × 1 stress test
        """
        from stability_framework.stability_assessor import StabilityAssessor
        
        assessor = StabilityAssessor(
            results_dir="tests/test_results",
            random_seed=42,
            device="cpu",
            max_samples=3
        )
        
        # Test 1: DistilBERT + SST-2
        results1 = assessor.run_experiment(
            model_name="distilbert-base-uncased-finetuned-sst-2-english",
            dataset_config={'config_key': 'sst2', 'sample_size': None, 'stratify': False},
            explainer_names=["lime", "shap"],
            stress_test_names=["seed_variation"],
            experiment_name="test_distilbert_sst2",
            explainer_params={"lime": {"num_samples": 50}},
            stress_test_params={"seed_variation": {"seeds": [42, 43]}}
        )
        
        assert results1 is not None
        by_explainer = results1["results"]["by_explainer"]
        assert by_explainer["LIME"]["SeedVariation"]["sample_statistics"]["failed_samples"] == 0
        # SHAP (Partition) is deterministic, so seed variation is skipped for it
        assert "SeedVariation" not in by_explainer["SHAP"]
        
        print("✓ Full experiment test passed!")
        print(f"  Explainers tested: {list(results1['results']['by_explainer'].keys())}")


def run_smoke_tests():
    """
    Quick smoke tests that can be run without pytest
    Run with: python tests/test_integration.py
    """
    print("\n" + "="*60)
    print("RUNNING SMOKE TESTS")
    print("="*60 + "\n")
    
    # Test 1: Model loading
    print("Test 1: Model Loading...")
    from stability_framework.model_loader import ModelLoader
    loader = ModelLoader(device="cpu")
    model, tokenizer = loader.load_model_and_tokenizer(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("✓ Model loads\n")
    
    # Test 2: Dataset loading
    print("Test 2: Dataset Loading...")
    from stability_framework.dataset_loader import DatasetLoader, DATASET_CONFIGS
    ds_loader = DatasetLoader(random_seed=42)
    texts, labels, metadata = ds_loader.load_dataset(DATASET_CONFIGS["sst2"], sample_size=5)
    print(f"✓ Dataset loads: {len(texts)} samples\n")
    
    # Test 3: Explainer
    print("Test 3: Explainer...")
    from stability_framework.explainers import create_explainer
    explainer = create_explainer("lime", random_seed=42, num_samples=50)
    result = explainer.explain(model, tokenizer, texts[0], target_class=1)
    print(f"✓ Explainer works: {len(result['attributions'])} attributions\n")
    
    # Test 4: Full pipeline
    print("Test 4: Full Pipeline (minimal)...")
    from stability_framework.stability_assessor import StabilityAssessor
    assessor = StabilityAssessor(
        results_dir="tests/smoke_test_results",
        random_seed=42,
        device="cpu",
        max_samples=2
    )
    results = assessor.run_experiment(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        dataset_config={'config_key': 'sst2', 'sample_size': None, 'stratify': False},
        explainer_names=["lime"],
        stress_test_names=["seed_variation"],
        experiment_name="smoke_test",
        explainer_params={"lime": {"num_samples": 50}},
        stress_test_params={"seed_variation": {"seeds": [42, 43]}}
    )
    print("✓ Full pipeline works\n")
    
    print("="*60)
    print("ALL SMOKE TESTS PASSED! ✓")
    print("="*60)
    print("\nYou can now refactor safely!")
    print("Run: pytest tests/test_integration.py for full test suite")


if __name__ == "__main__":
    # Run smoke tests when executed directly
    run_smoke_tests()
