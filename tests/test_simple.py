"""
Simple Fast Unit Tests for Stability Framework

NO model downloads, NO heavy computation - just pure logic tests.
Perfect for Test-Driven Refactoring!
"""

import pytest


class TestMetrics:
    """Test metric calculations (pure math, super fast!)"""
    
    def test_jaccard_similarity(self):
        """Test Jaccard@K metric"""
        from stability_framework.metrics import StabilityMetrics
        
        metrics = StabilityMetrics()
        
        # Test with overlapping attributions
        original = {"word1": 0.9, "word2": 0.8, "word3": 0.7, "word4": 0.6, "word5": 0.5}
        perturbed = {"word1": 0.85, "word2": 0.75, "word3": 0.65, "word4": 0.55, "word6": 0.5}
        
        jaccard = metrics.jaccard_similarity_at_k(original, perturbed, k=5)
        
        assert 0.0 <= jaccard <= 1.0
        assert jaccard > 0.5  # Should have 4/6 overlap = 0.67
        print(f"✓ Jaccard@5: {jaccard:.3f}")
    
    def test_jaccard_identical(self):
        """Test Jaccard with identical attributions"""
        from stability_framework.metrics import StabilityMetrics
        
        metrics = StabilityMetrics()
        same = {"a": 1.0, "b": 0.9, "c": 0.8}
        
        jaccard = metrics.jaccard_similarity_at_k(same, same, k=3)
        assert jaccard == 1.0  # Perfect match
        print("✓ Jaccard identical: 1.0")
    
    def test_jaccard_no_overlap(self):
        """Test Jaccard with no overlap"""
        from stability_framework.metrics import StabilityMetrics
        
        metrics = StabilityMetrics()
        attr1 = {"a": 1.0, "b": 0.9}
        attr2 = {"c": 1.0, "d": 0.9}
        
        jaccard = metrics.jaccard_similarity_at_k(attr1, attr2, k=2)
        assert jaccard == 0.0  # No overlap
        print("✓ Jaccard no overlap: 0.0")
    
    def test_spearman_correlation(self):
        """Test Spearman correlation"""
        from stability_framework.metrics import StabilityMetrics
        
        metrics = StabilityMetrics()
        
        # Same ranking, different values -> perfect correlation
        original = {"word1": 0.9, "word2": 0.8, "word3": 0.7}
        perturbed = {"word1": 0.85, "word2": 0.75, "word3": 0.65}
        
        spearman = metrics.spearman_rank_correlation(original, perturbed)
        
        assert -1.0 <= spearman <= 1.0
        assert spearman > 0.9  # Should be close to 1.0
        print(f"✓ Spearman correlation: {spearman:.3f}")


class TestStressTests:
    """Test stress test logic (no models needed!)"""
    
    def test_seed_variation_creates_variants(self):
        """Test seed variation creates correct number of variants"""
        from stability_framework.stress_tests import create_stress_test
        
        test = create_stress_test("seed_variation", seeds=[1, 2, 3])
        variants = test.apply("Test sentence.")
        
        assert len(variants) == 3
        assert all(v == "Test sentence." for v in variants)
        print("✓ Seed variation creates 3 identical variants")
    
    def test_preprocessing_variation_creates_variants(self):
        """Test preprocessing variation creates variants"""
        from stability_framework.stress_tests import create_stress_test
        
        test = create_stress_test("preprocessing_variation")
        variants = test.apply("Hello, World!")
        
        assert len(variants) > 0
        assert isinstance(variants, list)
        print(f"✓ Preprocessing variation creates {len(variants)} variants")


class TestModelConfig:
    """Test model configuration structure"""
    
    def test_experiment_models_exist(self):
        """Test that EXPERIMENT_MODELS is defined"""
        from stability_framework.model_loader import EXPERIMENT_MODELS
        
        assert isinstance(EXPERIMENT_MODELS, dict)
        assert len(EXPERIMENT_MODELS) > 0
        print(f"✓ Found {len(EXPERIMENT_MODELS)} registered models")
    
    def test_model_config_structure(self):
        """Test model config has required fields"""
        from stability_framework.model_loader import EXPERIMENT_MODELS
        
        for key, config in EXPERIMENT_MODELS.items():
            assert "name" in config
            assert "architecture" in config
            assert isinstance(config["name"], str)
            
        print("✓ All model configs have required fields")


class TestDatasetConfig:
    """Test dataset configuration structure"""
    
    def test_dataset_configs_exist(self):
        """Test that DATASET_CONFIGS is defined"""
        from stability_framework.dataset_loader import DATASET_CONFIGS
        
        assert isinstance(DATASET_CONFIGS, dict)
        assert len(DATASET_CONFIGS) > 0
        print(f"✓ Found {len(DATASET_CONFIGS)} registered datasets")
    
    def test_dataset_config_structure(self):
        """Test dataset config has required fields"""
        from stability_framework.dataset_loader import DATASET_CONFIGS, DatasetConfig
        
        for key, config in DATASET_CONFIGS.items():
            assert isinstance(config, DatasetConfig)
            assert config.dataset_name
            assert config.text_column
            assert config.label_column
            
        print("✓ All dataset configs have required fields")


def run_quick_tests():
    """
    Quick smoke tests without pytest
    Run: python tests/test_simple.py
    """
    print("\n" + "="*60)
    print("RUNNING QUICK UNIT TESTS (No Model Downloads!)")
    print("="*60 + "\n")
    
    # Test 1: Metrics
    print("Test 1: Jaccard Similarity...")
    from stability_framework.metrics import StabilityMetrics
    metrics = StabilityMetrics()
    attr1 = {"a": 1.0, "b": 0.9, "c": 0.8}
    jaccard = metrics.jaccard_similarity_at_k(attr1, attr1, k=3)
    assert jaccard == 1.0
    print(f"  ✓ Jaccard works: {jaccard}\n")
    
    # Test 2: Spearman
    print("Test 2: Spearman Correlation...")
    spearman = metrics.spearman_rank_correlation(attr1, attr1)
    assert spearman == 1.0
    print(f"  ✓ Spearman works: {spearman}\n")
    
    # Test 3: Stress test config
    print("Test 3: Stress Test Creation...")
    from stability_framework.stress_tests import create_stress_test
    test = create_stress_test("seed_variation", seeds=[1, 2])
    variants = test.apply("Test")
    assert len(variants) == 2
    print(f"  ✓ Stress test works: {len(variants)} variants\n")
    
    # Test 4: Model registry
    print("Test 4: Model Registry...")
    from stability_framework.model_loader import EXPERIMENT_MODELS
    assert len(EXPERIMENT_MODELS) > 0
    print(f"  ✓ Registry works: {len(EXPERIMENT_MODELS)} models\n")
    
    # Test 5: Dataset registry
    print("Test 5: Dataset Registry...")
    from stability_framework.dataset_loader import DATASET_CONFIGS
    assert len(DATASET_CONFIGS) > 0
    print(f"  ✓ Registry works: {len(DATASET_CONFIGS)} datasets\n")
    
    print("="*60)
    print("ALL QUICK TESTS PASSED! ✓")
    print("="*60)
    print("\nThese tests run in < 1 second!")
    print("Perfect for Test-Driven Refactoring.\n")
    print("Run full test suite: pytest tests/test_simple.py")


if __name__ == "__main__":
    run_quick_tests()
