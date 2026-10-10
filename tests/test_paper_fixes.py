"""
Regression tests for the fixes that align the implementation with the paper.

The unmarked tests are fast and need no model downloads. Tests marked `slow`
load real models from the Hugging Face Hub (cached after the first run);
skip them with: pytest -m "not slow"
"""

import pytest

from stability_framework.explainers.base import BaseExplainer


# ---------------------------------------------------------------------------
# Word-level attribution alignment
# ---------------------------------------------------------------------------

class TestWordAlignment:

    def test_subword_scores_sum_per_whitespace_word(self):
        words = BaseExplainer._aggregate_to_words(
            "un believ able film", [(0, 2), (3, 9), (10, 14), (15, 19)], [1, 2, 3, 4]
        )
        assert words == [("un", 1.0), ("believ", 2.0), ("able", 3.0), ("film", 4.0)]
        merged = BaseExplainer._aggregate_to_words("unbelievable film", [(0, 2), (2, 8), (8, 12), (13, 17)], [1, 2, 3, 4])
        assert merged == [("unbelievable", 6.0), ("film", 4.0)]

    def test_word_start_marker_joins_following_word(self):
        # SentencePiece/BPE tokens that start on the preceding space
        assert BaseExplainer._aggregate_to_words("a film", [(0, 1), (1, 2), (2, 6)], [1, 2, 3]) == [("a", 1.0), ("film", 5.0)]

    def test_trailing_whitespace_token_kept_with_last_word(self):
        words = BaseExplainer._aggregate_to_words("good film ", [(0, 4), (5, 9), (10, 10)], [1, 2, 3])
        assert words == [("good", 1.0), ("film", 5.0)]

    def test_special_tokens_skipped(self):
        words = BaseExplainer._aggregate_to_words("good", [(0, 0), (0, 4), (0, 0)], [9, 1, 9], [True, False, True])
        assert words == [("good", 1.0)]

    def test_repeated_words_are_not_overwritten(self):
        keyed = BaseExplainer._key_attributions([("not", 1), ("good", 2), ("not", 3), ("good", 4)])
        assert keyed == {"not": 1.0, "good": 2.0, "not#2": 3.0, "good#2": 4.0}

    def test_keys_ignore_case_and_edge_punctuation(self):
        original = BaseExplainer._key_attributions([("The", 1), ("Film,", 2)])
        upper = BaseExplainer._key_attributions([("THE", 1), ("FILM", 2)])
        assert original.keys() == upper.keys() == {"the", "film"}
        assert BaseExplainer._key_attributions([(",", 1)]) == {",": 1.0}


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestParaphrasing:

    @pytest.fixture(scope="class")
    def paraphraser(self, tmp_path_factory):
        from stability_framework.stress_tests import SemanticParaphrasingTest
        test = SemanticParaphrasingTest(cache_dir=str(tmp_path_factory.mktemp("paraphrases")))
        test.use_cache = False
        return test

    def test_original_first_then_up_to_four_variants(self, paraphraser):
        out = paraphraser.apply("the acting is strong , and the story is a delight . ")
        assert out[0] == "the acting is strong , and the story is a delight . "
        assert 1 <= len(out) - 1 <= 4
        assert len(set(out)) == len(out)

    def test_stopwords_are_never_substituted(self, paraphraser):
        text = "a film that is not as good as it thinks it is"
        stop = paraphraser._stopwords()
        for variant in paraphraser._generate_synonym_substitutions(text):
            changed = [a for a, b in zip(text.split(), variant.split()) if a != b]
            assert len(changed) == 1 and changed[0].lower() not in stop

    def test_synonyms_are_deterministic_and_lowercase(self, paraphraser):
        from nltk.corpus import wordnet
        first = paraphraser._get_synonyms("film", wordnet.NOUN)
        assert first == paraphraser._get_synonyms("film", wordnet.NOUN)
        assert all(s.islower() for s in first)

    def test_clause_reordering_swaps_clauses_and_keeps_punctuation(self, paraphraser):
        assert paraphraser._generate_reorderings("the cast is great , the script is weak . ") == [
            "the script is weak , the cast is great . "
        ]
        # Contrastive clauses are not moved
        assert paraphraser._generate_reorderings("the cast is great , but the script is weak .") == []

    def test_cache_key_includes_generator_version(self, paraphraser):
        from stability_framework import stress_tests
        key = paraphraser._get_cache_key("text")
        original = stress_tests.PARAPHRASE_GENERATOR_VERSION
        try:
            stress_tests.PARAPHRASE_GENERATOR_VERSION = original + 1
            assert paraphraser._get_cache_key("text") != key
        finally:
            stress_tests.PARAPHRASE_GENERATOR_VERSION = original


class TestAssessorConfiguration:

    def test_runner_stress_test_params_override_defaults(self, tmp_path):
        from stability_framework.stability_assessor import StabilityAssessor
        assessor = StabilityAssessor(results_dir=str(tmp_path), device="cpu")
        (test,) = assessor._initialize_stress_tests(["seed_variation"], {"seed_variation": {"seeds": [1, 2, 3]}})
        assert test.get_seeds() == [1, 2, 3]

    def test_explainer_params_survive_seed_change(self):
        from stability_framework.explainers import create_explainer
        lime = create_explainer("lime", random_seed=42, num_samples=500)
        variant = create_explainer("lime", random_seed=43, **lime.get_params())
        assert variant.params["num_samples"] == 500
        ig = create_explainer("integrated_gradients", random_seed=42, n_steps=25)
        assert create_explainer("ig", random_seed=43, **ig.get_params()).n_steps == 25

    def test_experiment_models_match_paper(self):
        from stability_framework.model_loader import EXPERIMENT_MODELS
        assert "t5" in EXPERIMENT_MODELS["experiment_2"]["name"]
        assert EXPERIMENT_MODELS["experiment_4"]["name"] == "textattack/distilbert-base-uncased-ag-news"


# ---------------------------------------------------------------------------
# Stability card
# ---------------------------------------------------------------------------

def _stats(mean, std=0.1):
    return {"mean": mean, "std": std, "ci_95": (mean - std, mean + std)}


def _synthetic_results(num_labels):
    combo = {
        "jaccard_at_5": _stats(0.5),
        "spearman_correlation": _stats(0.7),
        "attr_diff_relative_difference_percentage": _stats(12.5, 3.0),
        "attr_diff_mean_absolute_difference": _stats(0.01),
        "flip_rate": {"flip_rate_percentage": 25.0},
        "sample_statistics": {"total_samples": 2, "successful_samples": 2, "failed_samples": 0, "total_comparisons": 8},
        "by_predicted_class": {0: {"jaccard_at_5": _stats(0.4), "spearman_correlation": _stats(0.6),
                                   "attr_diff_relative_difference_percentage": _stats(10.0)}},
    }
    return {
        "configuration": {"model_name": "m", "explainer_names": ["lime"], "random_seed": 42},
        "model_info": {"num_labels": num_labels, "num_parameters": 1, "vocab_size": 1},
        "dataset_info": {"dataset_name": "d", "total_samples": 10, "evaluated_samples": 9,
                         "correct_prediction_filter": {"before": 10, "kept": 9, "removed": 1, "accuracy": 0.9}},
        "duration_seconds": 1.0,
        "results": {"by_explainer": {"LIME": {"PreprocessingVariation": combo}}},
    }


class TestStabilityCard:

    def test_summary_row_matches_paper_table_format(self):
        from stability_framework.stability_card_generator import StabilityCardGenerator
        card = StabilityCardGenerator().generate_card(_synthetic_results(2))
        row = next(line for line in card.splitlines() if line.startswith("| LIME | Preprocessing |"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[2:] == ["0.500 ± 0.100", "0.700 ± 0.100", "12.500 ± 3.000", "25.0%", "8"]
        assert "9 of 10 loaded instances were correctly classified" in card

    def test_class_breakdown_only_for_multiclass(self):
        from stability_framework.stability_card_generator import StabilityCardGenerator
        assert "Results by Predicted Class" not in StabilityCardGenerator().generate_card(_synthetic_results(2))
        assert "Results by Predicted Class" in StabilityCardGenerator().generate_card(_synthetic_results(4))


# ---------------------------------------------------------------------------
# Model-backed checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def distilbert():
    from stability_framework.model_loader import ModelLoader
    return ModelLoader(device="cpu").load_model_and_tokenizer("distilbert-base-uncased-finetuned-sst-2-english")


@pytest.mark.slow
class TestIntegratedGradients:

    TEXT = "a gripping , beautifully acted film that never lets up . "

    def test_deterministic_and_eval_mode(self, distilbert):
        from stability_framework.explainers import IntegratedGradientsExplainer
        model, tokenizer = distilbert
        ig = IntegratedGradientsExplainer(n_steps=20)
        assert ig.explain(model, tokenizer, self.TEXT)["attributions"] == ig.explain(model, tokenizer, self.TEXT)["attributions"]
        assert not model.training

    def test_explains_the_prediction_function(self, distilbert):
        import torch
        model, tokenizer = distilbert
        enc = tokenizer(self.TEXT, return_tensors="pt")
        with torch.no_grad():
            direct = model(**enc).logits
            via_embeddings = model(inputs_embeds=model.get_input_embeddings()(enc["input_ids"]),
                                   attention_mask=enc["attention_mask"]).logits
        assert torch.allclose(direct, via_embeddings, atol=1e-5)

    def test_word_scores_conserve_token_scores(self, distilbert):
        from stability_framework.explainers import IntegratedGradientsExplainer
        model, tokenizer = distilbert
        r = IntegratedGradientsExplainer(n_steps=20).explain(model, tokenizer, self.TEXT)
        token_total = sum(s for t, s in r["token_attributions"] if t not in tokenizer.all_special_tokens)
        assert sum(r["attributions"].values()) == pytest.approx(token_total, abs=1e-4)
        assert r["parameters"]["baseline"] == "zero_embedding"


@pytest.mark.slow
class TestShapAlignment:

    def test_trailing_space_and_case_variants_align(self, distilbert):
        from stability_framework.explainers import create_explainer
        model, tokenizer = distilbert
        text = "The Film is a gripping, beautifully acted thriller. "
        a = create_explainer("shap").explain(model, tokenizer, text)
        b = create_explainer("shap").explain(model, tokenizer, text.upper())
        assert [w for w, _ in a["words"]] == text.split()
        assert a["attributions"] == b["attributions"]


@pytest.mark.slow
class TestCorrectPredictionFilter:

    def test_only_correct_instances_kept(self, distilbert, tmp_path):
        from stability_framework.stability_assessor import StabilityAssessor
        model, tokenizer = distilbert
        texts = ["a wonderful , moving film .", "a dull and lifeless mess .", "a wonderful , moving film ."]
        labels = [1, 0, 0]  # the last label is deliberately wrong
        assessor = StabilityAssessor(results_dir=str(tmp_path), device="cpu")
        kept, kept_labels, info = assessor._filter_correct_predictions(model, tokenizer, texts, labels)
        assert kept == texts[:2] and kept_labels == [1, 0]
        assert info == {"before": 3, "kept": 2, "removed": 1, "accuracy": 2 / 3}


@pytest.mark.slow
class TestT5Classifier:

    @pytest.fixture(scope="class")
    def t5(self):
        from stability_framework.model_loader import ModelLoader, EXPERIMENT_MODELS
        return ModelLoader(device="cpu").load_model_and_tokenizer(EXPERIMENT_MODELS["experiment_2"]["name"])

    def test_prompt_reconstruction_matches_joint_tokenization(self, t5):
        from stability_framework.model_loader import T5_PROMPT_TEMPLATES
        _, tokenizer = t5
        template = T5_PROMPT_TEMPLATES["sst2"]["input_text"]
        for sentence in ["it 's a charming and often affecting journey . ", "'Unclassifiably' awful!"]:
            rebuilt = tokenizer.prefix_ids + tokenizer(sentence)["input_ids"][:-1] + tokenizer.suffix_ids + [tokenizer.eos_token_id]
            assert rebuilt == tokenizer.tokenizer(template.format(sentence=sentence))["input_ids"]

    def test_label_scoring_matches_generation(self, t5):
        import torch
        from stability_framework.model_loader import T5_PROMPT_TEMPLATES
        model, tokenizer = t5
        template = T5_PROMPT_TEMPLATES["sst2"]["input_text"]
        for sentence, label in [("a gripping , beautifully acted film . ", 1), ("dull , lifeless and far too long . ", 0)]:
            with torch.no_grad():
                scored = model(**tokenizer(sentence, return_tensors="pt")).logits.argmax(-1).item()
                generated = model.t5_model.generate(**tokenizer.tokenizer(template.format(sentence=sentence), return_tensors="pt"), max_new_tokens=3)
            assert scored == label
            assert tokenizer.tokenizer.decode(generated[0], skip_special_tokens=True).strip() == model.config.id2label[scored]

    def test_no_untrained_parameters(self, t5):
        model, _ = t5
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in model.t5_model.parameters())


class TestUndefinedMetrics:

    def test_spearman_undefined_is_nan_and_excluded(self):
        import math
        from stability_framework.metrics import StabilityMetrics
        m = StabilityMetrics()
        assert math.isnan(m.spearman_rank_correlation({"bad": 1.0, ".": 0.1}, {"bad": 0.5}))
        assert math.isnan(m.spearman_rank_correlation({"a": 1.0, "b": 1.0}, {"a": 0.3, "b": 0.9}))
        agg = m.aggregate_metrics([{"spearman_correlation": float("nan")}, {"spearman_correlation": 0.5}])
        assert agg["spearman_correlation"]["mean"] == 0.5 and agg["spearman_correlation"]["count"] == 1

    def test_attribution_difference_undefined_without_common_words(self):
        import math
        from stability_framework.metrics import StabilityMetrics
        result = StabilityMetrics().attribution_difference({"a": 1.0}, {"b": 1.0})
        assert math.isnan(result["relative_difference_percentage"]) and result["common_tokens"] == 0
