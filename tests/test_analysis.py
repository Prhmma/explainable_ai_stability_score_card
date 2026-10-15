"""Tests for the post-hoc analysis package (no models needed)."""

import numpy as np

from analysis.faithfulness import faithfulness, word_scores_for_text
from analysis.paper_tables import PAPER_RESULTS
from analysis.stability import stability_summary


def test_paper_tables_complete():
    assert len(PAPER_RESULTS) == 28
    assert PAPER_RESULTS[("experiment_1", "IntegratedGradients", "PreprocessingVariation")]["jaccard_at_5"] == (0.471, 0.217)
    assert PAPER_RESULTS[("experiment_4", "SHAP", "SemanticParaphrasing")]["attr_diff_relative_difference_percentage"] == (0.316, 0.390)


def test_word_scores_align_when_explainer_skips_words():
    # LIME has no feature for the punctuation-only word ","
    scores = word_scores_for_text("good , not bad", [["good", 1.0], ["not", -0.5], ["bad", 0.2]])
    assert scores.tolist() == [1.0, 0.0, -0.5, 0.2]


def _comparison(j, same):
    return {"jaccard_at_5": j, "spearman_correlation": j, "attr_diff_relative_difference_percentage": 1 - j,
            "comparison_info": {"same_prediction": same}}


def test_prediction_invariant_subset():
    records = [{"metrics": [_comparison(1.0, True), _comparison(0.0, False)]},
               {"metrics": [_comparison(0.5, True)]}]
    s = stability_summary(records)
    assert s["n_comparisons"] == 3 and abs(s["prediction_retention"] - 2 / 3) < 1e-12
    assert abs(s["all"]["jaccard_at_5"]["mean"] - 0.5) < 1e-12
    assert abs(s["prediction_invariant"]["jaccard_at_5"]["mean"] - 0.75) < 1e-12


def test_faithfulness_rewards_the_decisive_word():
    # Toy model: class 1 iff the word "great" is present
    def predict(texts):
        return np.array([[0.1, 0.9] if "great" in t.split() else [0.9, 0.1] for t in texts])

    text = "the plot was great but long and slow overall"
    good = [[w, 1.0 if w == "great" else 0.0] for w in text.split()]
    records = [{"original_text": text, "original_explanation": {"predicted_class": 1, "words": good}}]
    result = faithfulness(records, predict, random_draws=20)
    assert abs(result["explainer"]["comprehensiveness"] - 0.8) < 1e-9   # removing "great" always flips
    assert abs(result["explainer"]["sufficiency"]) < 1e-9               # keeping "great" alone suffices
    assert result["random"]["comprehensiveness"] < result["explainer"]["comprehensiveness"]


def test_word_edit_fraction():
    from analysis.strength import word_edit_fraction
    assert word_edit_fraction("The Film is good", "the film is good") == 0.0
    assert word_edit_fraction("the film is good", "the movie is good") == 0.25
    assert word_edit_fraction("a b c d", "c d a b") == 1.0


def test_stability_by_strength_bins_and_correlation():
    from analysis.strength import stability_by_strength
    rows = [{"edit_fraction": f, "jaccard_at_5": 1 - f, "spearman_correlation": 1 - f, "same_prediction": True}
            for f in (0.0, 0.1, 0.2, 0.5)]
    out = stability_by_strength(rows)
    assert [b["n"] for b in out["bins"]] == [1, 1, 1, 1]
    assert out["spearman_strength_vs_jaccard_at_5"] < -0.99


def test_sensitivity_undefined_as_zero_and_base_names():
    from analysis.sensitivity import _base_name, undefined_as_zero
    assert _base_name("LIMELegacyAlignment") == "LIME" and _base_name("IntegratedGradientsLegacyConfig") == "IntegratedGradients"
    nan = float("nan")
    rows = [{"metrics": [_comparison(0.8, True) | {"spearman_correlation": nan}, _comparison(0.6, True)]}]
    out = undefined_as_zero({("LIME", "PreprocessingVariation"): rows})["LIME/PreprocessingVariation"]
    assert out["current"]["all"]["spearman_correlation"]["mean"] == 0.6
    assert abs(out["reverted"]["all"]["spearman_correlation"]["mean"] - 0.3) < 1e-12


def test_shared_random_stream_detects_differing_variants():
    from analysis.sensitivity import shared_random_stream
    texts = [f"the film number {i} is good and long and slow ." for i in range(30)]
    s = shared_random_stream(texts)
    assert s["instances"] == 30 and 0 < s["instances_with_differing_variants"] <= 30


def test_measurement_levenshtein_matches_reference():
    from analysis.measurement import levenshtein
    from analysis.strength import word_edit_fraction
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = rng.integers(0, 4, rng.integers(0, 9)).tolist(), rng.integers(0, 4, rng.integers(0, 9)).tolist()
        reference = word_edit_fraction(" ".join(map(str, a)), " ".join(map(str, b))) * len(a) if a else len(b)
        assert levenshtein(a, b) == round(reference)


def test_measurement_signed_jaccard_and_union_ad():
    from analysis.measurement import ad_union, jaccard_at_k
    a = {"good": 1.0, "bad": -0.9, "film": 0.1}
    b = {"good": 1.0, "bad": 0.9, "film": 0.1}
    assert jaccard_at_k(a, b, k=1) == 1.0
    assert jaccard_at_k(a, b, k=2, signed=True) == 1 / 3   # {good, film} vs {good, bad}
    assert ad_union({"x": 1.0}, {"y": 1.0}) == 100.0           # nothing shared: maximal difference


def test_measurement_cluster_interval_is_wider_for_correlated_comparisons():
    from analysis.measurement import interval_summary
    rng = np.random.default_rng(0)
    groups = [np.full(7, v) for v in rng.random(100)]          # identical comparisons within an instance
    s = interval_summary(groups, np.random.default_rng(0))
    assert s['cluster_ci_half_width'] > 2 * s['naive_ci_half_width']
    assert abs(s['comparison_mean'] - s['instance_mean']) < 1e-12


def test_instance_factors_margin_and_partial_correlation():
    from analysis.instance_factors import _partial, _spearman, instance_table
    attrs = {f"w{i}": v for i, v in enumerate([1.0, 0.9, 0.8, 0.7, 0.6, 0.2])}
    row = {"sample_idx": 3, "original_text": "a b c d e f",
           "original_explanation": {"attributions": attrs, "confidence": 0.9},
           "metrics": [{"jaccard_at_5": 0.5, "spearman_correlation": 0.4}, {"jaccard_at_5": 1.0, "spearman_correlation": 0.8}]}
    t = instance_table([row])[3]
    assert abs(t["top5_margin"] - 0.4) < 1e-12 and t["jaccard_at_5"] == 0.75 and t["length"] == 6
    rng = np.random.default_rng(0)
    z = rng.random(500)
    x, y = z + 0.1 * rng.random(500), z + 0.1 * rng.random(500)   # related only through z
    assert _spearman(x, y) > 0.8 and abs(_partial(x, y, z)) < 0.2


def test_rbo_ext_bounds_and_order_sensitivity():
    from analysis.robustness import rbo_ext
    a = list("abcdefgh")
    assert abs(rbo_ext(a, a) - 1.0) < 1e-12
    assert rbo_ext(a, list("ijklmnop")) == 0.0
    swapped_top, swapped_tail = list("bacdefgh"), list("abcdefhg")
    assert rbo_ext(a, swapped_tail) > rbo_ext(a, swapped_top)   # top-weighted
    assert abs(rbo_ext(a, list("abc")) - rbo_ext(list("abc"), a)) < 1e-12


def test_faithfulness_replacement_keeps_length():
    from analysis.faithfulness import _variants
    texts = _variants("a b c d e f g h i j".split(), np.arange(10), replacement="[MASK]")
    assert all(len(t.split()) == 10 for t in texts)
    assert texts[0].startswith("[MASK] b c") and texts[1].startswith("a [MASK]")
