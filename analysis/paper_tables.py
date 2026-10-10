"""
Tables I and II of the IKT 2025 paper, transcribed from the published PDF.

Keys: (experiment, explainer, stress test) -> {metric: (mean, sd)} with
metrics J@5 ('jaccard_at_5'), rho ('spearman_correlation') and AD % 
('attr_diff_relative_difference_percentage').
"""

_COLUMNS = ('jaccard_at_5', 'spearman_correlation', 'attr_diff_relative_difference_percentage')
_EXPLAINERS = {'IG': 'IntegratedGradients', 'LIME': 'LIME', 'SHAP': 'SHAP'}
_TESTS = {'PREP': 'PreprocessingVariation', 'PARA': 'SemanticParaphrasing', 'SV': 'SeedVariation'}

_TABLE_I = """
experiment_1 IG PREP 0.471 0.217 0.687 0.223 20.811 9.635
experiment_1 IG PARA 0.400 0.200 0.626 0.258 22.939 9.775
experiment_1 LIME SV 0.699 0.194 0.791 0.113 1.002 0.678
experiment_1 LIME PREP 0.444 0.351 0.546 0.423 10.995 15.267
experiment_1 LIME PARA 0.570 0.206 0.716 0.381 15.819 12.796
experiment_1 SHAP PREP 0.482 0.362 0.876 0.290 7.392 10.611
experiment_1 SHAP PARA 0.639 0.233 0.823 0.389 7.681 9.451
experiment_2 IG PREP 0.196 0.188 0.357 0.382 8.5 4.2
experiment_2 IG PARA 0.289 0.207 0.501 0.294 9.1 4.6
experiment_2 LIME SV 0.801 0.200 0.952 0.037 6.702 2.706
experiment_2 LIME PREP 0.387 0.323 0.554 0.418 14.069 15.238
experiment_2 LIME PARA 0.544 0.198 0.844 0.220 15.389 10.210
experiment_2 SHAP PREP 0.298 0.279 0.714 0.326 0.892 1.156
experiment_2 SHAP PARA 0.572 0.288 0.856 0.203 0.645 0.845
"""

_TABLE_II = """
experiment_3 IG PREP 0.284 0.216 0.489 0.324 12.5 6.2
experiment_3 IG PARA 0.407 0.194 0.607 0.176 13.1 6.5
experiment_3 LIME SV 0.604 0.237 0.638 0.159 0.941 0.663
experiment_3 LIME PREP 0.358 0.276 0.488 0.393 2.761 5.397
experiment_3 LIME PARA 0.556 0.233 0.602 0.225 1.200 1.452
experiment_3 SHAP PREP 0.375 0.317 0.604 0.379 1.745 1.556
experiment_3 SHAP PARA 0.685 0.253 0.861 0.200 0.804 0.988
experiment_4 IG PREP 0.486 0.225 0.751 0.206 11.916 5.854
experiment_4 IG PARA 0.488 0.216 0.743 0.209 11.873 5.759
experiment_4 LIME SV 0.421 0.272 0.616 0.175 2.387 1.136
experiment_4 LIME PREP 0.255 0.243 0.549 0.320 2.885 1.871
experiment_4 LIME PARA 0.390 0.248 0.592 0.221 2.482 1.152
experiment_4 SHAP PREP 0.435 0.305 0.913 0.158 0.367 0.515
experiment_4 SHAP PARA 0.733 0.252 0.919 0.186 0.316 0.390
"""


def _parse(block):
    rows = {}
    for line in block.strip().splitlines():
        exp, explainer, test, *values = line.split()
        nums = [float(v) for v in values]
        rows[(exp, _EXPLAINERS[explainer], _TESTS[test])] = {
            col: (nums[2 * i], nums[2 * i + 1]) for i, col in enumerate(_COLUMNS)
        }
    return rows


PAPER_RESULTS = {**_parse(_TABLE_I), **_parse(_TABLE_II)}
