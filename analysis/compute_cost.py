"""
Compute cost of the explainers: wall-clock seconds per explanation.

Times each explainer, with the paper's settings, on the first N original texts
of an experiment's per-instance records (after one warm-up explanation), on
the given device. One explanation per text; a stress test multiplies this by
the number of variants (about 4-8).

Usage (from the repository root):
    uv run python -m analysis.compute_cost experiment_1 [--device mps] [--n 20]
"""

import argparse
import json
import time

import numpy as np

from analysis.report import REPO_ROOT, RUNNER_NAMES, latest_instance_dir
from analysis.stability import load_per_instance


def build(experiment_key: str, device: str, n: int):
    import torch
    from experiment_runners.common import EXPLAINER_PARAMS, RANDOM_SEED
    from stability_framework.explainers import create_explainer
    from stability_framework.model_loader import EXPERIMENT_MODELS, ModelLoader

    records = load_per_instance(latest_instance_dir(experiment_key))
    texts = [r['original_text'] for r in records[('LIME', 'PreprocessingVariation')][:n + 1]]
    model, tokenizer = ModelLoader(device=device).load_model_and_tokenizer(EXPERIMENT_MODELS[experiment_key]['name'])
    out = {'experiment': experiment_key, 'device': device, 'n': n,
           'mean_words': float(np.mean([len(t.split()) for t in texts[1:]])), 'explainers': {}}
    for name in ('integrated_gradients', 'lime', 'shap'):
        explainer = create_explainer(name, random_seed=RANDOM_SEED, **EXPLAINER_PARAMS[name])
        explainer.explain(model, tokenizer, texts[0])  # warm-up
        seconds = []
        for text in texts[1:]:
            if device == 'mps':
                torch.mps.synchronize()
            start = time.perf_counter()
            explainer.explain(model, tokenizer, text)
            if device == 'mps':
                torch.mps.synchronize()
            seconds.append(time.perf_counter() - start)
        out['explainers'][name] = {'mean_s': float(np.mean(seconds)), 'median_s': float(np.median(seconds)),
                                   'max_s': float(np.max(seconds))}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('experiment', choices=sorted(RUNNER_NAMES))
    parser.add_argument('--device', default='mps')
    parser.add_argument('--n', type=int, default=20)
    args = parser.parse_args(argv)
    result = build(args.experiment, args.device, args.n)
    out = REPO_ROOT / 'results' / 'analysis'
    (out / f'{args.experiment}_compute_cost.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
