"""
Experiment 1: DistilBERT on SST-2

DistilBERT fine-tuned on SST-2, full validation set (872 instances).

Usage (from the repository root):
    uv run python experiment_runners/experiment_1_distilbert_sst2.py [--max-samples N] [--device cuda]
"""

from common import run_experiment


def main(argv=None):
    return run_experiment(
        experiment_key='experiment_1',
        experiment_name='experiment_1_distilbert_sst2',
        title='Experiment 1: DistilBERT on SST-2',
        dataset_config={'config_key': 'sst2', 'sample_size': None, 'stratify': False},
        argv=argv
    )


if __name__ == "__main__":
    main()
