"""
Experiment 3: RoBERTa on AG News

RoBERTa-base fine-tuned on AG News, stratified test sample (2000 instances,
500 per class).

Usage (from the repository root):
    uv run python experiment_runners/experiment_3_roberta_agnews.py [--max-samples N] [--device cuda]
"""

from common import run_experiment


def main(argv=None):
    return run_experiment(
        experiment_key='experiment_3',
        experiment_name='experiment_3_roberta_agnews',
        title='Experiment 3: RoBERTa on AG News',
        dataset_config={'config_key': 'ag_news', 'sample_size': 2000, 'stratify': True},
        argv=argv
    )


if __name__ == "__main__":
    main()
