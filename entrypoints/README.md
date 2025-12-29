# `entrypoints/`

This folder contains the runnable scripts for your project (training, batch inference, evaluation).

Keeping entry points explicit makes it much easier to:

- Run jobs consistently in CI
- Containerize with Docker
- Schedule runs (workflow schedulers, Airflow, Prefect, etc.)

## What belongs here

- Thin scripts that parse CLI arguments / load config
- Calls into pipeline code in `src/pipelines/`
- Orchestration glue (but not the core ML logic)

## Suggested scripts (examples)

## Current scripts

### Data

- `download_kaggle_dataset.py`: Download and snapshot the Kaggle dataset into `data/01-raw/`.
- `prepare_dataset.py`: Clean the latest raw snapshot and write a prepared dataset + shared split into `data/02-preprocessed/`.

### Notebooks

- `generate_experiment_notebooks.py`: Generate timestamped experiment notebooks under `notebooks/attack_type/`.

### Attack Type training

- `train_attack_type_random_forest.py`: Train Random Forest for `Attack Type`.
- `train_attack_type_xgboost.py`: Train XGBoost for `Attack Type`.

### Attack Type inference

- `infer_attack_type_random_forest.py`: Score a chosen split (`--split`) with the latest RF model.
- `infer_attack_type_xgboost.py`: Score a chosen split (`--split`) with the latest XGBoost model.

## Conventions

- Keep scripts **small and boring**: argument parsing + calling library functions.
- Put ML logic in `src/pipelines/` so it’s testable.
- Use a `if __name__ == "__main__":` guard.

## Example (suggested)

```bash
python entrypoints/prepare_dataset.py

python entrypoints/train_attack_type_random_forest.py --n-estimators 500
python entrypoints/train_attack_type_xgboost.py --n-estimators 600 --learning-rate 0.05

python entrypoints/infer_attack_type_random_forest.py --split test --limit 1000
python entrypoints/infer_attack_type_xgboost.py --split test --limit 1000
```

## How This Fits

- Loads settings from [`config/`](../config/)
- Calls reusable logic in [`src/pipelines/`](../src/pipelines/)
- Reads/writes staged artifacts under [`data/`](../data/)
- Experiments that become stable often start in [`notebooks/`](../notebooks/)
- Runtime resources are often provisioned via [`infra/`](../infra/)
