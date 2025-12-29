# `src/pipelines/`

This folder contains reusable, testable pipeline code.

Treat feature engineering, training, inference, and evaluation as pipelines rather than one-off scripts. This makes your work easier to:

- Reuse
- Test
- Automate
- Productionize

## What belongs here

- Feature engineering and transformation logic
- Training routines (model fitting, validation, model selection)
- Inference routines (batch/online prediction logic)
- Evaluation and metric computation
- Shared utilities used by multiple pipelines

## Suggested layout (example)

```text
src/
  pipelines/
    features/
    attack_type/
    train/
    infer/
    evaluate/
    common/
```

## Design principles (recommended)

- Prefer small, pure functions where possible.
- Keep I/O at the edges: read inputs, call pipeline logic, write outputs.
- Make pipeline steps deterministic given config + input data.
- Avoid hard-coded paths; take inputs/outputs from config or parameters.

## Relationship to `entrypoints/`

Entry points should be thin wrappers that call the functions defined here.

## Attack Type training

Training entry points for the multiclass `Attack Type` target:

- `python entrypoints/train_attack_type_random_forest.py`
- `python entrypoints/train_attack_type_xgboost.py`

Common options:

- Random Forest: `--n-estimators`, `--max-depth`, `--seed`
- XGBoost: `--n-estimators`, `--max-depth`, `--learning-rate`, `--seed`

Artifacts are written under `data/04-predictions/attack_type/training/<timestamp>/`.
All filenames are prefixed with `attack_type_*` for traceability.

## Attack Type inference

Inference entry points (score the latest prepared dataset split using the latest
trained model artifact):

- `python entrypoints/infer_attack_type_random_forest.py --split test`
- `python entrypoints/infer_attack_type_xgboost.py --split test`

Common options:

- `--split {train,val,test}`
- `--limit N` (quick smoke runs)
- `--model-path <path>` (score a specific saved model)

## Dataset preparation

Both notebooks and pipelines expect a prepared dataset under `data/02-preprocessed/`.
To generate it from the raw snapshot in `data/01-raw/`:

- `python entrypoints/prepare_dataset.py`

Artifacts are written under `data/04-predictions/attack_type/inference/<timestamp>/`.

## How This Fits

- Implemented and tested here, executed via [`entrypoints/`](../../entrypoints/)
- Typically configured via [`config/`](../../config/)
- Consumes/produces artifacts in staged folders under [`data/`](../../data/)
- Supported by automated checks in [`tests/`](../../tests/)
