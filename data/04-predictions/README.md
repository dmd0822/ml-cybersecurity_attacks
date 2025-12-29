# `data/04-predictions/`

This folder stores **model outputs** produced by inference pipelines.

Predictions are often consumed by downstream systems (reports, APIs, dashboards) and should be versioned and traceable back to:

- the model version
- the feature set/version
- the input data snapshot

## What belongs here

- Batch prediction files
- Scored datasets
- Inference metadata (model ID, run timestamp, thresholds)

## Conventions (recommended)

- Version predictions by model/run.
- Include a lightweight manifest (inputs, config, model identifier).
- Avoid overwriting prediction outputs unless it is intentional.

## Example structure

```text
data/04-predictions/
  model_name/
    2025-12-22_run-001/
```

## Attack Type outputs

This project writes Attack Type artifacts under:

```text
data/04-predictions/
  attack_type/
    training/
      <timestamp>/
        attack_type_random_forest_model.joblib
        attack_type_random_forest_metrics.json
        attack_type_random_forest_test_predictions.csv
        attack_type_random_forest_manifest.json
        attack_type_xgboost_model.joblib
        attack_type_xgboost_metrics.json
        attack_type_xgboost_test_predictions.csv
        attack_type_xgboost_manifest.json
    inference/
      <timestamp>/
        attack_type_random_forest_test_predictions.csv
        attack_type_random_forest_test_manifest.json
        attack_type_xgboost_test_predictions.csv
        attack_type_xgboost_test_manifest.json
```

Each run is timestamped for traceability.

## How This Fits

- Upstream: feature artifacts in [`data/03-features/`](../03-features/)
- Produced by inference pipelines in pipeline code under `src/pipelines/` executed via [`entrypoints/`](../../entrypoints/)
- Often consumed by reporting, dashboards, or application services outside this repo

