# `notebooks/`

This folder is for exploratory work: EDA, quick baselines, and one-off analysis.

Keeping notebooks separate helps prevent exploratory code from drifting into production pipelines.

## What belongs here

- Exploratory data analysis (EDA)
- Baseline modeling experiments
- Visualizations and investigation notebooks
- Notes that help you decide what to productionize

Experiment notebooks for predicting `Attack Type` live under `notebooks/attack_type/`.

Most of the timestamped notebooks in `notebooks/attack_type/` were generated via:

```bash
python entrypoints/generate_experiment_notebooks.py
```

## What should NOT live here

- Core pipeline logic that needs to be reused (move that to `src/pipelines/`)
- Long-running scheduled jobs
- Secrets (keys/tokens)

## Conventions (suggested)

- Use descriptive names with a sortable prefix:

- `01_eda_overview.ipynb`
- `02_baseline_model.ipynb`
- `03_error_analysis.ipynb`

- Keep outputs lightweight. Prefer saving large artifacts to `data/`.
- If you export results, write them into the appropriate stage:

- cleaned datasets → `data/02-preprocessed/`
- features → `data/03-features/`
- predictions → `data/04-predictions/`

## Tip

Once an approach is working, extract the reusable parts into pipeline code so it can be tested and automated.

## How This Fits

- Reads inputs from [`data/01-raw/`](../data/01-raw/) and [`data/02-preprocessed/`](../data/02-preprocessed/)
- Can write derived artifacts to [`data/02-preprocessed/`](../data/02-preprocessed/), [`data/03-features/`](../data/03-features/), or [`data/04-predictions/`](../data/04-predictions/)
- Mature logic should move into pipeline code in `src/pipelines/` and be exercised via [`entrypoints/`](../entrypoints/)
