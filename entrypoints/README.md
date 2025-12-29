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

You can add scripts like:

- `train.py` (train a model)
- `predict.py` (generate predictions)
- `evaluate.py` (compute metrics / reports)

## Conventions

- Keep scripts **small and boring**: argument parsing + calling library functions.
- Put ML logic in `src/pipelines/` so it’s testable.
- Use a `if __name__ == "__main__":` guard.

## Example (suggested)

```bash
python -m entrypoints.train --config config/local.yaml
python -m entrypoints.predict --config config/prod.yaml
```

## How This Fits

- Loads settings from [`config/`](../config/)
- Calls reusable logic in [`src/pipelines/`](../src/pipelines/)
- Reads/writes staged artifacts under [`data/`](../data/)
- Experiments that become stable often start in [`notebooks/`](../notebooks/)
- Runtime resources are often provisioned via [`infra/`](../infra/)
