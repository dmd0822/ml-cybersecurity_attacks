# `config/`

Configuration is kept separate from code so you can change runtime behavior (paths, hyperparameters, feature toggles) without editing pipeline logic.

## What belongs here

- Environment-specific settings (e.g., `local` vs `prod`)
- Model and training parameters (e.g., `random_seed`, CV folds, thresholds)
- Dataset locations and feature flags
- Runtime settings (logging level, output directories)

## What should NOT live here

- Secrets (API keys, passwords, tokens). Use environment variables or a secrets manager.
- Large data files. Keep those under `data/`.

## Suggested conventions

- Keep **one default config** and override per environment.
- Prefer **human-readable formats** like YAML/TOML/JSON.
- Treat configs as part of the project contract: review changes like code.

## Example (suggested)

```text
config/
  default.yaml
  local.yaml
  prod.yaml
```

## Tips

- Make your pipelines accept a config object/path so runs are reproducible.
- If you add config validation (recommended), fail fast with clear error messages.

## How This Fits

- Used by pipeline code in [`src/pipelines/`](../src/pipelines/)
- Referenced by runnable scripts in [`entrypoints/`](../entrypoints/)
- Settings often control where artifacts are written under [`data/`](../data/)
