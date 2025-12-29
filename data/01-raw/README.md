# `data/01-raw/`

This folder stores **raw, immutable inputs**.

The goal is to preserve original data exactly as received so the full pipeline is reproducible and auditable.

## What belongs here

- Vendor extracts / source dumps
- Unmodified CSV/JSON/Parquet files
- Raw images/audio/text corpora
- Metadata about where the data came from

## Conventions (recommended)

- Do not edit files in-place.
- Keep a clear provenance trail (source, date, query, version).
- If you must re-pull raw data, write it to a new versioned path.

## Example structure

```text
data/01-raw/
  source_a/2025-12-22/
  source_b/v1/
  teamincribo_cyber-security-attacks/2025-12-29/
```

For this project, raw snapshots are stored under:

- `data/01-raw/teamincribo_cyber-security-attacks/<YYYY-MM-DD>/`

You can create a new snapshot using:

```bash
python entrypoints/download_kaggle_dataset.py --snapshot-date YYYY-MM-DD
```

## How This Fits

- Upstream: external systems / data sources
- Downstream: preprocessing outputs in [`data/02-preprocessed/`](../02-preprocessed/)
- Processing logic typically lives in pipeline code under `src/pipelines/` and is run via [`entrypoints/`](../../entrypoints/)

