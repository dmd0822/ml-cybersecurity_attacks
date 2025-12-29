# `tests/`

This folder is for automated tests.

Even a small number of tests goes a long way when refactoring pipelines, adding features, or changing data contracts.

## What to test

- Data contracts: schemas, required columns, expected data types
- Feature engineering: deterministic transforms, handling of missing values
- Training: reproducible splits, consistent metrics, model serialization behavior
- Inference: expected outputs for known inputs, batch edge cases

## Suggested structure

Mirror the code structure so tests are easy to find.

```text
tests/
  pipelines/
  fixtures/
```

## Running tests

Use whatever test runner matches your stack.

If you’re using Python, a common choice is `pytest`:

```bash
pytest
```

## Current test suite

- `test_python_files_compile.py`: Syntax/compile smoke test for all Python files under `src/` and `entrypoints/`.
- `test_attack_type_training_utils.py`: Unit tests for Attack Type pipeline utilities (dataset selection, splits, preprocessing, and error cases).

## How This Fits

- Primarily validates logic in [`src/pipelines/`](../src/pipelines/)
- Can also cover runnable behavior in [`entrypoints/`](../entrypoints/)
- Test data/fixtures may live under `tests/fixtures/` and should not replace staged artifacts in [`data/`](../data/)
