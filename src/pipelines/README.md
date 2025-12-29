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

## How This Fits

- Implemented and tested here, executed via [`entrypoints/`](../../entrypoints/)
- Typically configured via [`config/`](../../config/)
- Consumes/produces artifacts in staged folders under [`data/`](../../data/)
- Supported by automated checks in [`tests/`](../../tests/)
