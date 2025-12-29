# `reports/`

This folder contains human-readable reports that summarize the results of experiments and model runs.

Reports are intended for sharing (with teammates, stakeholders, or your future self), not for storing raw artifacts.

## Standard report format

To keep experiment summaries consistent and easy to compare, use the same section structure for each report.

### File naming

Use a date-stamped filename so reports sort naturally:

- `report_YYYY-MM-DD.md` (example: `report_2025-12-29.md`)

If you prefer a per-experiment folder (also supported), use `report.md` inside a dated experiment folder.

### Required section structure

Use these titles and numbering exactly:

1: Introduction and Problem Definition
1.1 Problem description and motivation
1.2 Project goals
1.3 Dataset description

2: Methodology
2.1 Preprocessing Pipeline
2.2 Model architecture and justification
2.3 Training strategy
2.4 Performance validation

3: Results and Evaluation
3.1 Quantitative results
3.2 Interpretation of results
3.3 Summary of final model performance

4: Discussion and Reflection
4.1 Limitations and sources of error
4.2 Potential improvements
4.3 Reflection on what you learned

Tip: Use Markdown headings for readability (`##` for the top-level numbered sections and `###` for subsections).

## What belongs here

- Experiment summaries (what changed, what was tested, what won/lost)
- Model evaluation write-ups (metrics, error analysis, slices)
- Comparisons across runs or feature sets
- Plots and figures used in write-ups (keep them reasonably sized)
- Run manifests or short run cards (linking to inputs/config/model identifiers)

## What should NOT live here

- Large datasets or intermediate artifacts (use `data/` stages)
- Model binaries (store in your chosen model registry/artifact store, or in a dedicated folder if you add one)
- Secrets (keys/tokens/passwords)

## Suggested layout (example)

Organize by time, experiment ID, or model name—whatever best matches your workflow.

```text
reports/
  2025-12-29_experiment-001_baseline/
    report.md
    figures/
      roc_curve.png
      pr_curve.png
  2025-12-30_experiment-002_feature-set-a/
    report.md
```

## Naming conventions (recommended)

- Prefer sortable, descriptive folder names:

  - `YYYY-MM-DD_experiment-###_short-name/`

- Keep a consistent report entry file name:

  - `report.md` (or `report.html` / `report.ipynb` if that’s your standard)

## How This Fits

- Inputs typically come from staged artifacts under `data/`:
  - cleaned datasets → `data/02-preprocessed/`
  - feature sets → `data/03-features/`
  - predictions → `data/04-predictions/`
- Results are produced by pipeline code in `src/pipelines/` and executed via `entrypoints/`.
- If you use notebooks for exploration, the final narrative should be distilled into a report here.
