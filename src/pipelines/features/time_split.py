"""Utilities for creating time-aware train/val/test splits.

The project uses shared split artifacts to ensure experiment notebooks evaluate
models on identical rows.

For time-aware evaluation, we split chronologically using an event-time column
(e.g., `Timestamp`). To minimize leakage, all rows sharing the same timestamp
are kept in the same split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


SplitLabel = Literal["train", "val", "test"]


@dataclass(frozen=True)
class TimeSplitReport:
    """Metadata about a generated time-aware split."""

    rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_target: int
    val_target: int


def make_time_aware_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    row_id_col: str = "row_id",
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, TimeSplitReport]:
    """Create a deterministic chronological split keyed by row id.

    Rows are ordered by (time_col ASC, row_id_col ASC). All rows with identical
    timestamps are assigned to the same split (timestamp-grouped splitting).

    Parameters:
        df: Dataframe containing at least time_col and row_id_col.
        time_col: Column name defining event time.
        row_id_col: Stable row identifier column.
        train_size: Train fraction.
        val_size: Validation fraction.
        test_size: Test fraction.

    Returns:
        (split_df, report) where split_df has columns [row_id_col, time_col, split].

    Raises:
        ValueError: If fractions are invalid or time_col has missing values.
        KeyError: If required columns are missing.
    """

    if not abs((train_size + val_size + test_size) - 1.0) < 1e-9:
        raise ValueError("train/val/test fractions must sum to 1.0")

    if time_col not in df.columns:
        raise KeyError(f"Missing time column: {time_col}")
    if row_id_col not in df.columns:
        raise KeyError(f"Missing row id column: {row_id_col}")

    working = df[[row_id_col, time_col]].copy()

    # Ensure time is datetime in UTC.
    if not pd.api.types.is_datetime64_any_dtype(working[time_col]):
        working[time_col] = pd.to_datetime(working[time_col], errors="coerce", utc=True)

    missing_time = int(working[time_col].isna().sum())
    if missing_time:
        raise ValueError(
            f"{missing_time} rows have missing/invalid {time_col}; cannot time-split"
        )

    working = working.sort_values(
        by=[time_col, row_id_col], ascending=[True, True]
    ).reset_index(drop=True)

    n = int(working.shape[0])
    train_target = int(n * train_size)
    val_target = int(n * (train_size + val_size))

    split_col: list[SplitLabel] = []
    current_split: SplitLabel = "train"
    seen = 0

    # Walk in timestamp groups.
    for ts, group in working.groupby(time_col, sort=False):
        group_n = int(group.shape[0])

        # Assign whole group to current split.
        split_col.extend([current_split] * group_n)
        seen += group_n

        # Advance split after finishing the group.
        if current_split == "train" and seen >= train_target:
            current_split = "val"
        elif current_split == "val" and seen >= val_target:
            current_split = "test"
        else:
            _ = ts  # keep loop variable referenced for linters

    out = working.copy()
    out["split"] = split_col

    report = TimeSplitReport(
        rows=n,
        train_rows=int((out["split"] == "train").sum()),
        val_rows=int((out["split"] == "val").sum()),
        test_rows=int((out["split"] == "test").sum()),
        train_target=train_target,
        val_target=val_target,
    )

    return out[[row_id_col, time_col, "split"]], report
