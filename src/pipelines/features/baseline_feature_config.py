"""Baseline feature selection/transforms for experiment notebooks.

The dataset prep step writes these artifacts:
- `feature_audit.csv`: per-column quality signals + suggested action
- `config/baseline_feature_config.json`: a deterministic baseline feature config

Experiment notebooks should load `config/baseline_feature_config.json` and apply it
consistently so all models compare on the same feature set.

This module implements:
- Loading the baseline feature config
- Applying baseline transforms (timestamp and port feature engineering)
- Dropping baseline-excluded columns

Important:
- This module does NOT do one-hot encoding. That stays inside experiment
  pipelines (e.g., sklearn ColumnTransformer + OneHotEncoder).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class BaselineFeatureConfig:
    """Baseline feature configuration loaded from JSON."""

    target_col: str
    row_id_col: str
    drop_cols: list[str]
    timestamp_cols: list[str]
    port_cols: list[str]


def load_baseline_feature_config(path: Path) -> BaselineFeatureConfig:
    """Load a baseline feature config from JSON.

    Parameters:
        path: Path to `config/baseline_feature_config.json`.

    Returns:
        BaselineFeatureConfig: Parsed configuration.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If required keys are missing.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))

    try:
        target_col = str(payload["target_col"])
        row_id_col = str(payload.get("row_id_col", "row_id"))
        drop_cols = [str(c) for c in payload.get("drop_cols", [])]

        transform = payload.get("transform", {})
        timestamp_cols = [str(c) for c in transform.get("timestamp_cols", [])]
        port_cols = [str(c) for c in transform.get("port_cols", [])]
    except Exception as exc:
        raise ValueError("Invalid baseline feature config JSON") from exc

    return BaselineFeatureConfig(
        target_col=target_col,
        row_id_col=row_id_col,
        drop_cols=drop_cols,
        timestamp_cols=timestamp_cols,
        port_cols=port_cols,
    )


def _ensure_columns_exist(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    """Return only columns that exist in df."""

    existing = set(df.columns)
    return [c for c in cols if c in existing]


def _add_timestamp_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add simple timestamp-derived features.

    Parameters:
        df: Input dataframe.
        col: Name of the timestamp column.

    Returns:
        Dataframe with added columns.
    """

    df = df.copy()

    ts = pd.to_datetime(df[col], errors="coerce", utc=True)

    prefix = col.replace(" ", "_")
    df[f"{prefix}_hour"] = ts.dt.hour
    df[f"{prefix}_dayofweek"] = ts.dt.dayofweek
    df[f"{prefix}_month"] = ts.dt.month
    df[f"{prefix}_is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype("Int64")

    return df


def _add_port_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add port bucket features.

    Parameters:
        df: Input dataframe.
        col: Name of the port column.

    Returns:
        Dataframe with added bucket columns.
    """

    df = df.copy()

    port = pd.to_numeric(df[col], errors="coerce")
    prefix = col.replace(" ", "_")

    # Common port ranges: well-known [0,1023], registered [1024,49151], dynamic [49152,65535]
    df[f"{prefix}_is_well_known"] = (port.between(0, 1023)).astype("Int64")
    df[f"{prefix}_is_registered"] = (port.between(1024, 49151)).astype("Int64")
    df[f"{prefix}_is_dynamic"] = (port.between(49152, 65535)).astype("Int64")

    return df


def apply_baseline_feature_config(
    df: pd.DataFrame,
    config: BaselineFeatureConfig,
    *,
    drop_original_timestamp_cols: bool = True,
) -> pd.DataFrame:
    """Apply baseline feature selection and transforms.

    This function:
    - Adds derived timestamp features
    - Adds port bucket features
    - Drops baseline-excluded columns, plus target/row_id columns

    Parameters:
        df: Cleaned full dataset dataframe.
        config: Baseline feature configuration.
        drop_original_timestamp_cols: If True, drop the original timestamp
            columns after creating derived features.

    Returns:
        pd.DataFrame: Feature matrix suitable for downstream encoding.
    """

    out = df.copy()

    timestamp_cols = _ensure_columns_exist(out, config.timestamp_cols)
    port_cols = _ensure_columns_exist(out, config.port_cols)

    for col in timestamp_cols:
        out = _add_timestamp_features(out, col)

    for col in port_cols:
        out = _add_port_features(out, col)

    drop_cols: list[str] = []
    drop_cols.extend(_ensure_columns_exist(out, config.drop_cols))

    # Always remove label + row_id from feature matrix.
    drop_cols.extend(_ensure_columns_exist(out, [config.target_col, config.row_id_col]))

    if drop_original_timestamp_cols:
        drop_cols.extend(timestamp_cols)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    deduped = []
    for c in drop_cols:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    return out.drop(columns=deduped, errors="ignore")


def split_xy(
    df: pd.DataFrame,
    config: BaselineFeatureConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into (X, y) using the config's target column.

    Parameters:
        df: Full dataset dataframe.
        config: Baseline feature config.

    Returns:
        (X, y): Features and label.

    Raises:
        KeyError: If the target column is missing.
    """

    y = df[config.target_col]
    X = apply_baseline_feature_config(df, config)
    return X, y
