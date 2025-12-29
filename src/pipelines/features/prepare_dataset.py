"""Prepare the cybersecurity attacks dataset for modeling.

This module implements dataset preparation for the Kaggle dataset
`teamincribo/cyber-security-attacks`.

Outputs are written to `data/02-preprocessed/` as:
- a cleaned tabular dataset (Parquet when available, else CSV)
- a shared stratified split (train/val/test) based on the target column
- a class mapping for the target column
- a feature audit table and a baseline feature config

Notes:
- Target column is `Attack Type` (multiclass classification).
- Predictor encoding strategy for experiments is one-hot encoding, but this
  module only prepares a *clean tabular* dataset + splits; individual
  experiments can fit their own encoders/pipelines.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.model_selection import train_test_split


TargetColumn = Literal["Attack Type"]


@dataclass(frozen=True)
class PreparedDatasetArtifacts:
    """Paths to artifacts produced by dataset preparation."""

    output_dir: str
    cleaned_data_path: str
    split_path: str
    class_map_path: str
    feature_audit_path: str
    baseline_feature_config_path: str
    manifest_path: str


def _utc_now_iso() -> str:
    """Return current UTC time as ISO string."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light normalization to column names and values.

    Parameters:
        df: Raw dataframe.

    Returns:
        A dataframe with standardized string columns.
    """

    df = df.copy()

    # Strip column names; keep original casing to match dataset docs.
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize common string columns: trim whitespace and standardize empties.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known columns to better dtypes.

    Parameters:
        df: Input dataframe.

    Returns:
        Dataframe with coerced dtypes.
    """

    df = df.copy()

    # Timestamp to datetime (best-effort).
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)

    # Numeric best-effort coercions.
    for numeric_col in [
        "Source Port",
        "Destination Port",
        "Packet Length",
        "Anomaly Scores",
    ]:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    return df


def _drop_empty_rows(df: pd.DataFrame, *, target_col: TargetColumn) -> pd.DataFrame:
    """Drop rows with missing target and rows that are fully empty.

    Parameters:
        df: Input dataframe.
        target_col: Target column name.

    Returns:
        Filtered dataframe.
    """

    df = df.copy()

    # Drop rows with missing target.
    df = df.dropna(subset=[target_col])

    # Drop rows that are entirely NA (should be rare).
    df = df.dropna(how="all")

    return df


def _make_row_id(df: pd.DataFrame) -> pd.Series:
    """Create a stable-ish row id within the cleaned dataset.

    This is used for split files so they are resilient to file reordering.

    Parameters:
        df: Dataframe.

    Returns:
        A pandas Series of string row IDs.
    """

    # Use existing index after reset for determinism.
    return pd.Series([f"row_{i:06d}" for i in range(len(df))], index=df.index)


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to Parquet when available, else CSV.

    Parameters:
        df: Dataframe to write.
        path: Path with suffix .parquet or .csv.
    """

    if path.suffix.lower() == ".parquet":
        # pandas will require pyarrow/fastparquet; if not installed, fall back.
        try:
            df.to_parquet(path, index=False)
            return
        except Exception:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return

    df.to_csv(path, index=False)


def _safe_str(value: Any) -> str:
    """Safely convert a value to a string.

    Parameters:
        value: Any value, including null-like values.

    Returns:
        A string representation or an empty string for missing values.
    """

    try:
        return "" if pd.isna(value) else str(value)
    except Exception:
        return ""


def _text_length_stats(series: pd.Series, *, seed: int = 42) -> dict[str, float]:
    """Compute cheap text length stats on a (possibly sampled) series.

    Parameters:
        series: Input series.
        seed: Random seed used if sampling is needed.

    Returns:
        Dict with avg/p95/max length metrics.
    """

    sample = series.dropna()

    # Keep this bounded if a column is unexpectedly huge.
    if len(sample) > 10_000:
        sample = sample.sample(10_000, random_state=seed)

    lengths = sample.map(lambda v: len(_safe_str(v))).astype(int)

    if lengths.empty:
        return {"avg_len": 0.0, "p95_len": 0.0, "max_len": 0.0}

    return {
        "avg_len": float(lengths.mean()),
        "p95_len": float(lengths.quantile(0.95)),
        "max_len": float(lengths.max()),
    }


def _build_feature_audit(
    df: pd.DataFrame,
    *,
    target_col: TargetColumn,
    seed: int,
) -> pd.DataFrame:
    """Build a feature audit table with heuristic suggested actions.

    The purpose of this audit is to produce a consistent baseline feature
    selection for experiment notebooks (drop/transform/keep) without manual
    per-notebook decisions.

    Parameters:
        df: Cleaned dataset dataframe.
        target_col: Target column name.
        seed: Random seed (used for sampling text columns).

    Returns:
        A dataframe with one row per column.
    """

    high_missing_threshold = 0.40
    high_cardinality_threshold = 500
    free_text_p95_len = 30
    free_text_max_len = 120

    rows: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        missing_frac = float(series.isna().mean())
        nunique = int(series.nunique(dropna=True))
        dtype = str(series.dtype)

        is_target = col == target_col
        is_row_id = col.lower() == "row_id"

        is_object_like = dtype == "object" or dtype.startswith("string")
        text_stats: dict[str, float] = {"avg_len": 0.0, "p95_len": 0.0,
                                        "max_len": 0.0}
        if is_object_like:
            text_stats = _text_length_stats(series, seed=seed)

        col_lc = col.lower()

        likely_id = (
            any(tok in col_lc for tok in ["id", "uuid", "guid"]) and not is_row_id
        )
        likely_ip = "ip" in col_lc
        likely_timestamp = any(tok in col_lc for tok in ["time", "date", "timestamp"])
        likely_port = "port" in col_lc

        likely_free_text = bool(
            is_object_like
            and (
                text_stats["p95_len"] >= free_text_p95_len
                or text_stats["max_len"] >= free_text_max_len
            )
        )

        high_missing = missing_frac >= high_missing_threshold
        high_cardinality = nunique >= high_cardinality_threshold

        transform_kind = ""

        if is_target:
            action = "target"
            reason = "label column"
        elif is_row_id:
            action = "drop"
            reason = "surrogate key"
        elif likely_ip:
            action = "drop"
            reason = "high-cardinality identifier (likely)"
        elif likely_id and high_cardinality:
            action = "drop"
            reason = "identifier-like + high cardinality"
        elif likely_timestamp:
            action = "transform"
            transform_kind = "timestamp"
            reason = "extract time-based features"
        elif likely_port:
            action = "transform"
            transform_kind = "port"
            reason = "bucket/parse port features"
        elif likely_free_text:
            action = "drop"
            reason = "free-text/log-like (one-hot unfriendly)"
        elif high_missing and is_object_like and nunique <= 2:
            action = "keep"
            reason = "binary-ish; treat missing as category"
        elif high_missing:
            action = "drop"
            reason = "too missing (baseline)"
        elif high_cardinality and is_object_like:
            action = "drop"
            reason = "high-cardinality categorical (baseline)"
        else:
            action = "keep"
            reason = "ok for baseline"

        rows.append(
            {
                "feature": col,
                "dtype": dtype,
                "missing_pct": round(missing_frac * 100, 2),
                "n_unique": nunique,
                "avg_len": round(text_stats["avg_len"], 2)
                if is_object_like
                else None,
                "p95_len": round(text_stats["p95_len"], 2)
                if is_object_like
                else None,
                "max_len": round(text_stats["max_len"], 2)
                if is_object_like
                else None,
                "suggested_action": action,
                "transform_kind": transform_kind,
                "reason": reason,
            }
        )

    audit = pd.DataFrame(rows)
    audit = audit.sort_values(
        by=["suggested_action", "missing_pct", "n_unique"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return audit


def _build_baseline_feature_config(
    audit: pd.DataFrame,
    *,
    target_col: TargetColumn,
) -> dict[str, Any]:
    """Build a baseline feature selection config from the audit table.

    Parameters:
        audit: Feature audit dataframe.
        target_col: Target column name.

    Returns:
        A JSON-serializable dict describing columns to drop and transform.
    """

    drop_cols = (
        audit.loc[audit["suggested_action"].eq("drop"), "feature"]
        .astype(str)
        .tolist()
    )

    timestamp_cols = (
        audit.loc[audit["transform_kind"].eq("timestamp"), "feature"]
        .astype(str)
        .tolist()
    )
    port_cols = (
        audit.loc[audit["transform_kind"].eq("port"), "feature"]
        .astype(str)
        .tolist()
    )

    return {
        "target_col": target_col,
        "row_id_col": "row_id",
        "drop_cols": drop_cols,
        "transform": {
            "timestamp_cols": timestamp_cols,
            "port_cols": port_cols,
        },
        "heuristics": {
            "high_missing_threshold": 0.40,
            "high_cardinality_threshold": 500,
            "free_text_p95_len": 30,
            "free_text_max_len": 120,
        },
    }


def prepare_dataset(
    *,
    raw_csv_path: Path,
    output_dir: Path,
    baseline_feature_config_path: Path | None = None,
    target_col: TargetColumn = "Attack Type",
    seed: int = 42,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> PreparedDatasetArtifacts:
    """Clean the raw dataset and produce a shared stratified split.

    Parameters:
        raw_csv_path: Path to the raw CSV.
        output_dir: Directory where outputs will be written.
        baseline_feature_config_path: Optional canonical path where the
            baseline feature config should be written (e.g.
            `config/baseline_feature_config.json`). If provided, a snapshot copy
            is still written into the prepared dataset folder.
        target_col: Target column name.
        seed: Random seed.
        train_size: Fraction of rows for training.
        val_size: Fraction of rows for validation.
        test_size: Fraction of rows for testing.

    Returns:
        PreparedDatasetArtifacts: Paths to written artifacts.

    Raises:
        ValueError: If split fractions are invalid.
    """

    if not abs((train_size + val_size + test_size) - 1.0) < 1e-9:
        raise ValueError("train/val/test fractions must sum to 1.0")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv_path)
    df = _standardize_columns(df)
    df = _coerce_types(df)
    df = _drop_empty_rows(df, target_col=target_col)

    df = df.reset_index(drop=True)
    df.insert(0, "row_id", _make_row_id(df))

    # Target class mapping
    classes = sorted(df[target_col].dropna().astype(str).unique().tolist())
    class_to_index = {name: idx for idx, name in enumerate(classes)}
    class_map_path = output_dir / "attack_type_classes.json"
    class_map_path.write_text(
        json.dumps(
            {"target": target_col, "classes": classes, "class_to_index": class_to_index},
            indent=2,
        ),
        encoding="utf-8",
    )

    # Shared stratified split based on row_id
    y = df[target_col].astype(str)

    train_df, temp_df = train_test_split(
        df[["row_id", target_col]],
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    # Split remaining into val/test with correct proportions.
    temp_y = temp_df[target_col].astype(str)
    val_fraction_of_temp = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_fraction_of_temp,
        random_state=seed,
        shuffle=True,
        stratify=temp_y,
    )

    split_path = output_dir / "split.csv"
    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    split_df.to_csv(split_path, index=False)

    cleaned_data_path = output_dir / "cleaned.parquet"
    _write_dataframe(df, cleaned_data_path)

    # Feature audit + baseline feature config for experiment notebooks.
    feature_audit = _build_feature_audit(df, target_col=target_col, seed=seed)
    feature_audit_path = output_dir / "feature_audit.csv"
    feature_audit.to_csv(feature_audit_path, index=False)

    baseline_feature_config = _build_baseline_feature_config(
        feature_audit,
        target_col=target_col,
    )
    baseline_feature_config_snapshot_path = output_dir / "baseline_feature_config.json"
    baseline_feature_config_snapshot_path.write_text(
        json.dumps(
            {
                "created_utc": _utc_now_iso(),
                **baseline_feature_config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Write/overwrite canonical baseline config if requested.
    canonical_baseline_feature_config_path = (
        baseline_feature_config_path
        if baseline_feature_config_path is not None
        else baseline_feature_config_snapshot_path
    )
    if baseline_feature_config_path is not None:
        baseline_feature_config_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_feature_config_path.write_text(
            baseline_feature_config_snapshot_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    manifest = {
        "created_utc": _utc_now_iso(),
        "raw_csv_path": str(raw_csv_path),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "target_col": target_col,
        "seed": seed,
        "fractions": {"train": train_size, "val": val_size, "test": test_size},
        "artifacts": {
            "cleaned_data": str(cleaned_data_path),
            "split": str(split_path),
            "class_map": str(class_map_path),
            "feature_audit": str(feature_audit_path),
            "baseline_feature_config": str(canonical_baseline_feature_config_path),
            "baseline_feature_config_snapshot": str(
                baseline_feature_config_snapshot_path
            ),
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return PreparedDatasetArtifacts(
        output_dir=str(output_dir),
        cleaned_data_path=str(cleaned_data_path),
        split_path=str(split_path),
        class_map_path=str(class_map_path),
        feature_audit_path=str(feature_audit_path),
        baseline_feature_config_path=str(canonical_baseline_feature_config_path),
        manifest_path=str(manifest_path),
    )
