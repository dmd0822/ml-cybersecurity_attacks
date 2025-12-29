"""Shared utilities for Attack Type training pipelines.

These helpers load the latest prepared dataset from `data/02-preprocessed/`,
apply the baseline feature config (timestamp/port feature engineering + drops),
build a consistent encoder, and write run artifacts.

Artifacts written by training entrypoints always include `attack_type` in the
filename for traceability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.pipelines.common.paths import RepoPaths
from src.pipelines.features.baseline_feature_config import (
    BaselineFeatureConfig,
    apply_baseline_feature_config,
    load_baseline_feature_config,
)


@dataclass(frozen=True)
class PreparedDatasetPaths:
    """Resolved locations of prepared dataset artifacts."""

    output_dir: Path
    cleaned_data_path: Path
    split_path: Path
    class_map_path: Path
    manifest_path: Path


def _parse_manifest_created_utc(payload: dict[str, Any]) -> datetime | None:
    """Parse `created_utc` from a manifest payload."""

    created = payload.get("created_utc")
    if not created:
        return None

    try:
        # Example: 2025-12-29T13:26:19+00:00
        return datetime.fromisoformat(str(created))
    except Exception:
        return None


def find_latest_prepared_dataset(paths: RepoPaths) -> PreparedDatasetPaths:
    """Find the latest prepared dataset folder under `data/02-preprocessed`.

    Selection is based on `manifest.json`'s `created_utc` when present, otherwise
    the manifest file's mtime.

    Parameters:
        paths: Repository paths.

    Returns:
        PreparedDatasetPaths: Resolved artifact paths.

    Raises:
        FileNotFoundError: If no prepared dataset manifests are found.
    """

    manifests = sorted(paths.preprocessed_dir.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            "No prepared dataset manifests found under data/02-preprocessed. "
            "Run entrypoints/prepare_dataset.py first."
        )

    best_manifest: Path | None = None
    best_created: datetime | None = None

    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

        created = _parse_manifest_created_utc(payload)
        if created is None:
            created = datetime.fromtimestamp(manifest_path.stat().st_mtime)

        if best_created is None or created > best_created:
            best_created = created
            best_manifest = manifest_path

    if best_manifest is None:
        raise FileNotFoundError("No valid prepared dataset manifest found.")

    output_dir = best_manifest.parent

    cleaned_parquet = output_dir / "cleaned.parquet"
    cleaned_csv = output_dir / "cleaned.csv"
    cleaned_data_path = cleaned_parquet if cleaned_parquet.exists() else cleaned_csv

    split_path = output_dir / "split.csv"
    class_map_path = output_dir / "attack_type_classes.json"

    return PreparedDatasetPaths(
        output_dir=output_dir,
        cleaned_data_path=cleaned_data_path,
        split_path=split_path,
        class_map_path=class_map_path,
        manifest_path=best_manifest,
    )


def load_cleaned_dataset(path: Path) -> pd.DataFrame:
    """Load cleaned dataset from parquet or CSV."""

    if path.suffix.lower() == ".parquet" and path.exists():
        return pd.read_parquet(path)

    return pd.read_csv(path)


def load_class_map(path: Path) -> dict[str, Any]:
    """Load the class map JSON (Attack Type classes)."""

    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_columns_exist(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    existing = set(df.columns)
    return [c for c in cols if c in existing]


def make_splits(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    *,
    config: BaselineFeatureConfig,
) -> dict[str, pd.DataFrame]:
    """Join the shared split onto the cleaned dataset.

    Parameters:
        df: Cleaned dataset dataframe.
        split_df: Split dataframe with `row_id`, target, and `split`.
        config: Baseline feature config.

    Returns:
        Dict of split name -> dataframe.

    Raises:
        ValueError: If the split join fails or labels mismatch.
    """

    row_id_col = config.row_id_col

    required_cols = _ensure_columns_exist(split_df, [row_id_col, "split"])
    if len(required_cols) < 2:
        raise ValueError("split.csv must contain row_id and split columns")

    # Avoid duplicate target columns during merge.
    split_join = split_df[[row_id_col, "split", config.target_col]].copy()

    merged = df.merge(split_join, on=row_id_col, how="inner", suffixes=("", "_split"))

    if merged.empty:
        raise ValueError("Split join produced 0 rows; check row_id alignment.")

    target_split_col = f"{config.target_col}_split"
    if target_split_col in merged.columns:
        mismatch = (merged[config.target_col].astype(str) != merged[target_split_col].astype(str)).sum()
        if mismatch:
            raise ValueError(
                f"Target label mismatch between cleaned dataset and split.csv for {mismatch} rows."
            )
        merged = merged.drop(columns=[target_split_col])

    out: dict[str, pd.DataFrame] = {}
    for name in ["train", "val", "test"]:
        out[name] = merged.loc[merged["split"] == name].copy()

    return out


def _infer_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical columns for preprocessing."""

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X.columns:
        dtype = X[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a baseline encoder compatible with the experiment notebooks."""

    numeric_cols, categorical_cols = _infer_feature_columns(X)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def encode_labels(y: pd.Series, *, classes: list[str]) -> pd.Series:
    """Encode string labels to integer indices using the provided class order."""

    mapping = {name: idx for idx, name in enumerate(classes)}
    return y.astype(str).map(mapping)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Compute core classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def write_run_artifacts(
    *,
    run_dir: Path,
    model_artifact_prefix: str,
    model: Any,
    metrics: dict[str, Any],
    test_predictions: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Write artifacts for a training run.

    Parameters:
        run_dir: Directory to write artifacts into.
        model_artifact_prefix: Prefix used for all filenames (must include
            `attack_type`).
        model: Trained model/pipeline.
        metrics: Metrics dictionary.
        test_predictions: Dataframe with row_id/y_true/y_pred (and optionally
            probabilities).
        manifest: Manifest dictionary.

    Returns:
        Dict of artifact name -> path string.
    """

    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / f"{model_artifact_prefix}_model.joblib"
    metrics_path = run_dir / f"{model_artifact_prefix}_metrics.json"
    preds_path = run_dir / f"{model_artifact_prefix}_test_predictions.csv"
    manifest_path = run_dir / f"{model_artifact_prefix}_manifest.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test_predictions.to_csv(preds_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "test_predictions": str(preds_path),
        "manifest": str(manifest_path),
    }
