"""Shared utilities for Attack Type inference pipelines.

Inference pipelines:
- load a previously saved sklearn Pipeline (`*.joblib`)
- load the latest prepared dataset under `data/02-preprocessed/`
- apply the baseline feature config (timestamp/port feature engineering)
- run predictions and write artifacts under `data/04-predictions/`

All output filenames include `attack_type` for traceability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.pipelines.attack_type.training_utils import (
    PreparedDatasetPaths,
    find_latest_prepared_dataset,
    load_class_map,
    load_cleaned_dataset,
    make_splits,
)
from src.pipelines.common.paths import RepoPaths
from src.pipelines.features.baseline_feature_config import (
    BaselineFeatureConfig,
    apply_baseline_feature_config,
    load_baseline_feature_config,
)


@dataclass(frozen=True)
class ModelArtifact:
    """Resolved model artifact file."""

    model_path: Path
    run_dir: Path


def _parse_run_stamp(run_dir_name: str) -> datetime | None:
    """Parse training run folder stamps like YYYYMMDD_HHMMSS."""

    try:
        return datetime.strptime(run_dir_name, "%Y%m%d_%H%M%S")
    except Exception:
        return None


def find_latest_trained_model(
    paths: RepoPaths,
    *,
    model_prefix: str,
) -> ModelArtifact:
    """Find the latest trained model artifact for a given prefix.

    Parameters:
        paths: Repository paths.
        model_prefix: Filename prefix, e.g. `attack_type_random_forest`.

    Returns:
        ModelArtifact: Resolved model path and run directory.

    Raises:
        FileNotFoundError: If no model artifacts are found.
    """

    training_root = paths.data_dir / "04-predictions" / "attack_type" / "training"
    candidates = sorted(training_root.glob(f"*/{model_prefix}_model.joblib"))

    if not candidates:
        raise FileNotFoundError(
            f"No trained model found for prefix '{model_prefix}'. "
            "Run the training pipeline first."
        )

    best: Path | None = None
    best_stamp: datetime | None = None

    for path in candidates:
        stamp = _parse_run_stamp(path.parent.name)
        if stamp is None:
            stamp = datetime.fromtimestamp(path.stat().st_mtime)

        if best_stamp is None or stamp > best_stamp:
            best_stamp = stamp
            best = path

    assert best is not None
    return ModelArtifact(model_path=best, run_dir=best.parent)


def load_model(path: Path) -> Any:
    """Load a joblib-serialized model pipeline."""

    try:
        return joblib.load(path)
    except AttributeError as exc:
        # Common when unpickling across incompatible scikit-learn versions.
        raise RuntimeError(
            "Failed to load model artifact. This is commonly caused by a "
            "scikit-learn version mismatch between training and inference. "
            f"Model path: {path}. "
            "Fix: re-train the model using the same Python environment used for "
            "inference (recommended), or align scikit-learn versions."
        ) from exc


def load_latest_inference_inputs(
    paths: RepoPaths,
) -> tuple[PreparedDatasetPaths, BaselineFeatureConfig, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the latest prepared dataset, split file, baseline config, and classes."""

    prepared = find_latest_prepared_dataset(paths)

    baseline_cfg_path = paths.config_dir / "baseline_feature_config.json"
    baseline_cfg = load_baseline_feature_config(baseline_cfg_path)

    df = load_cleaned_dataset(prepared.cleaned_data_path)
    split_df = pd.read_csv(prepared.split_path)

    class_map = load_class_map(prepared.class_map_path)
    classes = [str(c) for c in class_map.get("classes", [])]

    return prepared, baseline_cfg, df, split_df, classes


def run_inference_for_split(
    *,
    model: Any,
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    baseline_cfg: BaselineFeatureConfig,
    classes: list[str],
    split: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run inference for a particular split.

    Parameters:
        model: Trained sklearn Pipeline.
        df: Cleaned dataset dataframe.
        split_df: Split dataframe.
        baseline_cfg: Baseline feature config.
        classes: Class name list in index order.
        split: One of `train`, `val`, `test`.

    Returns:
        DataFrame with row_id, predicted class index and label.
    """

    split_frames = make_splits(df, split_df, config=baseline_cfg)
    frame = split_frames[split]
    if limit is not None:
        frame = frame.head(limit).copy()

    X = apply_baseline_feature_config(frame, baseline_cfg)

    pred_idx = model.predict(X)

    # Robust conversion for models that return numpy arrays.
    pred_series = pd.Series(pred_idx).astype(int)
    pred_label = pred_series.map(lambda i: classes[i] if 0 <= i < len(classes) else "")

    out = pd.DataFrame(
        {
            baseline_cfg.row_id_col: frame[baseline_cfg.row_id_col].astype(str),
            "y_pred": pred_series,
            "y_pred_label": pred_label,
        }
    )

    # Optional probability max.
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            out["pred_proba_max"] = pd.Series(proba.max(axis=1)).astype(float)
        except Exception:
            pass

    # Include y_true if available (useful for evaluation when labels exist).
    if baseline_cfg.target_col in frame.columns:
        out["y_true_label"] = frame[baseline_cfg.target_col].astype(str)

    return out


def write_inference_artifacts(
    *,
    run_dir: Path,
    artifact_prefix: str,
    predictions: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Write inference predictions + manifest.

    Parameters:
        run_dir: Directory to write into.
        artifact_prefix: Filename prefix (must include `attack_type`).
        predictions: Predictions dataframe.
        manifest: Inference manifest.

    Returns:
        Dict of artifact name -> path.
    """

    run_dir.mkdir(parents=True, exist_ok=True)

    preds_path = run_dir / f"{artifact_prefix}_predictions.csv"
    manifest_path = run_dir / f"{artifact_prefix}_manifest.json"

    predictions.to_csv(preds_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "predictions": str(preds_path),
        "manifest": str(manifest_path),
    }
