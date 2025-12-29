"""Train an XGBoost model for the `Attack Type` target.

This training pipeline mirrors the Random Forest pipeline but swaps the
estimator for XGBoost.

All output filenames include `attack_type`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.pipelines.common.paths import get_repo_paths
from src.pipelines.features.baseline_feature_config import load_baseline_feature_config

from src.pipelines.attack_type.training_utils import (
    build_preprocessor,
    compute_metrics,
    encode_labels,
    find_latest_prepared_dataset,
    load_class_map,
    load_cleaned_dataset,
    make_splits,
    write_run_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Train XGBoost for Attack Type.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=600,
        help="Number of boosting rounds.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Tree max depth.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate.",
    )
    return parser.parse_args()


def train_xgboost(
    *,
    repo_root: Path,
    seed: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Train and write artifacts."""

    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "xgboost is not installed. Install it with: pip install xgboost"
        ) from exc

    paths = get_repo_paths(repo_root)
    prepared = find_latest_prepared_dataset(paths)

    baseline_cfg_path = paths.config_dir / "baseline_feature_config.json"
    baseline_cfg = load_baseline_feature_config(baseline_cfg_path)

    df = load_cleaned_dataset(prepared.cleaned_data_path)
    split_df = pd.read_csv(prepared.split_path)

    split_frames = make_splits(df, split_df, config=baseline_cfg)

    class_map = load_class_map(prepared.class_map_path)
    classes = [str(c) for c in class_map.get("classes", [])]

    from src.pipelines.features.baseline_feature_config import apply_baseline_feature_config

    train_df = split_frames["train"]
    val_df = split_frames["val"]
    test_df = split_frames["test"]

    X_train = apply_baseline_feature_config(train_df, baseline_cfg)
    y_train = encode_labels(train_df[baseline_cfg.target_col], classes=classes)

    X_val = apply_baseline_feature_config(val_df, baseline_cfg)
    y_val = encode_labels(val_df[baseline_cfg.target_col], classes=classes)

    X_test = apply_baseline_feature_config(test_df, baseline_cfg)
    y_test = encode_labels(test_df[baseline_cfg.target_col], classes=classes)

    preprocessor = build_preprocessor(X_train)

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(classes),
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    model = Pipeline(steps=[("pre", preprocessor), ("model", clf)])
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    metrics = {
        "created_utc": _utc_now_iso(),
        "model": "xgboost",
        "params": {
            "seed": seed,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        },
        "val": compute_metrics(y_val, val_pred),
        "test": compute_metrics(y_test, test_pred),
    }

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = paths.data_dir / "04-predictions" / "attack_type" / "training" / run_stamp

    prefix = "attack_type_xgboost"

    test_predictions = pd.DataFrame(
        {
            baseline_cfg.row_id_col: test_df[baseline_cfg.row_id_col].astype(str),
            "y_true": y_test.astype(int),
            "y_pred": pd.Series(test_pred).astype(int),
        }
    )

    manifest = {
        "created_utc": _utc_now_iso(),
        "task": "attack_type",
        "model": "xgboost",
        "prepared_dataset_dir": str(prepared.output_dir),
        "prepared_manifest": str(prepared.manifest_path),
        "baseline_feature_config": str(baseline_cfg_path),
        "classes": classes,
        "run_dir": str(run_dir),
    }

    artifacts = write_run_artifacts(
        run_dir=run_dir,
        model_artifact_prefix=prefix,
        model=model,
        metrics=metrics,
        test_predictions=test_predictions,
        manifest=manifest,
    )

    return {
        "artifacts": artifacts,
        "test_macro_f1": metrics["test"]["macro_f1"],
        "test_weighted_f1": metrics["test"]["weighted_f1"],
    }


def main() -> int:
    """CLI entry point."""

    sys.path.insert(0, str(REPO_ROOT))

    args = parse_args()

    result = train_xgboost(
        repo_root=REPO_ROOT,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    print("Wrote artifacts:")
    for k, v in result["artifacts"].items():
        print(f"  {k}: {v}")
    print(
        "Test metrics: "
        f"macro_f1={result['test_macro_f1']:.4f} "
        f"weighted_f1={result['test_weighted_f1']:.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
