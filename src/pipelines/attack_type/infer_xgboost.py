"""Inference pipeline for the Attack Type XGBoost model.

By default this:
- Loads the latest saved XGBoost model artifact
- Loads the latest prepared dataset
- Runs inference on the `test` split
- Writes predictions under `data/04-predictions/attack_type/inference/`

All filenames include `attack_type`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipelines.common.paths import get_repo_paths

from src.pipelines.attack_type.inference_utils import (
    find_latest_trained_model,
    load_latest_inference_inputs,
    load_model,
    run_inference_for_split,
    write_inference_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(
        description="Run inference for Attack Type using the XGBoost model."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to score.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to a specific attack_type_xgboost_model.joblib.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick smoke runs.",
    )
    return parser.parse_args()


def run_inference(
    *,
    repo_root: Path,
    split: str,
    model_path: str | None,
    limit: int | None,
) -> dict[str, Any]:
    """Run inference and write artifacts."""

    paths = get_repo_paths(repo_root)

    model_prefix = "attack_type_xgboost"

    if model_path:
        model_file = Path(model_path)
        run_dir = model_file.parent
    else:
        found = find_latest_trained_model(paths, model_prefix=model_prefix)
        model_file = found.model_path
        run_dir = found.run_dir

    model = load_model(model_file)

    prepared, baseline_cfg, df, split_df, classes = load_latest_inference_inputs(paths)

    predictions = run_inference_for_split(
        model=model,
        df=df,
        split_df=split_df,
        baseline_cfg=baseline_cfg,
        classes=classes,
        split=split,
        limit=limit,
    )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        paths.data_dir
        / "04-predictions"
        / "attack_type"
        / "inference"
        / run_stamp
    )

    artifact_prefix = f"attack_type_xgboost_{split}"

    manifest = {
        "created_utc": _utc_now_iso(),
        "task": "attack_type",
        "model": "xgboost",
        "split": split,
        "model_artifact": str(model_file),
        "model_run_dir": str(run_dir),
        "prepared_dataset_dir": str(prepared.output_dir),
        "prepared_manifest": str(prepared.manifest_path),
        "outputs_dir": str(out_dir),
    }

    artifacts = write_inference_artifacts(
        run_dir=out_dir,
        artifact_prefix=artifact_prefix,
        predictions=predictions,
        manifest=manifest,
    )

    return {"artifacts": artifacts, "rows": int(predictions.shape[0])}


def main() -> int:
    """CLI entry point."""

    sys.path.insert(0, str(REPO_ROOT))

    args = parse_args()

    result = run_inference(
        repo_root=REPO_ROOT,
        split=args.split,
        model_path=args.model_path,
        limit=args.limit,
    )

    print("Wrote inference artifacts:")
    for k, v in result["artifacts"].items():
        print(f"  {k}: {v}")
    print(f"Rows scored: {result['rows']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
