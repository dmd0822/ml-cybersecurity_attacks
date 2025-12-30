"""Create a shared time-aware train/val/test split artifact.

This entrypoint reads a prepared dataset from `data/02-preprocessed/<name>/`,
creates a chronological split based on `Timestamp`, and writes a split CSV
alongside the prepared dataset.

The split is keyed by `row_id` and keeps identical timestamps together.

Run:

    python entrypoints/create_time_split.py \
        --preprocessed-name cybersecurity_attacks_v1_2025-12-29

By default it writes `split_time_70_15_15.csv` and updates `manifest.json` to
reference the new artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.pipelines.common.paths import get_repo_paths
from src.pipelines.features.time_split import make_time_aware_split


DEFAULT_PREPROCESSED_NAME = "cybersecurity_attacks_v1_2025-12-29"


def _utc_now_iso() -> str:
    """Return current UTC time as ISO string."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Create a time-aware split CSV.")
    parser.add_argument(
        "--preprocessed-name",
        default=DEFAULT_PREPROCESSED_NAME,
        help="Folder name under data/02-preprocessed/ to read.",
    )
    parser.add_argument(
        "--time-col",
        default="Timestamp",
        help="Time column used for chronological splitting.",
    )
    parser.add_argument(
        "--row-id-col",
        default="row_id",
        help="Row id column name.",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train fraction.")
    parser.add_argument("--val", type=float, default=0.15, help="Val fraction.")
    parser.add_argument("--test", type=float, default=0.15, help="Test fraction.")
    parser.add_argument(
        "--output-file",
        default="split_time_70_15_15.csv",
        help="Output split filename inside the prepared dataset folder.",
    )
    return parser.parse_args()


def _read_prepared_dataset(prepared_dir: Path) -> pd.DataFrame:
    """Read cleaned dataset from a prepared folder (parquet preferred)."""

    parquet_path = prepared_dir / "cleaned.parquet"
    csv_path = prepared_dir / "cleaned.csv"

    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            # Fall back to CSV if parquet dependencies are missing.
            pass

    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"No cleaned dataset found in {prepared_dir} (expected cleaned.parquet or cleaned.csv)"
    )


def _update_manifest(
    *,
    prepared_dir: Path,
    split_path: Path,
    time_col: str,
    row_id_col: str,
    train: float,
    val: float,
    test: float,
    report: object,
) -> None:
    """Update manifest.json (if present) to reference the time split artifact."""

    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    artifacts = manifest.get("artifacts", {})
    artifacts["split_time"] = str(split_path)
    manifest["artifacts"] = artifacts

    manifest["time_split"] = {
        "generated_utc": _utc_now_iso(),
        "time_col": time_col,
        "row_id_col": row_id_col,
        "strategy": "chronological_timestamp_grouped",
        "fractions": {"train": train, "val": val, "test": test},
        "report": getattr(report, "__dict__", {}),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    """Entry point."""

    args = parse_args()

    paths = get_repo_paths(REPO_ROOT)
    prepared_dir = paths.preprocessed_dir / args.preprocessed_name

    df = _read_prepared_dataset(prepared_dir)

    split_df, report = make_time_aware_split(
        df,
        time_col=args.time_col,
        row_id_col=args.row_id_col,
        train_size=args.train,
        val_size=args.val,
        test_size=args.test,
    )

    split_path = prepared_dir / args.output_file
    split_df.to_csv(split_path, index=False)

    _update_manifest(
        prepared_dir=prepared_dir,
        split_path=split_path,
        time_col=args.time_col,
        row_id_col=args.row_id_col,
        train=args.train,
        val=args.val,
        test=args.test,
        report=report,
    )

    print(f"Wrote time-aware split: {split_path}")
    print(f"Rows: train={report.train_rows}, val={report.val_rows}, test={report.test_rows}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
