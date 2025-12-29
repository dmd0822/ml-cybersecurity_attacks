"""Prepare the Kaggle cyber-security-attacks dataset.

This script reads the raw snapshot in `data/01-raw/`, cleans the data, and
writes the cleaned dataset + a shared stratified split into `data/02-preprocessed/`.

Run:

    python entrypoints/prepare_dataset.py

You can override defaults (snapshot date, output name, etc.) via CLI args.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.common.paths import get_repo_paths
from src.pipelines.features.prepare_dataset import prepare_dataset


DEFAULT_DATASET_DIR = "teamincribo_cyber-security-attacks"
DEFAULT_SNAPSHOT_DATE = "2025-12-29"
DEFAULT_RAW_FILENAME = "cybersecurity_attacks.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Clean dataset and create shared stratified train/val/test split."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Folder name under data/01-raw for the dataset.",
    )
    parser.add_argument(
        "--snapshot-date",
        default=DEFAULT_SNAPSHOT_DATE,
        help="Snapshot date folder name under data/01-raw/<dataset-dir>/.",
    )
    parser.add_argument(
        "--raw-file",
        default=DEFAULT_RAW_FILENAME,
        help="Raw CSV filename inside the snapshot folder.",
    )
    parser.add_argument(
        "--output-name",
        default=f"cybersecurity_attacks_v1_{datetime.now().strftime('%Y-%m-%d')}",
        help="Output folder name under data/02-preprocessed/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the shared split.",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="Train fraction.",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Validation fraction.",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.15,
        help="Test fraction.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""

    paths = get_repo_paths(REPO_ROOT)

    args = parse_args()

    raw_csv_path = paths.raw_dir / args.dataset_dir / args.snapshot_date / args.raw_file

    output_dir = paths.preprocessed_dir / args.output_name

    canonical_baseline_path = paths.config_dir / "baseline_feature_config.json"

    artifacts = prepare_dataset(
        raw_csv_path=raw_csv_path,
        output_dir=output_dir,
        baseline_feature_config_path=canonical_baseline_path,
        seed=args.seed,
        train_size=args.train,
        val_size=args.val,
        test_size=args.test,
    )

    print("Prepared dataset artifacts:")
    print(f"  cleaned:  {artifacts.cleaned_data_path}")
    print(f"  split:    {artifacts.split_path}")
    print(f"  classes:  {artifacts.class_map_path}")
    print(f"  audit:    {artifacts.feature_audit_path}")
    print(f"  baseline: {artifacts.baseline_feature_config_path}")
    print(f"  manifest: {artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
