"""Unit tests for Attack Type training utilities.

These tests avoid relying on the real dataset on disk by using tmp directories
and tiny synthetic dataframes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipelines.attack_type.training_utils import (
    build_preprocessor,
    find_latest_prepared_dataset,
    make_splits,
)
from src.pipelines.common.paths import RepoPaths
from src.pipelines.features.baseline_feature_config import BaselineFeatureConfig


def _write_manifest(path: Path, *, created_utc: str) -> None:
    """Write a minimal manifest.json used by `find_latest_prepared_dataset`."""

    payload = {
        "created_utc": created_utc,
        "artifacts": {
            "cleaned_data": str(path.parent / "cleaned.parquet"),
            "split": str(path.parent / "split.csv"),
            "class_map": str(path.parent / "attack_type_classes.json"),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_find_latest_prepared_dataset_prefers_newer_manifest(tmp_path: Path) -> None:
    """Select the most recent prepared dataset based on manifest created_utc."""

    repo_root = tmp_path
    preprocessed_dir = repo_root / "data" / "02-preprocessed"
    preprocessed_dir.mkdir(parents=True)

    older = preprocessed_dir / "older"
    newer = preprocessed_dir / "newer"
    older.mkdir()
    newer.mkdir()

    _write_manifest(older / "manifest.json", created_utc="2025-01-01T00:00:00+00:00")
    _write_manifest(newer / "manifest.json", created_utc="2025-01-02T00:00:00+00:00")

    paths = RepoPaths(
        repo_root=repo_root,
        config_dir=repo_root / "config",
        data_dir=repo_root / "data",
        raw_dir=repo_root / "data" / "01-raw",
        preprocessed_dir=preprocessed_dir,
    )

    resolved = find_latest_prepared_dataset(paths)
    assert resolved.output_dir.name == "newer"
    assert resolved.manifest_path.name == "manifest.json"


def test_make_splits_joins_and_partitions(tmp_path: Path) -> None:
    """Join split.csv to cleaned dataset using row_id and partition by split."""

    df = pd.DataFrame(
        {
            "row_id": ["row_000000", "row_000001", "row_000002"],
            "Attack Type": ["DDoS", "Intrusion", "Malware"],
            "Feature A": [1.0, 2.0, 3.0],
        }
    )

    split_df = pd.DataFrame(
        {
            "row_id": ["row_000000", "row_000001", "row_000002"],
            "Attack Type": ["DDoS", "Intrusion", "Malware"],
            "split": ["train", "val", "test"],
        }
    )

    cfg = BaselineFeatureConfig(
        target_col="Attack Type",
        row_id_col="row_id",
        drop_cols=[],
        timestamp_cols=[],
        port_cols=[],
    )

    splits = make_splits(df, split_df, config=cfg)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1


def test_build_preprocessor_fit_transform() -> None:
    """Preprocessor should fit/transform mixed numeric + categorical inputs."""

    X = pd.DataFrame(
        {
            "num_feature": [1.0, 2.5, None],
            "cat_feature": ["a", "b", None],
        }
    )

    pre = build_preprocessor(X)

    Xt = pre.fit_transform(X)

    # Should produce a 2D matrix with 3 rows.
    assert Xt.shape[0] == 3


def test_make_splits_raises_on_label_mismatch() -> None:
    """Mismatch between cleaned labels and split labels should raise."""

    df = pd.DataFrame(
        {
            "row_id": ["row_000000"],
            "Attack Type": ["DDoS"],
            "Feature": [1],
        }
    )

    split_df = pd.DataFrame(
        {
            "row_id": ["row_000000"],
            "Attack Type": ["Intrusion"],
            "split": ["train"],
        }
    )

    cfg = BaselineFeatureConfig(
        target_col="Attack Type",
        row_id_col="row_id",
        drop_cols=[],
        timestamp_cols=[],
        port_cols=[],
    )

    with pytest.raises(ValueError, match="Target label mismatch"):
        make_splits(df, split_df, config=cfg)
