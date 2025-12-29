"""Run inference using the latest trained Attack Type XGBoost model."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.attack_type.infer_xgboost import main


if __name__ == "__main__":
    raise SystemExit(main())
