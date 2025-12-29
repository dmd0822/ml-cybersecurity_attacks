"""Train an XGBoost model for Attack Type.

Run:

    python entrypoints/train_attack_type_xgboost.py

Outputs are written under `data/04-predictions/attack_type/training/` with
filenames prefixed by `attack_type_xgboost_*`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.attack_type.train_xgboost import main


if __name__ == "__main__":
    raise SystemExit(main())
