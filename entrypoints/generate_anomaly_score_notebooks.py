"""Generate anomaly score regression experiment notebooks.

This scaffolds one notebook per regression model under:
- `notebooks/anomaly_scores/`

Each notebook:
- Loads the latest prepared dataset from `data/02-preprocessed/`
- Loads the shared time-aware split (`split_time_70_15_15.csv`)
- Loads the canonical baseline feature config (`config/baseline_feature_config.json`)
- Applies baseline drop/transform logic (timestamp + port features)
- Builds a ColumnTransformer with OneHotEncoder for categoricals
- Trains/evaluates a regression model for `Anomaly Scores`
- Renders charts using `config/visualization.json` `color_palette` (via `COLOR_PALLETE`)

Run:
    python entrypoints/generate_anomaly_score_notebooks.py

Notes:
- Notebooks are intended as starting points for experimentation.
- The baseline feature config target is `Attack Type`; notebooks override the
  target to `Anomaly Scores` and drop `Attack Type` from features to reduce leakage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "nbformat is required to generate notebooks. Install with: pip install nbformat"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks" / "anomaly_scores"


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a generated notebook."""

    slug: str
    title: str
    needs_scaling: bool
    model_code: str
    extra_notes: str = ""


def _slugify(text: str) -> str:
    """Convert a model name to a safe filename slug."""

    s = text.strip().lower()
    s = re.sub(r"\([^)]*\)", "", s)
    s = s.replace("+", "plus")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _now_suffix() -> str:
    """Return a filesystem-safe timestamp string."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _common_setup_cell() -> str:
    return (
        "from __future__ import annotations\n\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n\n"
        "SEED = 42\n"
        "np.random.seed(SEED)\n\n"
        "# Resolve repo root robustly so this works from notebooks/ or notebooks/anomaly_scores/.\n"
        "cwd = Path.cwd().resolve()\n"
        "candidates = [cwd] + list(cwd.parents)\n"
        "REPO_ROOT = next((p for p in candidates if (p / 'src').exists() and (p / 'data').exists()), None)\n"
        "if REPO_ROOT is None:\n"
        "    raise FileNotFoundError('Could not locate repo root (expected src/ and data/).')\n"
        "sys.path.insert(0, str(REPO_ROOT))\n\n"
        "PREPROCESSED_ROOT = REPO_ROOT / 'data' / '02-preprocessed'\n"
        "BASELINE_CONFIG_JSON = REPO_ROOT / 'config' / 'baseline_feature_config.json'\n"
        "VIZ_CONFIG_JSON = REPO_ROOT / 'config' / 'visualization.json'\n\n"
        "prepared_dirs = sorted([p for p in PREPROCESSED_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)\n"
        "if not prepared_dirs:\n"
        "    raise FileNotFoundError(f'No prepared datasets found under: {PREPROCESSED_ROOT}')\n"
        "DATASET_DIR = prepared_dirs[-1]\n\n"
        "cleaned_parquet = DATASET_DIR / 'cleaned.parquet'\n"
        "cleaned_csv = DATASET_DIR / 'cleaned.csv'\n"
        "split_time_csv = DATASET_DIR / 'split_time_70_15_15.csv'\n\n"
        "print(f'Using prepared dataset: {DATASET_DIR}')\n"
        "print(f'Using baseline config:  {BASELINE_CONFIG_JSON}')\n"
        "print(f'Using time split:      {split_time_csv.name}')\n\n"
        "if cleaned_parquet.exists():\n"
        "    df = pd.read_parquet(cleaned_parquet)\n"
        "elif cleaned_csv.exists():\n"
        "    df = pd.read_csv(cleaned_csv)\n"
        "else:\n"
        "    raise FileNotFoundError('Expected cleaned.parquet or cleaned.csv')\n\n"
        "if not split_time_csv.exists():\n"
        "    raise FileNotFoundError(f'Missing time-aware split artifact: {split_time_csv}')\n"
        "splits = pd.read_csv(split_time_csv)\n\n"
        "from src.pipelines.features import (\n"
        "    BaselineFeatureConfig,\n"
        "    apply_baseline_feature_config,\n"
        "    load_baseline_feature_config,\n"
        ")\n\n"
        "cfg0 = load_baseline_feature_config(BASELINE_CONFIG_JSON)\n"
        "# Override the target to Anomaly Scores (regression) and drop Attack Type to reduce leakage.\n"
        "cfg = BaselineFeatureConfig(\n"
        "    target_col='Anomaly Scores',\n"
        "    row_id_col=cfg0.row_id_col,\n"
        "    drop_cols=sorted(set(cfg0.drop_cols + ['Attack Type'])),\n"
        "    timestamp_cols=cfg0.timestamp_cols,\n"
        "    port_cols=cfg0.port_cols,\n"
        ")\n\n"
        "required_cols = {cfg.row_id_col, cfg.target_col, 'Timestamp'}\n"
        "missing_required = required_cols - set(df.columns)\n"
        "if missing_required:\n"
        "    raise KeyError(f'Missing required columns in cleaned data: {sorted(missing_required)}')\n\n"
        "# Features and regression target\n"
        "X_full = apply_baseline_feature_config(df, cfg)\n"
        "y_full = pd.to_numeric(df[cfg.target_col], errors='coerce')\n\n"
        "# Split join by row_id\n"
        "df_split = df[[cfg.row_id_col]].merge(splits[[cfg.row_id_col, 'split']], on=cfg.row_id_col, how='left')\n"
        "if df_split['split'].isna().any():\n"
        "    raise ValueError('Some rows are missing split assignments (time split join failed)')\n\n"
        "mask_train = df_split['split'].eq('train')\n"
        "mask_val = df_split['split'].eq('val')\n"
        "mask_test = df_split['split'].eq('test')\n\n"
        "X_train, y_train = X_full.loc[mask_train].reset_index(drop=True), y_full.loc[mask_train].reset_index(drop=True)\n"
        "X_val, y_val = X_full.loc[mask_val].reset_index(drop=True), y_full.loc[mask_val].reset_index(drop=True)\n"
        "X_test, y_test = X_full.loc[mask_test].reset_index(drop=True), y_full.loc[mask_test].reset_index(drop=True)\n\n"
        "# Keep timestamps for test rows for time-based charts\n"
        "ts_full = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)\n"
        "ts_test = ts_full.loc[mask_test].reset_index(drop=True)\n\n"
        "print('Split sizes:', X_train.shape, X_val.shape, X_test.shape)\n"
        "print('Target summary (train):', y_train.describe())\n"
    )


def _preprocess_cell(needs_scaling: bool) -> str:
    scale_line = "('scaler', StandardScaler())" if needs_scaling else "# ('scaler', StandardScaler())"

    return (
        "# Build preprocessing: impute + one-hot for categoricals; impute (+ optional scale) for numeric\n\n"
        "cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' or str(X_train[c].dtype).startswith('string')]\n"
        "num_cols = [c for c in X_train.columns if c not in cat_cols]\n\n"
        "cat_pipe = Pipeline(steps=[\n"
        "    ('imputer', SimpleImputer(strategy='most_frequent')),\n"
        "    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),\n"
        "])\n\n"
        "num_steps = [\n"
        "    ('imputer', SimpleImputer(strategy='median')),\n"
        f"    {scale_line},\n"
        "]\n"
        "num_steps = [s for s in num_steps if not (isinstance(s, str) or s[0].startswith('#'))]\n"
        "num_pipe = Pipeline(steps=num_steps)\n\n"
        "preprocess = ColumnTransformer(\n"
        "    transformers=[('cat', cat_pipe, cat_cols), ('num', num_pipe, num_cols)],\n"
        "    remainder='drop',\n"
        ")\n\n"
        "print(f'Categorical cols: {len(cat_cols)}')\n"
        "print(f'Numeric cols:     {len(num_cols)}')\n"
    )


def _regression_eval_cell() -> str:
    return (
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
        "import numpy as np\n\n"
        "def eval_split(name: str, y_true, y_pred) -> dict[str, float]:\n"
        "    mae = mean_absolute_error(y_true, y_pred)\n"
        "    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))\n"
        "    r2 = r2_score(y_true, y_pred)\n"
        "    print(f'\\n== {name} ==')\n"
        "    print(f'MAE:  {mae:.4f}')\n"
        "    print(f'RMSE: {rmse:.4f}')\n"
        "    print(f'R2:   {r2:.4f}')\n"
        "    return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}\n\n"
        "val_pred = reg.predict(X_val)\n"
        "test_pred = reg.predict(X_test)\n\n"
        "val_metrics = eval_split('val', y_val, val_pred)\n"
        "test_metrics = eval_split('test', y_test, test_pred)\n"
    )


def _charts_cell() -> str:
    return (
        "# Charts (uses config/visualization.json color_palette via COLOR_PALLETE)\n\n"
        "import json\n"
        "from matplotlib import pyplot as plt\n"
        "from cycler import cycler\n\n"
        "def load_color_palette(path: Path) -> list[str]:\n"
        "    with open(path, 'r', encoding='utf-8') as f:\n"
        "        payload = json.load(f)\n"
        "    palette = payload.get('visualization', {}).get('color_palette', [])\n"
        "    if not isinstance(palette, list) or not palette:\n"
        "        raise ValueError(f\"Missing/invalid visualization.color_palette in {path}\")\n"
        "    return [str(c) for c in palette]\n\n"
        "COLOR_PALLETE = load_color_palette(VIZ_CONFIG_JSON)\n"
        "plt.rcParams['axes.prop_cycle'] = cycler(color=COLOR_PALLETE)\n"
        "print(f'Loaded COLOR_PALLETE ({len(COLOR_PALLETE)} colors) from {VIZ_CONFIG_JSON.name}')\n\n"
        "# 1) Target distribution (train)\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "ax.hist(y_train, bins=50, color=COLOR_PALLETE[0], alpha=0.9)\n"
        "ax.set_title('Anomaly Scores distribution (train)')\n"
        "ax.set_xlabel('Anomaly Scores')\n"
        "ax.set_ylabel('Count')\n"
        "ax.grid(alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "# 2) Predicted vs actual (test)\n"
        "fig, ax = plt.subplots(figsize=(6, 6))\n"
        "ax.scatter(y_test, test_pred, s=12, alpha=0.6, color=COLOR_PALLETE[1])\n"
        "mn = float(min(y_test.min(), test_pred.min()))\n"
        "mx = float(max(y_test.max(), test_pred.max()))\n"
        "ax.plot([mn, mx], [mn, mx], linestyle='--', color=COLOR_PALLETE[-1], linewidth=1)\n"
        "ax.set_title('Predicted vs actual (test)')\n"
        "ax.set_xlabel('Actual')\n"
        "ax.set_ylabel('Predicted')\n"
        "ax.grid(alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "# 3) Residual histogram (test)\n"
        "residuals = (y_test - test_pred)\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "ax.hist(residuals, bins=50, color=COLOR_PALLETE[2], alpha=0.9)\n"
        "ax.axvline(0.0, color=COLOR_PALLETE[-1], linestyle='--', linewidth=1)\n"
        "ax.set_title('Residuals (test): actual - predicted')\n"
        "ax.set_xlabel('Residual')\n"
        "ax.set_ylabel('Count')\n"
        "ax.grid(alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "# 4) Actual vs predicted over time (test; chronological)\n"
        "df_time = pd.DataFrame({'Timestamp': ts_test, 'actual': y_test, 'pred': test_pred}).dropna()\n"
        "df_time = df_time.sort_values('Timestamp').reset_index(drop=True)\n"
        "fig, ax = plt.subplots(figsize=(12, 4))\n"
        "ax.plot(df_time['Timestamp'], df_time['actual'], color=COLOR_PALLETE[3], linewidth=1, label='actual')\n"
        "ax.plot(df_time['Timestamp'], df_time['pred'], color=COLOR_PALLETE[4], linewidth=1, label='pred')\n"
        "ax.set_title('Anomaly Scores over time (test)')\n"
        "ax.set_xlabel('Time')\n"
        "ax.set_ylabel('Anomaly Scores')\n"
        "ax.legend()\n"
        "ax.grid(alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )


def _make_notebook(spec: ModelSpec) -> nbformat.NotebookNode:
    """Create a notebook for a given model spec."""

    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
    )

    title_md = (
        f"# Experiment: {spec.title}\n\n"
        "Task: **regression**\n\n"
        "Target: `Anomaly Scores`\n\n"
        "Shared split: `split_time_70_15_15.csv` (time-aware, timestamp-grouped)\n\n"
        f"{spec.extra_notes}".strip()
    )

    cells = [
        new_markdown_cell(title_md),
        new_code_cell(_common_setup_cell()),
        new_code_cell(_preprocess_cell(spec.needs_scaling)),
    ]

    fit_code = (
        "# Model definition + training\n\n"
        f"{spec.model_code}\n\n"
        "reg = Pipeline(steps=[('preprocess', preprocess), ('model', model)])\n"
        "reg.fit(X_train, y_train)\n"
    )

    cells.append(new_code_cell(fit_code))
    cells.append(new_code_cell(_regression_eval_cell()))
    cells.append(new_code_cell(_charts_cell()))

    nb["cells"] = cells
    return nb


def _model_specs() -> list[ModelSpec]:
    """Return the list of regression model notebooks to generate."""

    specs: list[ModelSpec] = []

    specs.append(
        ModelSpec(
            slug=_slugify("linear_regression"),
            title="Linear Regression",
            needs_scaling=True,
            model_code=(
                "from sklearn.linear_model import LinearRegression\n\n"
                "model = LinearRegression()"
            ),
        )
    )

    specs.append(
        ModelSpec(
            slug=_slugify("ridge_regression"),
            title="Ridge Regression",
            needs_scaling=True,
            model_code=(
                "from sklearn.linear_model import Ridge\n\n"
                "model = Ridge(alpha=1.0, random_state=SEED)"
            ),
        )
    )

    specs.append(
        ModelSpec(
            slug=_slugify("lasso_regression"),
            title="Lasso Regression",
            needs_scaling=True,
            model_code=(
                "from sklearn.linear_model import Lasso\n\n"
                "model = Lasso(alpha=0.001, random_state=SEED, max_iter=10000)"
            ),
        )
    )

    specs.append(
        ModelSpec(
            slug=_slugify("elasticnet_regression"),
            title="ElasticNet Regression",
            needs_scaling=True,
            model_code=(
                "from sklearn.linear_model import ElasticNet\n\n"
                "model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=SEED, max_iter=10000)"
            ),
        )
    )

    specs.append(
        ModelSpec(
            slug=_slugify("svr"),
            title="Support Vector Regression (SVR)",
            needs_scaling=True,
            model_code=(
                "from sklearn.svm import SVR\n\n"
                "model = SVR(C=10.0, epsilon=0.1, kernel='rbf')"
            ),
        )
    )

    specs.append(
        ModelSpec(
            slug=_slugify("knn_regression"),
            title="kNN Regression",
            needs_scaling=True,
            model_code=(
                "from sklearn.neighbors import KNeighborsRegressor\n\n"
                "model = KNeighborsRegressor(n_neighbors=15, weights='distance')"
            ),
        )
    )

    specs.append(
        ModelSpec(
            slug=_slugify("random_forest_regression"),
            title="Random Forest Regression",
            needs_scaling=False,
            model_code=(
                "from sklearn.ensemble import RandomForestRegressor\n\n"
                "model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)"
            ),
        )
    )

    specs.append(
        ModelSpec(
            slug=_slugify("gradient_boosting_regression"),
            title="Gradient Boosting Regression",
            needs_scaling=False,
            model_code=(
                "from sklearn.ensemble import GradientBoostingRegressor\n\n"
                "model = GradientBoostingRegressor(random_state=SEED)"
            ),
        )
    )

    specs.append(
        ModelSpec(
            slug=_slugify("xgboost_regression"),
            title="XGBoost Regression",
            needs_scaling=False,
            model_code=(
                "try:\n"
                "    from xgboost import XGBRegressor\n"
                "except Exception as exc:\n"
                "    raise SystemExit('xgboost is required for this notebook. Install with: pip install xgboost') from exc\n\n"
                "model = XGBRegressor(\n"
                "    n_estimators=500,\n"
                "    learning_rate=0.05,\n"
                "    max_depth=6,\n"
                "    subsample=0.8,\n"
                "    colsample_bytree=0.8,\n"
                "    random_state=SEED,\n"
                "    n_jobs=-1,\n"
                "    reg_alpha=0.0,\n"
                "    reg_lambda=1.0,\n"
                ")"
            ),
        )
    )

    return specs


def main() -> int:
    """Entry point."""

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = _now_suffix()

    count = 0
    for spec in _model_specs():
        nb = _make_notebook(spec)
        filename = f"experiment_{spec.slug}_{suffix}.ipynb"
        out_path = NOTEBOOKS_DIR / filename
        nbformat.write(nb, out_path)
        count += 1

    print(f"Wrote {count} notebooks to: {NOTEBOOKS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
