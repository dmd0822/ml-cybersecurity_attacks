"""Generate experiment notebooks for the cybersecurity attacks project.

This script scaffolds one notebook per model listed in `notes.txt` under:
"Models to Create Experiment Notebooks For".

Each notebook:
- Loads the latest prepared dataset from `data/02-preprocessed/`
- Loads the shared split (`split.csv`)
- Loads the canonical baseline feature config (`config/baseline_feature_config.json`)
- Applies baseline drop/transform logic (timestamp + port features)
- Builds a ColumnTransformer with OneHotEncoder for categoricals
- Trains/evaluates a model (or runs clustering + evaluates vs labels)

The resulting notebooks are written under `notebooks/attack_type/` with filenames:
`experiment_<model>_<datetime>.ipynb`

Run:
    python entrypoints/generate_experiment_notebooks.py

Notes:
- Notebooks are intended as starting points. Some models are mapped to
  classification equivalents (e.g., "Lasso Regression" -> L1-regularized
  LogisticRegression) given the multiclass target (Attack Type).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


try:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "nbformat is required to generate notebooks. Install with: pip install nbformat"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks" / "attack_type"


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a generated notebook."""

    slug: str
    title: str
    task: str  # "classification" | "clustering"
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
        "# Resolve repo root robustly so this works from notebooks/ or notebooks/attack_type/.\n"
        "cwd = Path.cwd().resolve()\n"
        "candidates = [cwd] + list(cwd.parents)\n"
        "REPO_ROOT = next((p for p in candidates if (p / 'src').exists() and (p / 'data').exists()), None)\n"
        "if REPO_ROOT is None:\n"
        "    raise FileNotFoundError('Could not locate repo root (expected src/ and data/).')\n"
        "sys.path.insert(0, str(REPO_ROOT))\n\n"
        "PREPROCESSED_ROOT = REPO_ROOT / 'data' / '02-preprocessed'\n"
        "BASELINE_CONFIG_JSON = REPO_ROOT / 'config' / 'baseline_feature_config.json'\n\n"
        "prepared_dirs = sorted([p for p in PREPROCESSED_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)\n"
        "if not prepared_dirs:\n"
        "    raise FileNotFoundError(f'No prepared datasets found under: {PREPROCESSED_ROOT}')\n"
        "DATASET_DIR = prepared_dirs[-1]\n\n"
        "cleaned_parquet = DATASET_DIR / 'cleaned.parquet'\n"
        "cleaned_csv = DATASET_DIR / 'cleaned.csv'\n"
        "split_csv = DATASET_DIR / 'split.csv'\n\n"
        "print(f'Using prepared dataset: {DATASET_DIR}')\n"
        "print(f'Using baseline config:  {BASELINE_CONFIG_JSON}')\n\n"
        "if cleaned_parquet.exists():\n"
        "    df = pd.read_parquet(cleaned_parquet)\n"
        "elif cleaned_csv.exists():\n"
        "    df = pd.read_csv(cleaned_csv)\n"
        "else:\n"
        "    raise FileNotFoundError('Expected cleaned.parquet or cleaned.csv')\n\n"
        "splits = pd.read_csv(split_csv)\n\n"
        "from src.pipelines.features import apply_baseline_feature_config, load_baseline_feature_config\n"
        "cfg = load_baseline_feature_config(BASELINE_CONFIG_JSON)\n\n"
        "required_cols = {cfg.row_id_col, cfg.target_col}\n"
        "missing_required = required_cols - set(df.columns)\n"
        "if missing_required:\n"
        "    raise KeyError(f'Missing required columns in cleaned data: {sorted(missing_required)}')\n\n"
        "X_full = apply_baseline_feature_config(df, cfg)\n"
        "y_full = df[cfg.target_col].astype(str)\n\n"
        "df_split = df[[cfg.row_id_col]].merge(splits[[cfg.row_id_col, 'split']], on=cfg.row_id_col, how='left')\n"
        "if df_split['split'].isna().any():\n"
        "    raise ValueError('Some rows are missing split assignments (split.csv join failed)')\n\n"
        "mask_train = df_split['split'].eq('train')\n"
        "mask_val = df_split['split'].eq('val')\n"
        "mask_test = df_split['split'].eq('test')\n\n"
        "X_train, y_train = X_full.loc[mask_train].reset_index(drop=True), y_full.loc[mask_train].reset_index(drop=True)\n"
        "X_val, y_val = X_full.loc[mask_val].reset_index(drop=True), y_full.loc[mask_val].reset_index(drop=True)\n"
        "X_test, y_test = X_full.loc[mask_test].reset_index(drop=True), y_full.loc[mask_test].reset_index(drop=True)\n\n"
        "print('Split sizes:', X_train.shape, X_val.shape, X_test.shape)\n"
    )


def _preprocess_cell(needs_scaling: bool) -> str:
    scale_line = (
        "('scaler', StandardScaler())" if needs_scaling else "# ('scaler', StandardScaler())"
    )

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


def _classification_eval_cell() -> str:
    return (
        "from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score\n\n"
        "def eval_split(name: str, y_true, y_pred) -> None:\n"
        "    print(f'\\n== {name} ==')\n"
        "    print('macro_f1:', f1_score(y_true, y_pred, average='macro'))\n"
        "    print('weighted_f1:', f1_score(y_true, y_pred, average='weighted'))\n"
        "    print('balanced_acc:', balanced_accuracy_score(y_true, y_pred))\n"
        "    print('confusion_matrix:\\n', confusion_matrix(y_true, y_pred))\n"
        "    print(classification_report(y_true, y_pred))\n\n"
        "val_pred = clf.predict(X_val)\n"
        "test_pred = clf.predict(X_test)\n\n"
        "eval_split('val', y_val, val_pred)\n"
        "eval_split('test', y_test, test_pred)\n"
    )


def _clustering_eval_cell() -> str:
    return (
        "from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score\n\n"
        "def eval_clusters(name: str, y_true, labels) -> None:\n"
        "    print(f'\\n== {name} ==')\n"
        "    print('ARI:', adjusted_rand_score(y_true, labels))\n"
        "    print('NMI:', normalized_mutual_info_score(y_true, labels))\n\n"
        "eval_clusters('train', y_train, train_labels)\n"
    )


def _make_notebook(spec: ModelSpec) -> nbformat.NotebookNode:
    """Create a notebook for a given model spec."""

    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
    )

    title_md = f"# Experiment: {spec.title}\n\nTask: **{spec.task}**\n\n{spec.extra_notes}".strip()

    cells = [
        new_markdown_cell(title_md),
        new_code_cell(_common_setup_cell()),
        new_code_cell(_preprocess_cell(spec.needs_scaling)),
    ]

    # Model + fit cell
    if spec.task == "classification":
        fit_code = (
            "# Model definition + training\n\n"
            f"{spec.model_code}\n\n"
            "from sklearn.pipeline import Pipeline\n\n"
            "clf = Pipeline(steps=[('preprocess', preprocess), ('model', model)])\n"
            "clf.fit(X_train, y_train)\n"
        )
        cells.append(new_code_cell(fit_code))
        cells.append(new_code_cell(_classification_eval_cell()))

    elif spec.task == "clustering":
        fit_code = (
            "# Model definition + clustering\n\n"
            f"{spec.model_code}\n\n"
            "# For clustering, fit on train only (no leakage), then evaluate cluster quality vs labels.\n"
            "X_train_t = preprocess.fit_transform(X_train)\n"
            "train_labels = model.fit_predict(X_train_t)\n"
        )
        cells.append(new_code_cell(fit_code))
        cells.append(new_code_cell(_clustering_eval_cell()))
    else:
        raise ValueError(f"Unknown task: {spec.task}")

    nb["cells"] = cells
    return nb


def _model_specs() -> list[ModelSpec]:
    """Return the full list of model notebooks to generate."""

    # Notes.txt model list (mapped to a multiclass classification workflow when needed).
    specs: list[ModelSpec] = [
        ModelSpec(
            slug="linear_regression",
            title="Linear Regression (OVR argmax)",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.linear_model import LinearRegression\n"
                "from sklearn.multiclass import OneVsRestClassifier\n\n"
                "model = OneVsRestClassifier(LinearRegression())"
            ),
            extra_notes=(
                "This notebook maps linear regression to classification via one-vs-rest regression and argmax. "
                "Prefer logistic regression for a probabilistic linear baseline."
            ),
        ),
        ModelSpec(
            slug="lasso_regression",
            title="Lasso Regression (L1 Logistic Regression)",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.linear_model import LogisticRegression\n\n"
                "model = LogisticRegression(\n"
                "    penalty='l1', solver='saga', class_weight='balanced', max_iter=5000\n"
                ")"
            ),
        ),
        ModelSpec(
            slug="ridge_regression",
            title="Ridge Regression (L2 Logistic Regression)",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.linear_model import LogisticRegression\n\n"
                "model = LogisticRegression(\n"
                "    penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=2000\n"
                ")"
            ),
        ),
        ModelSpec(
            slug="elasticnet_regression",
            title="ElasticNet Regression (ElasticNet Logistic Regression)",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.linear_model import LogisticRegression\n\n"
                "model = LogisticRegression(\n"
                "    penalty='elasticnet', l1_ratio=0.5, solver='saga',\n"
                "    class_weight='balanced', max_iter=5000\n"
                ")"
            ),
        ),
        ModelSpec(
            slug="forward_selection",
            title="Forward Selection (SequentialFeatureSelector + LogisticRegression)",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.feature_selection import SequentialFeatureSelector\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from sklearn.pipeline import Pipeline\n\n"
                "base = LogisticRegression(max_iter=2000, class_weight='balanced')\n"
                "# Select features after preprocessing (encoded feature space)\n"
                "sfs = SequentialFeatureSelector(base, n_features_to_select=50, direction='forward', n_jobs=-1)\n"
                "model = Pipeline(steps=[('sfs', sfs), ('lr', base)])"
            ),
            extra_notes="Forward selection runs in the post-encoded feature space; it may be slow.",
        ),
        ModelSpec(
            slug="backward_selection",
            title="Backward Selection (SequentialFeatureSelector + LogisticRegression)",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.feature_selection import SequentialFeatureSelector\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from sklearn.pipeline import Pipeline\n\n"
                "base = LogisticRegression(max_iter=2000, class_weight='balanced')\n"
                "sfs = SequentialFeatureSelector(base, n_features_to_select=50, direction='backward', n_jobs=-1)\n"
                "model = Pipeline(steps=[('sfs', sfs), ('lr', base)])"
            ),
            extra_notes="Backward selection runs in the post-encoded feature space; it may be slow.",
        ),
        ModelSpec(
            slug="pcr",
            title="Principal Component Regression (PCA + LogisticRegression)",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.decomposition import PCA\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from sklearn.pipeline import Pipeline\n\n"
                "model = Pipeline(steps=[('pca', PCA(n_components=50, random_state=SEED)),\n"
                "                     ('lr', LogisticRegression(max_iter=2000, class_weight='balanced'))])"
            ),
        ),
        ModelSpec(
            slug="plsr",
            title="Partial Least Squares Regression (PLSRegression + OVR LogisticRegression)",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.cross_decomposition import PLSRegression\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from sklearn.multiclass import OneVsRestClassifier\n"
                "from sklearn.pipeline import Pipeline\n\n"
                "pls = PLSRegression(n_components=10)\n"
                "ovr = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight='balanced'))\n"
                "model = Pipeline(steps=[('pls', pls), ('ovr', ovr)])"
            ),
        ),
        ModelSpec(
            slug="logistic_regression",
            title="Logistic Regression",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.linear_model import LogisticRegression\n\n"
                "model = LogisticRegression(max_iter=2000, class_weight='balanced')"
            ),
        ),
        ModelSpec(
            slug="svm",
            title="Support Vector Machines (LinearSVC)",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.svm import LinearSVC\n\n"
                "model = LinearSVC(class_weight='balanced')"
            ),
        ),
        ModelSpec(
            slug="decision_trees",
            title="Decision Trees",
            task="classification",
            needs_scaling=False,
            model_code=(
                "from sklearn.tree import DecisionTreeClassifier\n\n"
                "model = DecisionTreeClassifier(random_state=SEED, class_weight='balanced')"
            ),
        ),
        ModelSpec(
            slug="random_forests",
            title="Random Forests",
            task="classification",
            needs_scaling=False,
            model_code=(
                "from sklearn.ensemble import RandomForestClassifier\n\n"
                "model = RandomForestClassifier(\n"
                "    n_estimators=300, random_state=SEED, n_jobs=-1, class_weight='balanced'\n"
                ")"
            ),
        ),
        ModelSpec(
            slug="knn",
            title="K-Nearest Neighbors (KNN)",
            task="classification",
            needs_scaling=True,
            model_code=(
                "from sklearn.neighbors import KNeighborsClassifier\n\n"
                "model = KNeighborsClassifier(n_neighbors=15)"
            ),
        ),
        ModelSpec(
            slug="gradient_boosting",
            title="Gradient Boosting (HistGradientBoostingClassifier)",
            task="classification",
            needs_scaling=False,
            model_code=(
                "from sklearn.ensemble import HistGradientBoostingClassifier\n\n"
                "model = HistGradientBoostingClassifier(random_state=SEED)"
            ),
        ),
        ModelSpec(
            slug="xgboost",
            title="XGBoost",
            task="classification",
            needs_scaling=False,
            model_code=(
                "from sklearn.preprocessing import LabelEncoder\n\n"
                "try:\n"
                "    from xgboost import XGBClassifier\n"
                "except Exception as exc:\n"
                "    raise ImportError('xgboost is required for this notebook. Install: pip install xgboost') from exc\n\n"
                "# XGBoost expects integer labels\n"
                "le = LabelEncoder()\n"
                "y_train_enc = le.fit_transform(y_train)\n"
                "y_val_enc = le.transform(y_val)\n"
                "y_test_enc = le.transform(y_test)\n\n"
                "model = XGBClassifier(\n"
                "    n_estimators=500, max_depth=6, learning_rate=0.1,\n"
                "    subsample=0.9, colsample_bytree=0.9,\n"
                "    objective='multi:softmax', num_class=len(le.classes_),\n"
                "    eval_metric='mlogloss', random_state=SEED\n"
                ")\n\n"
                "# Override y_* used downstream\n"
                "y_train = y_train_enc\n"
                "y_val = y_val_enc\n"
                "y_test = y_test_enc"
            ),
            extra_notes="This notebook label-encodes y for XGBoost, then evaluates on encoded labels.",
        ),
        ModelSpec(
            slug="k_means_clustering",
            title="K-Means Clustering",
            task="clustering",
            needs_scaling=True,
            model_code=(
                "from sklearn.cluster import KMeans\n\n"
                "n_clusters = int(y_train.nunique())\n"
                "model = KMeans(n_clusters=n_clusters, random_state=SEED, n_init='auto')"
            ),
        ),
        ModelSpec(
            slug="dbscan",
            title="DBSCAN",
            task="clustering",
            needs_scaling=True,
            model_code=(
                "from sklearn.cluster import DBSCAN\n\n"
                "model = DBSCAN(eps=0.5, min_samples=10)"
            ),
        ),
        ModelSpec(
            slug="hierarchical_agglomerative_clustering",
            title="Hierarchical Agglomerative Clustering",
            task="clustering",
            needs_scaling=True,
            model_code=(
                "from sklearn.cluster import AgglomerativeClustering\n\n"
                "n_clusters = int(y_train.nunique())\n"
                "model = AgglomerativeClustering(n_clusters=n_clusters)"
            ),
        ),
    ]

    return specs


def generate_notebooks(*, out_dir: Path, suffix: str) -> list[Path]:
    """Generate all experiment notebooks.

    Parameters:
        out_dir: Output directory (typically `notebooks/`).
        suffix: Timestamp suffix for filenames.

    Returns:
        List of written notebook paths.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for spec in _model_specs():
        filename = f"experiment_{spec.slug}_{suffix}.ipynb"
        path = out_dir / filename
        nb = _make_notebook(spec)
        path.write_text(nbformat.writes(nb), encoding="utf-8")
        written.append(path)

    return written


def main() -> int:
    """CLI entry point."""

    suffix = _now_suffix()
    paths = generate_notebooks(out_dir=NOTEBOOKS_DIR, suffix=suffix)

    print(f"Generated {len(paths)} notebooks:")
    for p in paths:
        print(f"  {p.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
