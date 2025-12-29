---
description: 'This prompt directs GitHub Copilot to scan a Machine Learning workspace, gather the latest experiment results, and automatically generate a polished, date‑stamped Jupyter notebook for analysis. The notebook includes structured sections, Markdown explanations, required plots, and reproducible code that clearly presents training behavior, evaluation strategy, and final test metrics.
'
agent: 'agent'
---

Goal:
Scan the current workspace for Machine Learning experiment directories, gather the latest results from each experiment, and automatically generate a new Jupyter notebook for analysis. The notebook must be named using the current date/time and must follow the workspace’s structure and conventions.

Instructions for GitHub Copilot
You are an assistant that generates a complete, polished Jupyter notebook for analyzing Machine Learning experiment results. Follow these instructions exactly.
1. Workspace Scanning
- Search the workspace recursively for experiment folders.
- An “04-predictions” contains artifacts such as:
- model checkpoints (.keras, .h5, .pt, etc.)
- training logs (.json, .csv, .txt)
- metrics files (metrics.json, results.json)
- confusion matrices or predictions (predictions.csv, confusion_matrix.npy)
- For each experiment, identify the latest run based on timestamped folders or file modification times.
2. Notebook Creation
- Create a new Jupyter notebook named with the current date/time, e.g.:
analysis_2025-12-23_09-15.ipynb
- Create a folder with the same name to store generated PNG charts.

3. Notebook Structure Requirements
Professional Header (Markdown cell)
Include:
- Project Title
- Date of notebook creation
- One-sentence description of the notebook’s purpose
(“This notebook analyzes the latest ML experiment results.”)

4. Section Organization (Part A Structure)
Follow this structure:
A.1 — Dataset & Evaluation Strategy
Markdown cell explaining:
- How the validation split was used
- How the test set was kept separate
- Which metrics are appropriate for the dataset and why
A.2 — Training & Validation Analysis
For each experiment:
- Load training logs
- Generate and save:
- Training vs. validation loss curve (PNG)
- Training vs. validation accuracy or primary metric curve (PNG)
- Insert Markdown before each plot explaining:
- What the plot shows
- Why it matters for model evaluation
A.3 — Test Set Evaluation
For each experiment:
- Load test predictions and ground truth
- Compute:
- Confusion matrix (PNG)
- Per-class metrics table
- F1 score table (macro/micro/weighted)
- Include a clean printout of final test metrics:
- Accuracy
- Precision/Recall/F1
- Any dataset-specific metrics

5. Code Quality Requirements
Markdown Documentation
Before every major code cell, include a Markdown cell describing:
- What the code does
- How it fits into the modeling workflow
Readable, Commented Code
- Comment every major block
- Use clear function names
- Avoid magic numbers — define constants at the top of the cell
- Use modular code where appropriate
Reproducibility
- Notebook must run end‑to‑end without errors
- No hard‑coded absolute paths
- Use workspace‑relative paths
- Set random seeds for all frameworks used (NumPy, TensorFlow, PyTorch, etc.)

6. Plot Requirements
All plots must:
- Be readable
- Have axis labels, titles, legends
- Be saved as PNG files in the notebook‑named folder
- Be displayed inline in the notebook
- Be integrated into the narrative with Markdown explanations

7. Output Requirements
The final notebook must:
- Be polished and easy for an LF to follow
- Contain all required Markdown, plots, tables, and metrics
- Reflect the structure and conventions of the workspace
- Be created fresh each time the prompt is executed

8. Final Task
Generate the complete notebook file with all sections, Markdown, code, plots, and analysis.
Use the workspace’s experiment results to populate the notebook. Compare experiments where applicable.
Ensure the notebook is ready for immediate use by data scientists and ML engineers for result interpretation and decision-making.
Check the config files in the workspace for any dataset-specific or experiment-specific details that should be included in the analysis, including random seeds, evaluation metrics, data splits, and color palettes. Write out which model performed best based on the final test metrics.

Python warning 
FutureWarning: In the future `np.object` will be defined as the corresponding NumPy scalar. if not hasattr(np, "object")

Example Chart Generation Snippet
```python
def plot_training_curves(history, experiment_name=None,  metric_name=None):
    """
    Plot training vs validation loss and a chosen metric from a Keras History object.
    metric_name can be 'accuracy', 'sparse_categorical_accuracy', 'macro_f1', etc.
    If None, it tries to infer a reasonable metric automatically.
    """
    # Infer a metric if not provided
    if metric_name is None:
        for cand in ["accuracy", "sparse_categorical_accuracy", "macro_f1", "f1"]:
            if cand in history:
                metric_name = cand
                break
        if metric_name is None:
            raise ValueError(
                f"Could not infer metric_name. Available keys: {list(history.keys())}"
            )

    epochs = range(1, len(history["loss"]) + 1)

    # --- Loss ---
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], linestyle="--", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if experiment_name is not None:
        plt.title(f"{experiment_name} Training vs Validation Loss")
    else:
        plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("")

    # --- Metric ---
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history[metric_name], label=f"Train {metric_name}")
    val_key = f"val_{metric_name}"
    if val_key in history:
        plt.plot(
            epochs,
            history[val_key],
            linestyle="--",
            label=f"Val {metric_name}",
        )
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    if experiment_name is not None:
        plt.title(f"{experiment_name} Training vs Validation {metric_name}")
    else:
        plt.title(f"Training vs Validation {metric_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, experiment_name=None, normalize=True):
    """
    Plot a confusion matrix.
    If normalize=True, rows are normalized to sum to 1.
    """

    cmap = LinearSegmentedColormap.from_list(
            "custom_cm",
            [
                COLOR_PALETTE[0],
                COLOR_PALETTE[5],
            ]
        )

    cm = np.array(cm)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # avoid division by zero if any class has zero support
        row_sums[row_sums == 0] = 1.0
        cm = cm.astype("float") / row_sums

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if experiment_name is not None:
        title = (
            f"{experiment_name} Normalized Confusion Matrix"
            if normalize
            else f"{experiment_name} Confusion Matrix"
        )
    else:
        title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"

    ax.set_title(title)

    # Add values in cells (optional)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_str = f"{value:.2f}" if normalize else int(value)
            ax.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                fontsize=6,
            )

    fig.tight_layout()
    plt.show()
    return fig, ax

def print_final_test_metrics(y_true, y_pred):
    """
    Print a clean summary of overall test performance:
    accuracy, macro-F1, and weighted-F1.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("=== Final Test Metrics ===")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
```
