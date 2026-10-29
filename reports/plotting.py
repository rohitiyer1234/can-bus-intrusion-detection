"""
plotting.py
============
Shared, publication-quality plotting utilities for the CAN IDS pipeline.
Every function here takes REAL data (arrays/dicts computed by the caller)
and saves a figure to reports/figures/ -- nothing in this file generates
placeholder or synthetic figure content.

Called by:
  - training/train.py    -> dataset distribution, loss curves, learning
                             curve, feature importance
  - evaluation/evaluate.py -> confusion matrix, ROC, PR, per-class F1
  - training/window_size_sweep.py -> window-size comparison
  - explainability/shap_analysis.py (Phase 7) -> SHAP summary/dependence

Dependencies
------------
matplotlib, seaborn, numpy, scikit-learn (for confusion_matrix/roc_curve/etc,
called by the consumer, not here)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe for training runs / CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] saved {out_path}")


def plot_class_distribution(labels: np.ndarray, out_path: Path, title: str = "Dataset Class Distribution"):
    values, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(values[order], counts[order], color=sns.color_palette("crest", len(values)))
    ax.set_ylabel("Window count")
    ax.set_title(title)
    for b, c in zip(bars, counts[order]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(c), ha="center", va="bottom", fontsize=9)
    _save(fig, out_path)


def plot_window_size_comparison(window_sizes_ms: list[int], scores: list[float], out_path: Path, metric_name: str = "Macro F1"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(window_sizes_ms, scores, marker="o", linewidth=2)
    ax.set_xlabel("Window size (ms)")
    ax.set_ylabel(metric_name)
    ax.set_title("Window Size vs. Detection Performance")
    _save(fig, out_path)


def plot_loss_curve(iterations: np.ndarray, loss: np.ndarray, out_path: Path, title: str, color: str = "#1f77b4"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iterations, loss, color=color, linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Multiclass log-loss")
    ax.set_title(title)
    _save(fig, out_path)


def plot_learning_curve(train_sizes: np.ndarray, train_scores: np.ndarray, val_scores: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train_sizes, train_scores, marker="o", label="Train", linewidth=2)
    ax.plot(train_sizes, val_scores, marker="s", label="Validation", linewidth=2)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Macro F1")
    ax.set_title("Learning Curve")
    ax.legend()
    _save(fig, out_path)


def plot_feature_importance(feature_names: list[str], importances: np.ndarray, out_path: Path, top_n: int = 25):
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.28 * len(order))))
    ax.barh(np.array(feature_names)[order][::-1], importances[order][::-1], color=sns.color_palette("crest", len(order)))
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importance")
    _save(fig, out_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path, normalize: bool = True):
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    _save(fig, out_path)


def plot_multiclass_roc(fpr_dict: dict, tpr_dict: dict, auc_dict: dict, class_names: list[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for cls in class_names:
        ax.plot(fpr_dict[cls], tpr_dict[cls], label=f"{cls} (AUC={auc_dict[cls]:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_multiclass_pr(precision_dict: dict, recall_dict: dict, ap_dict: dict, class_names: list[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for cls in class_names:
        ax.plot(recall_dict[cls], precision_dict[cls], label=f"{cls} (AP={ap_dict[cls]:.3f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_per_class_f1(class_names: list[str], f1_scores: np.ndarray, out_path: Path):
    order = np.argsort(f1_scores)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(np.array(class_names)[order], np.array(f1_scores)[order], color=sns.color_palette("flare", len(class_names)))
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1 Score")
    for b, s in zip(bars, np.array(f1_scores)[order]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{s:.2f}", ha="center", va="bottom", fontsize=9)
    _save(fig, out_path)