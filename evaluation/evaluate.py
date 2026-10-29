"""
evaluate.py
============
Phase 6 held-out evaluation. Loads the trained XGBoost booster and the
leakage-fixed test split (produced by preprocessing/preprocess.py under the
grouped split strategy), scores it, and writes honest test-set metrics and
publication-quality figures.

This is the artifact that produces the project's *headline* number. It is
deliberately separate from train.py: train.py's reported macro-F1 was a
VALIDATION score from the model-comparison stage, not a held-out test metric.
The whole point of the Section 8 split fix is that this test set is genuinely
held out (no overlapping-window leakage from train), so whatever macro-F1 this
prints is the honest generalization estimate -- report it as-is, do not tune.

Outputs
-------
reports/metrics/evaluation_summary.json  -- accuracy, macro/weighted P/R/F1, per-class
reports/metrics/evaluation_metrics.csv   -- per-class precision/recall/f1/support
reports/figures/eval_confusion_matrix.png
reports/figures/eval_per_class_f1.png
reports/figures/eval_roc_curves.png
reports/figures/eval_pr_curves.png

Figure filenames use an `eval_` prefix so they never collide with train.py's
unprefixed figures in the shared reports/figures/ directory (fixes the naming
drift noted in README Section 9 rather than reintroducing it).

Dependencies
------------
numpy, pandas, scikit-learn, xgboost, matplotlib, seaborn, pyyaml, pyarrow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from feature_extractor import load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reports"))
import plotting  # noqa: E402


def load_artifacts(cfg: dict):
    models_dir = Path(cfg["paths"]["models_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])

    booster = xgb.Booster()
    booster.load_model(str(models_dir / "xgb_model.json"))

    with open(models_dir / "feature_columns.json") as f:
        feature_cols = json.load(f)
    with open(models_dir / "label_encoder.json") as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]

    test_df = pd.read_parquet(processed_dir / "test.parquet")
    return booster, feature_cols, class_names, test_df


def predict(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """Return per-class probability matrix (n_samples, n_classes)."""
    proba = booster.predict(xgb.DMatrix(X))
    if proba.ndim == 1:  # binary edge case -- promote to 2 columns
        proba = np.column_stack([1 - proba, proba])
    return proba


def compute_summary(y_true, y_pred, y_proba, class_names) -> dict:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    # One-vs-rest ROC-AUC / PR-AUC, guarded for classes absent from the test set.
    y_true_oh = np.eye(len(class_names))[y_true]
    per_class = {}
    for i, cls in enumerate(class_names):
        entry = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        if y_true_oh[:, i].sum() > 0 and y_true_oh[:, i].sum() < len(y_true):
            entry["roc_auc"] = float(roc_auc_score(y_true_oh[:, i], y_proba[:, i]))
            entry["pr_auc"] = float(average_precision_score(y_true_oh[:, i], y_proba[:, i]))
        else:
            entry["roc_auc"] = None
            entry["pr_auc"] = None
        per_class[cls] = entry

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_test_windows": int(len(y_true)),
        "per_class": per_class,
    }


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    booster, feature_cols, class_names, test_df = load_artifacts(cfg)
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["label_encoded"].to_numpy()

    y_proba = predict(booster, X_test)
    y_pred = y_proba.argmax(axis=1)

    summary = compute_summary(y_test, y_pred, y_proba, class_names)

    # --- Console report ---
    print("\n===== HELD-OUT TEST EVALUATION (leakage-fixed split) =====")
    print(f"  test windows : {summary['n_test_windows']}")
    print(f"  accuracy     : {summary['accuracy']:.4f}")
    print(f"  macro F1     : {summary['macro_f1']:.4f}   <-- honest headline number")
    print(f"  weighted F1  : {summary['weighted_f1']:.4f}")
    print("  per-class F1 :")
    for cls in class_names:
        pc = summary["per_class"][cls]
        print(f"    {cls:8s} f1={pc['f1']:.4f}  precision={pc['precision']:.4f}  recall={pc['recall']:.4f}  support={pc['support']}")

    # --- Persist metrics ---
    with open(metrics_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    per_class_rows = [
        {"class": cls, **{k: summary["per_class"][cls][k] for k in ("precision", "recall", "f1", "support", "roc_auc", "pr_auc")}}
        for cls in class_names
    ]
    pd.DataFrame(per_class_rows).to_csv(metrics_dir / "evaluation_metrics.csv", index=False)

    # --- Figures (reuse the shared plotting utilities; eval_ prefix) ---
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    plotting.plot_confusion_matrix(cm, class_names, figures_dir / "eval_confusion_matrix.png", normalize=True)

    f1_scores = np.array([summary["per_class"][cls]["f1"] for cls in class_names])
    plotting.plot_per_class_f1(class_names, f1_scores, figures_dir / "eval_per_class_f1.png")

    # ROC / PR curves, one-vs-rest, only for classes present in the test set.
    y_true_oh = np.eye(len(class_names))[y_test]
    present = [cls for i, cls in enumerate(class_names) if 0 < y_true_oh[:, i].sum() < len(y_test)]
    if present:
        fpr_d, tpr_d, auc_d, prec_d, rec_d, ap_d = {}, {}, {}, {}, {}, {}
        for i, cls in enumerate(class_names):
            if cls not in present:
                continue
            fpr_d[cls], tpr_d[cls], _ = roc_curve(y_true_oh[:, i], y_proba[:, i])
            auc_d[cls] = summary["per_class"][cls]["roc_auc"]
            prec_d[cls], rec_d[cls], _ = precision_recall_curve(y_true_oh[:, i], y_proba[:, i])
            ap_d[cls] = summary["per_class"][cls]["pr_auc"]
        plotting.plot_multiclass_roc(fpr_d, tpr_d, auc_d, present, figures_dir / "eval_roc_curves.png")
        plotting.plot_multiclass_pr(prec_d, rec_d, ap_d, present, figures_dir / "eval_pr_curves.png")

    print(f"\n[done] Phase 6 evaluation complete.")
    print(f"  Summary  -> {metrics_dir / 'evaluation_summary.json'}")
    print(f"  Per-class-> {metrics_dir / 'evaluation_metrics.csv'}")
    print(f"  Figures  -> {figures_dir}/ (eval_* )")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")
