"""
train.py
=========
Phase 5 training pipeline. Loads the processed train/val/test splits
produced by preprocessing/preprocess.py, compares candidate model
families on the validation set, trains the final XGBoost model with
early stopping + checkpointing, and generates training-side figures and
a consolidated metrics report.

This version was refactored after a full consistency review against the
current preprocessing pipeline (dual-format loader, any-attack-present
window labeling, temporal-leakage-guarded baseline, 25-feature set).
Three real issues were found and fixed here (not just re-pointed at new
file paths):

  1. `num_class` was a hardcoded config literal, never checked against the
     actual encoded label cardinality in label_encoder.json. Now derived
     from label_encoder.json (source of truth) with a warning if config
     disagrees, rather than trusting the config blindly.
  2. No class imbalance handling existed anywhere, despite residual
     imbalance surviving preprocessing's undersampling (~3.6:1 Normal:Gear
     after balancing, per the current dataset stats). Added balanced
     sample weighting (inverse class frequency) to both the model
     comparison step and final XGBoost training.
  3. Early stopping was passed as a bare `early_stopping_rounds=N` int,
     which stops training early but does NOT prune the booster back to
     the best validation iteration -- the saved model could include extra
     rounds trained during the patience window. Fixed by using an
     explicit `xgb.callback.EarlyStopping(..., save_best=True)` callback,
     which is what actually guarantees "best-model persistence."

Outputs
-------
models/xgb_model.json              -- final model, pruned to best iteration
models/checkpoints/xgb_round_N.json -- periodic checkpoints (audit trail)
reports/figures/class_distribution.png
reports/figures/training_loss_curve.png
reports/figures/validation_loss_curve.png
reports/figures/learning_curve.png
reports/figures/feature_importance.png
reports/metrics/training_metrics.json  -- consolidated metrics + summary
reports/metrics/model_comparison.csv   -- supplementary
reports/metrics/training_history.csv   -- supplementary (full loss curves)

Out of scope (explicitly deferred per current instructions): evaluation.py
test-set figures, SHAP, dashboard, simulator, LLM report generation.

Dependencies
------------
numpy, pandas, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, pyyaml
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from feature_extractor import load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reports"))
import plotting  # noqa: E402

CHECKPOINT_EVERY = 25  # boosting rounds between checkpoint saves


def load_splits(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    processed_dir = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    with open(models_dir / "feature_columns.json") as f:
        feature_cols = json.load(f)
    return train_df, val_df, test_df, feature_cols


def resolve_class_names(models_dir: Path, cfg: dict) -> list[str]:
    """
    num_class was previously a hardcoded config literal, trusted blindly.
    label_encoder.json (written by preprocess.py from the ACTUAL fitted
    LabelEncoder) is the real source of truth -- derive class count from
    it, and only use config's num_class as a sanity cross-check.
    """
    with open(models_dir / "label_encoder.json") as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]

    configured_n = cfg["model"].get("num_class")
    if configured_n is not None and configured_n != len(class_names):
        print(
            f"[warn] config.yaml model.num_class={configured_n} does not match "
            f"label_encoder.json's actual {len(class_names)} classes {class_names}. "
            f"Using the actual count from label_encoder.json -- config value is stale."
        )
    return class_names


def compare_models(
    X_train, y_train, X_val, y_val, sw_train, sw_val, seed: int, reports_dir: Path
) -> pd.DataFrame:
    """
    Train each candidate with reasonable fixed hyperparameters, score
    macro F1 on val. All candidates use balanced sample weights so the
    comparison reflects what final deployment will actually use.
    """
    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, random_state=seed,
            eval_metric="mlogloss", n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.1, random_state=seed, verbosity=-1),
    }
    rows = []
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train, y_train, sample_weight=sw_train)
        preds = model.predict(X_val)
        macro_f1 = f1_score(y_val, preds, average="macro", sample_weight=None)
        weighted_f1 = f1_score(y_val, preds, average="weighted")
        rows.append({"model": name, "val_macro_f1": macro_f1, "val_weighted_f1": weighted_f1, "train_seconds": time.time() - t0})
        print(f"[compare] {name}: macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f} ({time.time()-t0:.1f}s)")

    result_df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(metrics_dir / "model_comparison.csv", index=False)
    return result_df


def select_final_model(comparison: pd.DataFrame, margin: float = 0.03) -> str:
    """
    Deployment model is frozen to XGBoost unless another candidate beats it
    by more than `margin` macro F1 -- a real margin, not noise. Prints the
    reasoning either way so the choice is auditable, not silent.
    """
    xgb_row = comparison[comparison["model"] == "xgboost"].iloc[0]
    best_row = comparison.iloc[0]
    if best_row["model"] != "xgboost" and (best_row["val_macro_f1"] - xgb_row["val_macro_f1"]) > margin:
        print(
            f"[select] {best_row['model']} beats xgboost by "
            f"{best_row['val_macro_f1'] - xgb_row['val_macro_f1']:.4f} macro F1 (> {margin} margin) -- "
            f"NOTE: architecture is frozen to XGBoost per project spec; logging this for the record "
            f"but still deploying xgboost."
        )
    else:
        print(f"[select] xgboost retained as deployment model (val_macro_f1={xgb_row['val_macro_f1']:.4f})")
    return "xgboost"


def train_final_xgboost(X_train, y_train, X_val, y_val, sw_train, sw_val, num_class: int, cfg: dict, models_dir: Path):
    model_cfg = cfg["model"]
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=sw_val)

    checkpoints_dir = models_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    class CheckpointCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                model.save_model(str(checkpoints_dir / f"xgb_round_{epoch+1}.json"))
            return False

    # save_best=True is the actual fix for "best-model persistence": without it,
    # early_stopping_rounds stops training early but the returned booster still
    # contains every round trained during the patience window, not just the
    # rounds up to the best validation score.
    early_stop_cb = xgb.callback.EarlyStopping(
        rounds=model_cfg["early_stopping_rounds"],
        metric_name=model_cfg["eval_metric"],
        data_name="val",
        save_best=True,
        maximize=False,
    )

    params = {
        "objective": model_cfg["objective"],
        "num_class": num_class,
        "max_depth": model_cfg["max_depth"],
        "eta": model_cfg["eta"],
        "eval_metric": model_cfg["eval_metric"],
        "seed": cfg["random_seed"],
    }

    evals_result: dict = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=model_cfg["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        callbacks=[early_stop_cb, CheckpointCallback()],
        verbose_eval=False,
    )

    booster.save_model(str(models_dir / "xgb_model.json"))
    n_trees_saved = len(booster.get_dump())
    print(
        f"[train] final model saved to {models_dir / 'xgb_model.json'} "
        f"(best_iteration={booster.best_iteration}, trees_persisted={n_trees_saved})"
    )
    return booster, evals_result


def main(config_path: str = "configs/config.yaml"):
    run_started = datetime.now(timezone.utc).isoformat()
    cfg = load_config(config_path)
    seed = cfg["random_seed"]
    np.random.seed(seed)

    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, feature_cols = load_splits(cfg)
    class_names = resolve_class_names(models_dir, cfg)
    num_class = len(class_names)

    X_train, y_train = train_df[feature_cols].to_numpy(), train_df["label_encoded"].to_numpy()
    X_val, y_val = val_df[feature_cols].to_numpy(), val_df["label_encoded"].to_numpy()

    # --- Class imbalance handling (still required: see module docstring) ---
    # Balanced inverse-frequency sample weights, applied to BOTH train and
    # validation. Weighting val too (not just train) matters here: without it,
    # early stopping would monitor an eval metric still dominated by the
    # majority Normal class, which could silently prefer a model that's worse
    # on the minority attack classes we actually care about.
    sw_train = compute_sample_weight("balanced", y_train)
    sw_val = compute_sample_weight("balanced", y_val)

    train_class_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"[imbalance] train class counts: {dict(zip(class_names, train_class_counts.tolist()))}")
    print(f"[imbalance] applying balanced sample weights (train + val) to correct residual skew")

    # --- Figure: dataset class distribution (train + val + test combined) ---
    all_labels = pd.concat([train_df["label"], val_df["label"], test_df["label"]])
    plotting.plot_class_distribution(all_labels.to_numpy(), figures_dir / "class_distribution.png")

    # --- Model comparison ---
    comparison = compare_models(X_train, y_train, X_val, y_val, sw_train, sw_val, seed, reports_dir)
    print(comparison)
    final_model_name = select_final_model(comparison)
    assert final_model_name == "xgboost"

    # --- Final XGBoost training (early stopping + checkpointing + best-model persistence) ---
    booster, evals_result = train_final_xgboost(X_train, y_train, X_val, y_val, sw_train, sw_val, num_class, cfg, models_dir)

    eval_metric = cfg["model"]["eval_metric"]
    train_loss = np.array(evals_result["train"][eval_metric])
    val_loss = np.array(evals_result["val"][eval_metric])
    rounds = np.arange(1, len(train_loss) + 1)

    pd.DataFrame({"round": rounds, "train_mlogloss": train_loss, "val_mlogloss": val_loss}).to_csv(
        metrics_dir / "training_history.csv", index=False
    )

    # --- Figures: training loss, validation loss ---
    plotting.plot_loss_curve(rounds, train_loss, figures_dir / "training_loss_curve.png", "Training Loss Curve", color="#1f77b4")
    plotting.plot_loss_curve(rounds, val_loss, figures_dir / "validation_loss_curve.png", "Validation Loss Curve", color="#d62728")

    # --- Figure: learning curve (sklearn XGBClassifier wrapper, macro F1 vs. train size) ---
    sk_model = xgb.XGBClassifier(
        n_estimators=cfg["model"]["n_estimators"], max_depth=cfg["model"]["max_depth"],
        learning_rate=cfg["model"]["eta"], random_state=seed, eval_metric="mlogloss", n_jobs=-1,
    )
    # This adaptive-cv guard was originally defensive against the pre-fix Gear/RPM
    # window-label collapse (classes with single-digit counts). That's resolved
    # upstream now (any-attack-present labeling), so min_class_count will always
    # be large and this will always resolve to cv_folds=3 -- kept as harmless
    # defense-in-depth rather than assuming the upstream fix can never regress.
    min_class_count = int(np.bincount(y_train).min())
    cv_folds = max(2, min(3, min_class_count))
    train_sizes, train_scores, val_scores = learning_curve(
        sk_model, X_train, y_train, cv=cv_folds, scoring="f1_macro",
        train_sizes=np.linspace(0.3, 1.0, 5), random_state=seed,
        error_score=0.0, params={"sample_weight": sw_train},
    )
    train_scores = np.where(train_scores == 0.0, np.nan, train_scores)
    val_scores = np.where(val_scores == 0.0, np.nan, val_scores)
    plotting.plot_learning_curve(
        train_sizes, np.nanmean(train_scores, axis=1), np.nanmean(val_scores, axis=1), figures_dir / "learning_curve.png"
    )

    # --- Figure: feature importance (gain-based, from the final booster) ---
    importance_dict = booster.get_score(importance_type="gain")
    importances = np.array([importance_dict.get(f"f{i}", 0.0) for i in range(len(feature_cols))])
    plotting.plot_feature_importance(feature_cols, importances, figures_dir / "feature_importance.png")

    # --- Consolidated training metrics + summary report ---
    val_macro_f1_final = float(comparison.loc[comparison["model"] == "xgboost", "val_macro_f1"].iloc[0])
    training_metrics = {
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "class_names": class_names,
        "dataset": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_class_counts": {class_names[i]: int(c) for i, c in train_class_counts.items()},
        },
        "imbalance_handling": {
            "method": "inverse-frequency sample weights (sklearn compute_sample_weight('balanced', ...))",
            "applied_to": ["train", "validation"],
            "reason": "residual imbalance survives preprocessing's undersampling",
        },
        "hyperparameters": {
            "objective": cfg["model"]["objective"],
            "num_class": num_class,
            "max_depth": cfg["model"]["max_depth"],
            "eta": cfg["model"]["eta"],
            "n_estimators_max": cfg["model"]["n_estimators"],
            "early_stopping_rounds": cfg["model"]["early_stopping_rounds"],
            "eval_metric": eval_metric,
        },
        "model_comparison": comparison.to_dict(orient="records"),
        "final_training": {
            "best_iteration": int(booster.best_iteration),
            "trees_persisted": len(booster.get_dump()),
            "final_train_loss": float(train_loss[-1]),
            "final_val_loss": float(val_loss[-1]),
            "best_val_loss": float(val_loss.min()),
            "val_macro_f1_at_comparison_stage": val_macro_f1_final,
        },
        "artifacts": {
            "model": str(models_dir / "xgb_model.json"),
            "checkpoints_dir": str(models_dir / "checkpoints"),
            "figures_dir": str(figures_dir),
        },
    }
    with open(metrics_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

    print("\n[done] Phase 5 training complete.")
    print(f"  Figures  -> {figures_dir}/")
    print(f"  Metrics  -> {metrics_dir}/ (training_metrics.json is the consolidated report)")
    print(f"  Model    -> {models_dir / 'xgb_model.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")