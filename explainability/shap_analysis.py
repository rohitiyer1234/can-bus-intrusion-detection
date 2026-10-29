"""
shap_analysis.py
================
Phase 7 explainability. Computes SHAP values for the leakage-fixed XGBoost
model and exposes both:

  - GLOBAL explanations: a SHAP summary plot (feature ranking across all
    classes) + per-class beeswarm plots, saved to reports/figures/ with a
    `shap_` prefix.
  - LOCAL explanations: `explain_prediction(...)`, a reusable function that
    returns the top contributing features for ONE window's classification.
    This is imported by the Phase 8 LLM incident reporter (to ground its
    "Technical Analysis" in real feature attributions) and by the Phase 10
    dashboard's SHAP Explanations panel.

Explicit dependency (README Section 8): this must be run against the
leakage-fixed model. SHAP importances computed on a contaminated evaluation
set would inherit the same validity problem the split fix resolves.

`shap.TreeExplainer` is the natural fit for XGBoost -- exact, fast tree SHAP.

Dependencies
------------
numpy, pandas, shap, xgboost, matplotlib, pyyaml, pyarrow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from feature_extractor import load_config  # noqa: E402


class ShapExplainer:
    """Wraps a TreeExplainer over the trained booster; caches feature/class names."""

    def __init__(self, booster: xgb.Booster, feature_cols: list[str], class_names: list[str]):
        self.booster = booster
        self.feature_cols = feature_cols
        self.class_names = class_names
        self.explainer = shap.TreeExplainer(booster)

    @classmethod
    def from_config(cls, cfg: dict) -> "ShapExplainer":
        models_dir = Path(cfg["paths"]["models_dir"])
        booster = xgb.Booster()
        booster.load_model(str(models_dir / "xgb_model.json"))
        with open(models_dir / "feature_columns.json") as f:
            feature_cols = json.load(f)
        with open(models_dir / "label_encoder.json") as f:
            label_map = json.load(f)
        class_names = [label_map[str(i)] for i in range(len(label_map))]
        return cls(booster, feature_cols, class_names)

    def _shap_values(self, X: np.ndarray) -> list[np.ndarray]:
        """
        Return a list of per-class SHAP arrays [(n_samples, n_features), ...].
        Normalizes across shap versions, which variously return a list or a
        single (n_samples, n_features, n_classes) ndarray for multiclass trees.
        """
        vals = self.explainer.shap_values(X)
        if isinstance(vals, list):
            return vals
        if vals.ndim == 3:  # (n_samples, n_features, n_classes)
            return [vals[:, :, k] for k in range(vals.shape[2])]
        return [vals]  # binary/regression fallback

    def explain_prediction(self, features: dict | pd.Series | np.ndarray, top_k: int = 5) -> dict:
        """
        LOCAL explanation for a single window.

        `features` may be a dict/Series keyed by feature name, or a raw array in
        `self.feature_cols` order. Returns the predicted class, its confidence,
        and the top_k features driving THAT class's score (signed SHAP value +
        the feature's actual value), ready to feed the LLM incident report.
        """
        x = self._coerce_row(features)
        proba = self.booster.predict(xgb.DMatrix(x.reshape(1, -1)))[0]
        pred_idx = int(np.argmax(proba))
        pred_class = self.class_names[pred_idx]

        class_shap = self._shap_values(x.reshape(1, -1))[pred_idx][0]  # (n_features,)
        order = np.argsort(np.abs(class_shap))[::-1][:top_k]
        top_features = [
            {
                "feature": self.feature_cols[i],
                "value": float(x[i]),
                "shap_value": float(class_shap[i]),
                "direction": "increases" if class_shap[i] >= 0 else "decreases",
            }
            for i in order
        ]
        return {
            "predicted_class": pred_class,
            "confidence": float(proba[pred_idx]),
            "class_probabilities": {c: float(p) for c, p in zip(self.class_names, proba)},
            "top_features": top_features,
        }

    def _coerce_row(self, features) -> np.ndarray:
        if isinstance(features, dict):
            return np.array([float(features[c]) for c in self.feature_cols], dtype=np.float64)
        if isinstance(features, pd.Series):
            return features[self.feature_cols].to_numpy(dtype=np.float64)
        arr = np.asarray(features, dtype=np.float64).ravel()
        if arr.size != len(self.feature_cols):
            raise ValueError(f"expected {len(self.feature_cols)} features, got {arr.size}")
        return arr

    def global_summary(self, X: np.ndarray, figures_dir: Path, max_display: int = 20) -> None:
        """Save a global multiclass SHAP summary bar plot + per-class beeswarm plots."""
        figures_dir.mkdir(parents=True, exist_ok=True)
        shap_list = self._shap_values(X)

        # Multiclass summary (mean |SHAP| per feature, stacked by class).
        plt.figure()
        shap.summary_plot(
            shap_list, X, feature_names=self.feature_cols, class_names=self.class_names,
            plot_type="bar", max_display=max_display, show=False,
        )
        plt.title("SHAP Global Feature Importance (all classes)")
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[shap] saved {figures_dir / 'shap_summary.png'}")

        # Per-class beeswarm.
        for k, cls in enumerate(self.class_names):
            plt.figure()
            shap.summary_plot(
                shap_list[k], X, feature_names=self.feature_cols,
                max_display=max_display, show=False,
            )
            plt.title(f"SHAP Feature Effects — class: {cls}")
            plt.tight_layout()
            out = figures_dir / f"shap_class_{cls}.png"
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[shap] saved {out}")


def main(config_path: str = "configs/config.yaml", sample_n: int = 2000):
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    figures_dir = Path(cfg["paths"]["reports_dir"]) / "figures"

    explainer = ShapExplainer.from_config(cfg)
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    # SHAP on the full test set can be large; sample for the global plots
    # (stratified-ish by taking a head after shuffle) while keeping it deterministic.
    if len(test_df) > sample_n:
        test_df = test_df.sample(n=sample_n, random_state=cfg["random_seed"])
    X = test_df[explainer.feature_cols].to_numpy()

    explainer.global_summary(X, figures_dir)

    # Demonstrate a local explanation on the first test window.
    demo = explainer.explain_prediction(test_df.iloc[0])
    print("\n[shap] example local explanation (first test window):")
    print(f"  predicted: {demo['predicted_class']}  (confidence {demo['confidence']:.4f})")
    for tf in demo["top_features"]:
        print(f"    {tf['feature']:22s} value={tf['value']:.4f}  shap={tf['shap_value']:+.4f} ({tf['direction']})")
    print(f"\n[done] Phase 7 SHAP analysis complete. Figures -> {figures_dir}/ (shap_*)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")
