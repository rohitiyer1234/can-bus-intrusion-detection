"""
can_stream_simulator.py
=======================
Phase 9 real-time CAN stream simulator. Replays a recorded HCRL capture as a
"live" stream through the SAME detection path used at training time, for
demonstration only (no real CAN hardware -- out of scope per project constraints).

Two design constraints are load-bearing here:

  1. Reuse the persisted baseline AS-IS (README Section 4 / Section 12).
     `models/id_baseline.pkl` is loaded and never refit online. Refitting the
     per-ID baseline during a live, possibly-attacker-influenced stream would let
     the model "learn" an ongoing attack as normal -- a deliberate anti-leakage
     decision from Phase 4.

  2. No train/serve skew (README Section 12). Windows are extracted with the
     exact same `feature_extractor` code path (`_prepare_arrays` +
     `extract_window_features`) used to build the training matrix, just driven
     incrementally so each window is emitted as it "arrives".

Each emitted window is scored by the trained booster; optionally an incident
report (Phase 8) is generated for attack windows. Feeds the Phase 10 dashboard.

Dependencies
------------
numpy, pandas, xgboost, pyyaml, pyarrow  (+ optional: llm.incident_report)
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "preprocessing"))
from feature_extractor import (  # noqa: E402
    IDBaseline,
    StreamArrays,
    _prepare_arrays,
    extract_window_features,
    load_config,
)
from preprocess import load_and_label_file  # noqa: E402


@dataclass
class WindowPrediction:
    window_index: int
    window_start: float
    window_end: float
    predicted_class: str
    confidence: float
    class_probabilities: dict
    true_label: Optional[str]  # from replayed ground truth, if present
    features: dict


class CANStreamSimulator:
    """Replays a capture window-by-window and scores each window as it arrives."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        models_dir = Path(cfg["paths"]["models_dir"])

        # (1) Load the persisted baseline AS-IS -- never refit.
        with open(models_dir / "id_baseline.pkl", "rb") as f:
            self.baseline: IDBaseline = pickle.load(f)

        self.booster = xgb.Booster()
        self.booster.load_model(str(models_dir / "xgb_model.json"))
        with open(models_dir / "feature_columns.json") as f:
            self.feature_cols = json.load(f)
        with open(models_dir / "label_encoder.json") as f:
            label_map = json.load(f)
        self.class_names = [label_map[str(i)] for i in range(len(label_map))]

        self.window_s = cfg["windowing"]["window_ms"] / 1000.0
        self.stride_s = cfg["windowing"]["stride_ms"] / 1000.0
        self.min_msgs = cfg["windowing"]["min_msgs_per_window"]

    def _predict_window(self, feats: dict) -> tuple[str, float, dict]:
        x = np.array([feats[c] for c in self.feature_cols], dtype=np.float64).reshape(1, -1)
        proba = self.booster.predict(xgb.DMatrix(x))[0]
        idx = int(np.argmax(proba))
        return self.class_names[idx], float(proba[idx]), {c: float(p) for c, p in zip(self.class_names, proba)}

    def stream_dataframe(self, df: pd.DataFrame, realtime_factor: float = 0.0) -> Iterator[WindowPrediction]:
        """
        Slide a time-based window over a prepared capture DataFrame, yielding one
        WindowPrediction per window as it "arrives". `realtime_factor` > 0 sleeps
        proportionally to the stride so a demo can play at (a fraction of) wall
        clock; 0 replays as fast as possible.

        Windowing math is identical to feature_extractor.build_feature_matrix, so
        features here match training features exactly (no skew).
        """
        df = df.sort_values("timestamp").reset_index(drop=True)
        arrays: StreamArrays = _prepare_arrays(df)
        ts_arr = arrays.ts
        t_start, t_end = float(ts_arr[0]), float(ts_arr[-1])

        w_start = t_start
        widx = 0
        while w_start < t_end:
            w_end = w_start + self.window_s
            lo = int(np.searchsorted(ts_arr, w_start, side="left"))
            hi = int(np.searchsorted(ts_arr, w_end, side="left"))
            if hi - lo >= self.min_msgs:
                feats = extract_window_features(lo, hi, arrays, self.baseline)
                if feats is not None:
                    true_label = feats.get("label")
                    pred_class, conf, probs = self._predict_window(feats)
                    feat_only = {k: v for k, v in feats.items()
                                 if k not in ("label", "window_start", "window_end")}
                    yield WindowPrediction(
                        window_index=widx,
                        window_start=feats["window_start"],
                        window_end=feats["window_end"],
                        predicted_class=pred_class,
                        confidence=conf,
                        class_probabilities=probs,
                        true_label=true_label,
                        features=feat_only,
                    )
                    widx += 1
            w_start += self.stride_s
            if realtime_factor > 0:
                time.sleep(self.stride_s * realtime_factor)

    def stream_file(self, file_key: str, realtime_factor: float = 0.0, max_messages: Optional[int] = None) -> Iterator[WindowPrediction]:
        """Load a raw capture by config key (normal/dos/fuzzy/gear/rpm) and replay it."""
        raw_dir = Path(self.cfg["paths"]["raw_dir"])
        fname = self.cfg["dataset"]["files"][file_key]
        attack_names = {"normal": "Normal", "dos": "DoS", "fuzzy": "Fuzzy", "gear": "Gear", "rpm": "RPM"}
        df = load_and_label_file(raw_dir / fname, attack_names[file_key], self.cfg["dataset"]["raw_columns"])
        if max_messages is not None:
            df = df.iloc[:max_messages].reset_index(drop=True)
        yield from self.stream_dataframe(df, realtime_factor=realtime_factor)


def main(config_path: str = "configs/config.yaml", file_key: str = "dos", max_messages: int = 60000):
    cfg = load_config(config_path)
    sim = CANStreamSimulator(cfg)
    print(f"[sim] replaying {file_key} (first {max_messages} msgs), baseline reused as-is "
          f"({len(sim.baseline.known_ids)} known IDs, not refit)")

    counts, correct, total = {}, 0, 0
    for wp in sim.stream_file(file_key, max_messages=max_messages):
        counts[wp.predicted_class] = counts.get(wp.predicted_class, 0) + 1
        if wp.true_label is not None:
            total += 1
            correct += int(wp.predicted_class == wp.true_label)
        if wp.window_index < 5 or (wp.predicted_class != "Normal" and wp.window_index % 200 == 0):
            print(f"  win {wp.window_index:5d} @ {wp.window_start:.3f}s -> {wp.predicted_class:7s} "
                  f"conf={wp.confidence:.3f}  (true={wp.true_label})")

    print(f"\n[sim] windows by predicted class: {counts}")
    if total:
        print(f"[sim] streaming accuracy vs replayed ground truth: {correct}/{total} = {correct/total:.4f}")
    print("[done] Phase 9 stream simulation complete.")


if __name__ == "__main__":
    args = sys.argv[1:]
    cfg_path = args[0] if len(args) > 0 else "configs/config.yaml"
    fkey = args[1] if len(args) > 1 else "dos"
    main(cfg_path, fkey)
