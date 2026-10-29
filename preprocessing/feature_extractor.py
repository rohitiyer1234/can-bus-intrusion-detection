

"""
feature_extractor.py
=====================
Converts a cleaned, timestamp-sorted CAN message stream into a windowed
feature matrix for the CAN IDS pipeline.

PERFORMANCE NOT (this version)
--------------------------------
The original implementation called `df.iloc[lo:hi]` per window (allocates
a new DataFrame every window) and then did pandas groupby/value_counts/
sort_values/fillna/to_numpy INSIDE the window loop -- worst offender was
compute_hamming_by_id, which ran a fresh `window.groupby("can_id")` plus a
`fillna(0).to_numpy()` conversion for every ID *within* every window. On
the full HCRL dataset (~16.6M messages, tens of thousands of 100ms/50ms
windows), that per-window pandas overhead dominates runtime.

This version converts the input DataFrame to plain NumPy arrays ONCE per
source file (`_prepare_arrays`), then every window operates on array
VIEWS (`arr[lo:hi]`, O(1), no copy) using vectorized NumPy ops instead of
pandas method calls. Hamming distance is computed via a lexsort + a
256-entry popcount lookup table instead of a Python-level groupby +
`bin(x).count('1')` loop. See the benchmark this file's authors ran
(preprocessing/feature_extractor_benchmark.py) for measured numbers.

Design
------
- Time-based sliding window (default 100ms window / 50ms stride), NOT
  frame-count based, so features are comparable between idle and busy
  bus states and translate directly to the real-time replay engine
  (simulation/can_stream_simulator.py) without re-derivation.
- A one-time "baseline" pass over normal-only traffic builds a per-ID
  reference table (expected inter-arrival time, known-ID set) used by
  the periodicity-deviation and unseen-ID features. This mirrors what
  the live system would do: calibrate once on trusted traffic, then
  score incoming windows against that baseline.
- Every feature in this file corresponds 1:1 to a row in the Phase 3
  feature table. Do not add/remove features without updating configs/
  config.yaml and the SHAP feature-name list in explainability/.
  Feature DEFINITIONS are unchanged from the pre-optimization version --
  only how they're computed changed.

Inputs
------
A pandas DataFrame with columns (post preprocess.py cleaning):
    timestamp (float, seconds) | can_id (str, hex, no '0x' prefix)
    | dlc (int) | d0..d7 (int, 0-255, NaN if unused) | label (str)

Outputs
-------
A pandas DataFrame, one row per window, with the 25 engineered
features + a majority-vote 'label' column for that window.

Dependencies
------------
numpy, pandas, pyyaml
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("feature_extractor")

PAYLOAD_COLS = [f"d{i}" for i in range(8)]
MAX_DLC = 8

# 256-entry popcount lookup table, built once at import time. Replaces the
# Python-level `bin(x).count('1')` loop (called per-byte, per-consecutive-pair,
# per-ID, per-window in the old implementation) with a single vectorized
# NumPy array-index lookup.
POPCOUNT_TABLE = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


# ------------------------------------------------------------------ #
# Baseline (calibration) table -- unchanged, only runs once per file
# ------------------------------------------------------------------ #
@dataclass
class IDBaseline:
    """Per-CAN-ID reference statistics learned from normal-only traffic."""

    known_ids: set[str] = field(default_factory=set)
    mean_iat_by_id: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, normal_df: pd.DataFrame, baseline_seconds: float) -> "IDBaseline":
        df = normal_df[normal_df["timestamp"] <= normal_df["timestamp"].iloc[0] + baseline_seconds]
        known_ids = set(df["can_id"].unique())
        mean_iat_by_id: dict[str, float] = {}
        for can_id, group in df.groupby("can_id"):
            ts = group["timestamp"].sort_values().to_numpy()
            if len(ts) >= 2:
                mean_iat_by_id[can_id] = float(np.mean(np.diff(ts)))
        return cls(known_ids=known_ids, mean_iat_by_id=mean_iat_by_id)


# ------------------------------------------------------------------ #
# Shared small helpers -- pure NumPy, no pandas
# ------------------------------------------------------------------ #
def _entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy in bits from a NumPy array of category counts."""
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


@dataclass
class StreamArrays:
    """
    Whole-file NumPy arrays computed ONCE by _prepare_arrays(), then sliced
    (as O(1) views, not copies) per window during the sliding-window loop.
    """

    ts: np.ndarray                 # float64 (N,)
    id_codes: np.ndarray           # int32 (N,) -- factorized can_id
    id_uniques: np.ndarray         # object (K,) -- code -> original hex string
    dlc: np.ndarray                # int16 (N,)
    payload_raw: np.ndarray        # float64 (N, 8), NaN where byte absent (dlc < 8)
    payload_filled: np.ndarray     # uint8 (N, 8), NaN -> 0 -- computed ONCE, not per window
    label_codes: Optional[np.ndarray]    # int32 (N,) or None if no label column
    label_uniques: Optional[np.ndarray]  # object (L,) or None
    normal_code: Optional[int]           # index into label_uniques for "Normal", or None


def _prepare_arrays(df: pd.DataFrame) -> StreamArrays:
    """
    Convert the input DataFrame to plain NumPy arrays ONCE per source file.
    This is the fix for requirements 4/5 (no repeated per-window/per-group
    DataFrame allocation or fillna/to_numpy conversion): payload_filled in
    particular used to be recomputed via `.fillna(0).to_numpy()` for every
    CAN ID within every window; here it's computed a single time for the
    whole file.
    """
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    id_codes, id_uniques = pd.factorize(df["can_id"].to_numpy())
    id_codes = id_codes.astype(np.int32)
    dlc = df["dlc"].to_numpy(dtype=np.int16)

    payload_raw = df[PAYLOAD_COLS].to_numpy(dtype=np.float64)  # NaN preserved
    payload_filled = np.nan_to_num(payload_raw, nan=0.0).astype(np.uint8)  # computed ONCE

    if "label" in df.columns:
        label_codes, label_uniques = pd.factorize(df["label"].to_numpy())
        label_codes = label_codes.astype(np.int32)
        normal_matches = np.where(label_uniques == "Normal")[0]
        normal_code = int(normal_matches[0]) if len(normal_matches) else None
    else:
        label_codes, label_uniques, normal_code = None, None, None

    return StreamArrays(
        ts=ts, id_codes=id_codes, id_uniques=id_uniques, dlc=dlc,
        payload_raw=payload_raw, payload_filled=payload_filled,
        label_codes=label_codes, label_uniques=label_uniques, normal_code=normal_code,
    )


def _hamming_stats_vectorized(w_id: np.ndarray, w_ts: np.ndarray, w_payload_filled: np.ndarray) -> tuple[float, float, float]:
    """
    Mean/max/std Hamming distance between consecutive payloads of the SAME
    CAN ID within a window. Replaces the old per-ID pandas groupby with:
    one lexsort (group by id, ordered by time within group) + vectorized
    XOR + popcount-table lookup, all NumPy, no Python-level loops.
    """
    n = len(w_id)
    if n < 2:
        return 0.0, 0.0, 0.0

    order = np.lexsort((w_ts, w_id))  # primary key: w_id, secondary: w_ts
    sorted_id = w_id[order]
    sorted_payload = w_payload_filled[order]

    xor = np.bitwise_xor(sorted_payload[1:], sorted_payload[:-1])   # (n-1, 8)
    popcounts = POPCOUNT_TABLE[xor].sum(axis=1)                     # (n-1,) vectorized lookup
    same_id_mask = sorted_id[1:] == sorted_id[:-1]

    valid = popcounts[same_id_mask]
    if valid.size == 0:
        return 0.0, 0.0, 0.0
    valid = valid.astype(np.float64)
    return float(valid.mean()), float(valid.max()), float(valid.std())


def extract_window_features(lo: int, hi: int, arrays: StreamArrays, baseline: IDBaseline) -> Optional[dict]:
    """
    Compute the 25 features for a single window, given as an index range
    [lo, hi) into the precomputed whole-file arrays. No DataFrame slicing,
    no per-window fillna/to_numpy/groupby -- every step below is a view or
    a vectorized NumPy op over the window slice.

    Deliberately takes no attack-target-ID knowledge as input (see design
    review notes retained from the prior version): CAN arbitration IDs are
    not standardized across vehicles, so hardcoding "this ID is the gear
    signal" only holds for the HCRL capture vehicle. max_id_msg_rate is
    the generic (no-prior-knowledge) replacement for that removed feature.
    """
    n = hi - lo
    if n == 0:
        return None

    w_ts = arrays.ts[lo:hi]
    w_id = arrays.id_codes[lo:hi]
    w_dlc = arrays.dlc[lo:hi]
    w_payload_raw = arrays.payload_raw[lo:hi]
    w_payload_filled = arrays.payload_filled[lo:hi]

    iat = np.diff(w_ts) if n > 1 else np.zeros(1)
    duration = max(float(w_ts[-1] - w_ts[0]), 1e-6)

    unique_codes, counts = np.unique(w_id, return_counts=True)
    dominant_idx = int(np.argmax(counts))
    dominant_count = int(counts[dominant_idx])

    # --- A. timing / periodicity ---
    mean_iat = float(iat.mean())
    std_iat = float(iat.std())
    min_iat = float(iat.min())
    max_iat = float(iat.max())
    msg_rate = n / duration
    burstiness = float(iat.var() / mean_iat) if mean_iat > 0 else 0.0

    top5_order = np.argsort(-counts)[:5]
    dev_list = []
    for idx in top5_order:
        code = unique_codes[idx]
        can_id_str = arrays.id_uniques[code]
        if can_id_str not in baseline.mean_iat_by_id:
            continue
        mask = w_id == code
        sub_ts = w_ts[mask]
        if sub_ts.size >= 2:
            obs_mean = float(np.diff(np.sort(sub_ts)).mean())
            dev_list.append(abs(obs_mean - baseline.mean_iat_by_id[can_id_str]))
    id_periodicity_dev = float(np.mean(dev_list)) if dev_list else 0.0

    # --- B. ID distribution / diversity ---
    unique_id_count = int(unique_codes.size)
    id_diversity_ratio = unique_id_count / n
    dominant_id_ratio = dominant_count / n
    id_entropy = _entropy_from_counts(counts)
    unseen_count = sum(1 for code in unique_codes if arrays.id_uniques[code] not in baseline.known_ids)
    unseen_id_ratio = unseen_count / unique_id_count if unique_id_count else 0.0

    # --- C. DLC / payload entropy ---
    dlc_mean = float(w_dlc.mean())
    dlc_std = float(w_dlc.std())
    dlc_counts = np.bincount(w_dlc.astype(np.int64), minlength=MAX_DLC + 1)
    dlc_entropy = _entropy_from_counts(dlc_counts)

    valid_bytes = w_payload_raw[~np.isnan(w_payload_raw)]
    if valid_bytes.size:
        byte_counts = np.bincount(valid_bytes.astype(np.int64), minlength=256)
        payload_byte_entropy = _entropy_from_counts(byte_counts)
        payload_mean = float(valid_bytes.mean())
        payload_std = float(valid_bytes.std())
    else:
        payload_byte_entropy = 0.0
        payload_mean = 0.0
        payload_std = 0.0
    zero_payload_ratio = float((w_payload_filled.sum(axis=1) == 0).mean())

    # --- D. payload statistics ---
    col0 = w_payload_raw[:, 0]
    col7 = w_payload_raw[:, 7]
    payload_byte0_mean = float(np.nanmean(col0)) if np.any(~np.isnan(col0)) else 0.0
    payload_byte7_mean = float(np.nanmean(col7)) if np.any(~np.isnan(col7)) else 0.0

    # --- E. per-ID Hamming distance ---
    mean_hamming, max_hamming, std_hamming = _hamming_stats_vectorized(w_id, w_ts, w_payload_filled)

    # --- F. compound ---
    max_id_msg_rate = float(dominant_count / duration)

    # --- label (majority vote), vectorized via bincount instead of pandas .mode() ---
    label = None
    # --- label: ANY-ATTACK-PRESENT (attack-priority), not majority vote ---
    # Majority vote was found to catastrophically collapse Gear/RPM spoofing windows:
    # those attacks inject at a normal-priority CAN ID that must share bus bandwidth
    # fairly with background traffic (unlike DoS, which floods the highest-priority
    # ID 0x000 and dominates the bus outright), so injected messages are typically
    # only ~20-25% of a window's traffic even during an active attack burst --
    # never a majority. Any-attack-present labeling (label = the attack type if ANY
    # attack message is present, else Normal) fixes this without changing feature
    # definitions. This also matches published methodology (e.g. Yagiz et al.,
    # "Transforming In-Vehicle Network Intrusion Detection", CPSIoTSec'24, label
    # windows by attack type "if any injected message is present"). If multiple
    # attack types appear in one window (not expected -- windows are built per
    # source file, so only one attack type per file, but handled defensively),
    # the most frequent attack type among the non-Normal messages wins.
    label = None
    if arrays.label_codes is not None:
        w_label = arrays.label_codes[lo:hi]
        label_counts = np.bincount(w_label, minlength=len(arrays.label_uniques))
        if arrays.normal_code is not None:
            label_counts[arrays.normal_code] = 0  # zero out Normal so argmax picks an attack class if any is present
        if label_counts.sum() > 0:
            label = str(arrays.label_uniques[int(np.argmax(label_counts))])
        else:
            label = "Normal"

    return {
        "mean_iat": mean_iat,
        "std_iat": std_iat,
        "min_iat": min_iat,
        "max_iat": max_iat,
        "msg_rate": msg_rate,
        "burstiness": burstiness,
        "id_periodicity_dev": id_periodicity_dev,
        "unique_id_count": unique_id_count,
        "id_diversity_ratio": id_diversity_ratio,
        "dominant_id_ratio": dominant_id_ratio,
        "id_entropy": id_entropy,
        "unseen_id_ratio": unseen_id_ratio,
        "dlc_mean": dlc_mean,
        "dlc_std": dlc_std,
        "dlc_entropy": dlc_entropy,
        "payload_byte_entropy": payload_byte_entropy,
        "zero_payload_ratio": zero_payload_ratio,
        "payload_mean": payload_mean,
        "payload_std": payload_std,
        "payload_byte0_mean": payload_byte0_mean,
        "payload_byte7_mean": payload_byte7_mean,
        "mean_hamming_same_id": mean_hamming,
        "max_hamming_same_id": max_hamming,
        "hamming_std_same_id": std_hamming,
        "max_id_msg_rate": max_id_msg_rate,
        "window_start": float(w_ts[0]),
        "window_end": float(w_ts[-1]),
        "label": label,
    }


# ------------------------------------------------------------------ #
# Sliding window driver
# ------------------------------------------------------------------ #
def build_feature_matrix(
    df: pd.DataFrame,
    baseline: IDBaseline,
    config: dict,
    progress_every: int = 5000,
) -> pd.DataFrame:
    """
    Slide a time-based window across a timestamp-sorted CAN stream and
    compute features for every window. Windows with fewer than
    min_msgs_per_window rows are skipped. Reports progress (windows
    processed, throughput, ETA) every `progress_every` windows.
    """
    window_s = config["windowing"]["window_ms"] / 1000.0
    stride_s = config["windowing"]["stride_ms"] / 1000.0
    min_msgs = config["windowing"]["min_msgs_per_window"]

    df = df.sort_values("timestamp").reset_index(drop=True)
    t_start, t_end = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]

    arrays = _prepare_arrays(df)  # ONE conversion for the whole file
    ts_arr = arrays.ts

    total_estimated = max(1, int((t_end - t_start) / stride_s))
    logger.info(
        f"[extract] {len(df)} messages, ~{total_estimated} windows estimated "
        f"(window={window_s*1000:.0f}ms, stride={stride_s*1000:.0f}ms)"
    )

    rows = []
    w_start = t_start
    n_processed = 0
    t0 = time.time()
    while w_start < t_end:
        w_end = w_start + window_s
        lo = np.searchsorted(ts_arr, w_start, side="left")
        hi = np.searchsorted(ts_arr, w_end, side="left")
        if hi - lo >= min_msgs:
            feats = extract_window_features(lo, hi, arrays, baseline)
            if feats is not None:
                rows.append(feats)
        w_start += stride_s
        n_processed += 1

        if progress_every and n_processed % progress_every == 0:
            elapsed = time.time() - t0
            rate = n_processed / elapsed if elapsed > 0 else 0.0
            remaining = max(0, total_estimated - n_processed)
            eta_s = remaining / rate if rate > 0 else 0.0
            logger.info(
                f"[extract] {n_processed}/{total_estimated} windows "
                f"({rate:.0f} windows/s, elapsed={elapsed:.1f}s, ETA={eta_s:.1f}s)"
            )

    elapsed = time.time() - t0
    logger.info(f"[extract] done: {len(rows)} windows kept out of {n_processed} scanned in {elapsed:.1f}s")
    return pd.DataFrame(rows)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Smoke test on synthetic data — real invocation happens from preprocess.py
    cfg = load_config()
    rng = np.random.default_rng(cfg["random_seed"])
    n = 2000
    synthetic = pd.DataFrame(
        {
            "timestamp": np.sort(rng.uniform(0, 5, n)),
            "can_id": rng.choice(["043", "0316", "000", "18F", "220"], size=n),
            "dlc": rng.integers(0, 9, n),
            **{f"d{i}": rng.integers(0, 256, n) for i in range(8)},
            "label": "Normal",
        }
    )
    baseline = IDBaseline.fit(synthetic, cfg["baseline"]["baseline_seconds"])
    features = build_feature_matrix(synthetic, baseline, cfg)
    print(f"Generated {len(features)} windows x {features.shape[1]} columns")
    print(features.head())