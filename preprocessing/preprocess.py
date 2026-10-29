"""
preprocess.py
==============
End-to-end preprocessing for the CAN IDS pipeline:
  1. Load raw HCRL Car-Hacking CSVs. The published dataset actually ships
     TWO different line formats, not one:
       - normal_run_data.csv: keyword-tagged, whitespace-delimited --
         "Timestamp: 1479121434.850202 ID: 0350 000 DLC: 8 05 28 84 66 6d 00 00 a2"
       - {DoS,Fuzzy,gear,RPM}_dataset.csv: plain comma-delimited --
         "1478195721.903877,0545,8,d8,00,00,8a,00,00,00,00,R"
     Both are handled by auto-detecting format per line (see
     _load_ragged_csv below).
  2. Label each message: 'R' -> Normal, 'T' -> that file's attack name.
  3. Fit the IDBaseline on normal traffic only, persist it (Phase 9 reuses
     this exact artifact rather than refitting online).
  4. Run feature_extractor.build_feature_matrix() PER SOURCE FILE (capture
     streams are independent; never window across concatenated files).
  5. Concatenate windowed features, label-encode, balance, split, persist.

Inputs
------
Raw CSVs at paths given in configs/config.yaml -> dataset.files. Column
order/names in configs/config.yaml -> dataset.raw_columns describe the
STANDARDIZED output schema this loader produces, not a fixed input
format -- the two real input formats above are handled separately.

Outputs
-------
data/processed/{train,val,test}.parquet
models/label_encoder.json
models/id_baseline.pkl

Dependencies
------------
numpy, pandas, pyyaml, scikit-learn, pyarrow (for parquet)
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from feature_extractor import IDBaseline, build_feature_matrix, load_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("preprocess")

STANDARD_COLUMNS = ["timestamp", "can_id", "dlc"] + [f"d{i}" for i in range(8)] + ["flag"]
MAX_DLC = 8


# ------------------------------------------------------------------ #
# Loading + cleaning
# ------------------------------------------------------------------ #
def _parse_keyword_line(tokens: list[str]) -> list | None:
    """
    Parse one line of the normal_run_data.csv keyword-tagged format:
        Timestamp: <float> ID: <hex_id> <unused> DLC: <int> <hex byte>...
    Returns [timestamp, can_id, dlc, d0..d7, flag=None] or None if the
    line doesn't match this format / is malformed.
    """
    try:
        ts_idx = tokens.index("Timestamp:")
        id_idx = tokens.index("ID:")
        dlc_idx = tokens.index("DLC:")
    except ValueError:
        return None  # missing a required keyword -> not this format

    try:
        timestamp = float(tokens[ts_idx + 1])
        can_id = tokens[id_idx + 1]
        dlc = int(tokens[dlc_idx + 1])
    except (IndexError, ValueError):
        return None

    if not (0 <= dlc <= MAX_DLC):
        return None

    payload_tokens = tokens[dlc_idx + 2: dlc_idx + 2 + dlc]
    if len(payload_tokens) < dlc:
        return None  # truncated line -- fewer payload bytes than DLC claims

    try:
        payload = [int(b, 16) for b in payload_tokens]  # payload bytes are HEX
    except ValueError:
        return None

    payload = payload + [np.nan] * (MAX_DLC - len(payload))
    return [timestamp, can_id, dlc] + payload + [None]  # this format carries no R/T flag


def _parse_csv_line(parts: list[str]) -> list | None:
    """
    Parse one line of the attack-file comma-delimited format:
        <float>,<hex_id>,<int dlc>,<hex byte>,...,<hex byte>,<R|T>
    Returns [timestamp, can_id, dlc, d0..d7, flag] or None if malformed.
    """
    if len(parts) < 3:
        return None
    try:
        timestamp = float(parts[0])
        can_id = parts[1]
        dlc = int(parts[2])
    except ValueError:
        return None

    if not (0 <= dlc <= MAX_DLC):
        return None

    remainder = parts[3:]
    flag = None
    if remainder and remainder[-1] in ("R", "T"):
        flag = remainder[-1]
        remainder = remainder[:-1]

    payload_tokens = remainder[:dlc]
    if len(payload_tokens) < dlc:
        return None  # truncated row -- fewer payload bytes than DLC claims

    try:
        payload = [int(b, 16) for b in payload_tokens]  # payload bytes are HEX
    except ValueError:
        return None

    payload = payload + [np.nan] * (MAX_DLC - len(payload))
    return [timestamp, can_id, dlc] + payload + [flag]


def _load_ragged_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """
    Load one raw HCRL capture file into the standardized schema
    (timestamp, can_id, dlc, d0..d7, flag), auto-detecting which of the
    two known HCRL line formats each line uses:
      - keyword-tagged (normal_run_data.csv): "Timestamp: ... ID: ... DLC: ..."
      - comma-delimited (attack files): "<ts>,<id>,<dlc>,<bytes...>,<R|T>"
    Detection is per-LINE (not per-file) so a file is never wrongly
    rejected wholesale if it turns out to mix formats; blank lines are
    skipped, and lines that match neither format are skipped with a
    logged warning rather than crashing the whole load.

    `columns` is accepted for interface compatibility with the
    configs/config.yaml -> dataset.raw_columns declaration (which
    documents the standardized OUTPUT schema) but does not drive parsing,
    since the two real input formats have fundamentally different token
    layouts that can't be described by a single column list.
    """
    keyword_count = 0
    csv_count = 0
    skipped_count = 0
    rows = []

    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # blank line, skip safely

            row = None
            if "Timestamp:" in line:
                row = _parse_keyword_line(line.split())
                if row is not None:
                    keyword_count += 1
            else:
                parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                row = _parse_csv_line(parts)
                if row is not None:
                    csv_count += 1

            if row is None:
                skipped_count += 1
                if skipped_count <= 5:  # cap per-file spam, the total is still logged
                    logger.warning(f"[load][{path.name}] line {line_no}: unparsable, skipped -> {line[:80]!r}")
                continue

            rows.append(row)

    if not rows:
        raise ValueError(f"{path}: no parsable rows found (checked {line_no} lines) -- check file format/path")

    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    # Safety-net numeric coercion for the decimal fields only -- payload bytes
    # are already Python ints from explicit hex parsing above and must NOT be
    # re-coerced with pd.to_numeric's base-10 assumption (that was the original
    # bug: hex bytes like "6d"/"8a" would silently become NaN under base-10).
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["timestamp", "can_id", "dlc"]).reset_index(drop=True)
    dropped_bad_numeric = before - len(df)

    logger.info(
        f"[load][{path.name}] format detected: keyword={keyword_count} rows, csv={csv_count} rows "
        f"({'mixed -- unexpected, verify source file' if keyword_count and csv_count else 'consistent'}); "
        f"skipped_unparsable={skipped_count}, dropped_bad_numeric={dropped_bad_numeric}, "
        f"final={len(df)} rows"
    )
    if keyword_count and csv_count:
        logger.warning(
            f"[load][{path.name}] file contains BOTH known formats -- this is unexpected for HCRL "
            f"Car-Hacking files; double-check {path} wasn't concatenated from multiple sources."
        )
    return df


def load_and_label_file(path: Path, attack_name: str, columns: list[str]) -> pd.DataFrame:
    """Load one raw capture and assign per-message ground-truth labels."""
    df = _load_ragged_csv(path, columns)
    if df["flag"].isna().all():
        # normal_run_data.csv: no flag column, every row is Normal
        df["label"] = "Normal"
    else:
        df["label"] = np.where(df["flag"] == "T", attack_name, "Normal")
    df = df.drop(columns=["flag"])
    return df


# ------------------------------------------------------------------ #
# Balancing
# ------------------------------------------------------------------ #
def balance_classes(df: pd.DataFrame, majority_class: str, multiplier: float, seed: int) -> pd.DataFrame:
    """
    Undersample `majority_class` down to `multiplier` * mean(other class counts).
    Leaves minority (attack) classes untouched.
    """
    counts = df["label"].value_counts()
    other_counts = counts.drop(labels=[majority_class], errors="ignore")
    if other_counts.empty:
        return df
    target_n = int(other_counts.mean() * multiplier)
    target_n = min(target_n, counts.get(majority_class, 0))

    majority_rows = df[df["label"] == majority_class]
    minority_rows = df[df["label"] != majority_class]
    if len(majority_rows) > target_n:
        majority_rows = majority_rows.sample(n=target_n, random_state=seed)
    return pd.concat([majority_rows, minority_rows], ignore_index=True)


# ------------------------------------------------------------------ #
# Split (Section 8 leakage fix)
# ------------------------------------------------------------------ #
def assign_group_ids(frame: pd.DataFrame, block_seconds: float) -> pd.Series:
    """
    Build the grouping key used to prevent train/val/test overlap leakage.

    Windows are 100ms wide with a 50ms stride, so every window overlaps ~50%
    with each temporal neighbour. A plain row-wise split scatters these
    near-duplicate windows across splits, so the "held-out" set is barely held
    out (the Section 8 audit measured 98.8% of test windows having an
    overlapping neighbour in train). We instead bucket each window into a
    contiguous, non-overlapping time block PER SOURCE FILE and later assign whole
    blocks to a single split -- an overlapping neighbour is at most one block
    away, and whole blocks never straddle splits, so overlap can no longer leak
    across the train/val/test boundary.

    `window_start` is a wall-clock epoch timestamp, so we subtract each file's own
    first window_start to get file-relative seconds before flooring into blocks.
    """
    file_start = frame.groupby("source_file")["window_start"].transform("min")
    block_idx = ((frame["window_start"] - file_start) // block_seconds).astype(int)
    return frame["source_file"].astype(str) + "_blk" + block_idx.astype(str)


def _grouped_train_val_test(
    frame: pd.DataFrame, split_cfg: dict, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leakage-free split: `group_id` (source_file + time block) is the atomic unit,
    so no group's windows are ever split across train/val/test. Groups are
    stratified by each group's dominant window label to keep all classes present
    in every split; stratification degrades to unstratified (rather than crashing)
    if any dominant-label class has fewer than 2 groups.
    """
    dominant = frame.groupby("group_id")["label"].agg(lambda s: s.value_counts().idxmax())
    group_ids = dominant.index.to_numpy()
    group_dom = dominant.to_numpy()
    dom_by_group = dict(zip(group_ids, group_dom))

    def _safe(labels: np.ndarray) -> np.ndarray | None:
        if not split_cfg["stratify"]:
            return None
        _, counts = np.unique(labels, return_counts=True)
        if (counts < 2).any():
            print("[split][warn] a group-dominant class has <2 groups; disabling stratification for this split")
            return None
        return labels

    train_g, temp_g = train_test_split(
        group_ids, test_size=(1 - split_cfg["train"]), stratify=_safe(group_dom), random_state=seed
    )
    temp_dom = np.array([dom_by_group[g] for g in temp_g])
    val_ratio_of_temp = split_cfg["val"] / (split_cfg["val"] + split_cfg["test"])
    val_g, test_g = train_test_split(
        temp_g, test_size=(1 - val_ratio_of_temp), stratify=_safe(temp_dom), random_state=seed
    )

    train_set, val_set, test_set = set(train_g), set(val_g), set(test_g)
    print(f"[split] groups: train={len(train_set)} val={len(val_set)} test={len(test_set)} (total {len(group_ids)})")
    return (
        frame[frame["group_id"].isin(train_set)].reset_index(drop=True),
        frame[frame["group_id"].isin(val_set)].reset_index(drop=True),
        frame[frame["group_id"].isin(test_set)].reset_index(drop=True),
    )


def _random_train_val_test(
    frame: pd.DataFrame, split_cfg: dict, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    LEGACY ungrouped row-wise stratified split (the leaky Section 8 behaviour).
    Retained ONLY so the before/after leakage inflation can be measured directly;
    never deploy a model evaluated under this strategy.
    """
    def _safe_stratify(f: pd.DataFrame, min_per_class: int = 2) -> pd.Series | None:
        if not split_cfg["stratify"]:
            return None
        counts = f["label_encoded"].value_counts()
        if (counts < min_per_class).any():
            too_few = counts[counts < min_per_class].index.tolist()
            print(f"[split][warn] classes {too_few} have <{min_per_class} rows; disabling stratification for this split")
            return None
        return f["label_encoded"]

    train_df, temp_df = train_test_split(
        frame, test_size=(1 - split_cfg["train"]), stratify=_safe_stratify(frame), random_state=seed
    )
    val_ratio_of_temp = split_cfg["val"] / (split_cfg["val"] + split_cfg["test"])
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_of_temp), stratify=_safe_stratify(temp_df), random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ------------------------------------------------------------------ #
# Full pipeline
# ------------------------------------------------------------------ #
def run_pipeline(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    files = cfg["dataset"]["files"]
    columns = cfg["dataset"]["raw_columns"]
    seed = cfg["random_seed"]

    # 1. Load + label every capture file
    file_attack_names = {
        "normal": "Normal",
        "dos": "DoS",
        "fuzzy": "Fuzzy",
        "gear": "Gear",
        "rpm": "RPM",
    }
    loaded = {}
    for key, fname in files.items():
        fpath = raw_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected raw file at {fpath}. Download the HCRL Car-Hacking dataset "
                f"into {raw_dir}/ before running preprocess.py."
            )
        loaded[key] = load_and_label_file(fpath, file_attack_names[key], columns)
        print(f"[load] {fname}: {len(loaded[key])} messages")

    # 2. Fit baseline on normal traffic only, persist it
    baseline_seconds = cfg["baseline"]["baseline_seconds"]
    baseline = IDBaseline.fit(loaded["normal"], baseline_seconds)
    with open(models_dir / "id_baseline.pkl", "wb") as f:
        pickle.dump(baseline, f)
    print(f"[baseline] known_ids={len(baseline.known_ids)} calibrated_ids={len(baseline.mean_iat_by_id)}")

    # Leakage fix: the calibration slice (first `baseline_seconds` of normal_run_data.csv)
    # was used to FIT the baseline, so it must never also become a train/val/test window --
    # otherwise the baseline is partially calibrated on data that reappears in val/test,
    # which is temporal leakage. Drop that slice from the normal capture before windowing.
    normal_cutoff = loaded["normal"]["timestamp"].iloc[0] + baseline_seconds
    remaining_normal = loaded["normal"][loaded["normal"]["timestamp"] > normal_cutoff].reset_index(drop=True)
    if remaining_normal.empty:
        capture_span = loaded["normal"]["timestamp"].iloc[-1] - loaded["normal"]["timestamp"].iloc[0]
        raise ValueError(
            f"normal_run_data.csv spans only {capture_span:.1f}s, which is <= "
            f"baseline.baseline_seconds ({baseline_seconds}s) in config.yaml. Nothing would "
            f"be left to window after excluding the calibration slice. Lower baseline_seconds "
            f"or use a longer normal capture."
        )
    loaded["normal"] = remaining_normal
    print(f"[leakage-guard] dropped calibration slice (<= {normal_cutoff:.2f}s) from normal capture before windowing")

    # 3. Window-extract features PER FILE, then concatenate.
    # Tag each frame with its source file BEFORE concat: source-file identity is
    # required by the grouped split (Section 8 fix) and was previously discarded here.
    feature_frames = []
    for key, df in loaded.items():
        feats = build_feature_matrix(df, baseline, cfg)
        feats["source_file"] = key
        print(f"[windowing] {files[key]}: {len(feats)} windows")
        feature_frames.append(feats)
    all_features = pd.concat(feature_frames, ignore_index=True)
    all_features = all_features.dropna(subset=["label"]).reset_index(drop=True)
    print(f"[merge] total windows: {len(all_features)}")
    print(all_features["label"].value_counts())

    # 4. Label encode (on the FULL window table, before any split/balance so the
    #    encoder sees every class regardless of how blocks fall into splits).
    encoder = LabelEncoder()
    all_features["label_encoded"] = encoder.fit_transform(all_features["label"])
    with open(models_dir / "label_encoder.json", "w") as f:
        json.dump({int(i): cls for i, cls in enumerate(encoder.classes_)}, f, indent=2)
    print(f"[encode] classes: {dict(enumerate(encoder.classes_))}")

    # 5. Split -- STRATEGY-DRIVEN (Section 8 leakage fix).
    # Grouping/splitting happens on the FULL, UNBALANCED window table so that group
    # atomicity (no overlapping-neighbour leakage) is established before any row-level
    # resampling can perturb it. Balancing is applied afterwards to the TRAIN split
    # only -- val/test stay untouched, so held-out metrics are honest.
    split_cfg = cfg["split"]
    strategy = split_cfg.get("strategy", "grouped_by_time_block")
    if strategy == "grouped_by_time_block":
        all_features["group_id"] = assign_group_ids(all_features, split_cfg.get("block_seconds", 10))
        n_groups = all_features["group_id"].nunique()
        print(f"[split] strategy=grouped_by_time_block block_seconds={split_cfg.get('block_seconds', 10)} -> {n_groups} groups")
        train_df, val_df, test_df = _grouped_train_val_test(all_features, split_cfg, seed)
    elif strategy == "random":
        print("[split][warn] strategy=random -- LEGACY leaky split, for before/after comparison only, do NOT deploy")
        train_df, val_df, test_df = _random_train_val_test(all_features, split_cfg, seed)
    else:
        raise ValueError(f"Unknown split.strategy={strategy!r}; expected 'grouped_by_time_block' or 'random'")

    # 6. Balance the TRAIN split only (never val/test).
    bal_cfg = cfg.get("balance", {})
    train_df = balance_classes(
        train_df,
        majority_class=bal_cfg.get("majority_class", "Normal"),
        multiplier=bal_cfg.get("multiplier", 3.0),
        seed=seed,
    )
    print(f"[balance] train windows after balancing: {len(train_df)}")
    print(train_df["label"].value_counts())

    # 7. Persist. Drop bookkeeping columns from the feature list, but KEEP them in the
    #    parquet files so downstream leakage/contamination checks and the simulator can
    #    use window_start/source_file.
    non_feature = ("label", "label_encoded", "window_start", "window_end", "source_file", "group_id")
    feature_cols = [c for c in train_df.columns if c not in non_feature]
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)
    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(
        f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"-> saved to {processed_dir}/"
    )


if __name__ == "__main__":
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")