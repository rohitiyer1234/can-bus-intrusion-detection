# Explainable & LLM-Assisted CAN Bus Intrusion Detection

### A leakage-audited benchmark on the HCRL Car-Hacking dataset, with SHAP explanations and a fully local LLM incident reporter

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-XGBoost-orange)](https://github.com/dmlc/xgboost)
[![Explainability](https://img.shields.io/badge/explainability-TreeSHAP-green)](https://github.com/shap/shap)
[![LLM](https://img.shields.io/badge/LLM-Ollama%20%2B%20qwen3%3A4b-purple)](https://ollama.com/library/qwen3:4b)
[![Dashboard](https://img.shields.io/badge/dashboard-Streamlit-red)](https://github.com/streamlit/streamlit)
![Cloud](https://img.shields.io/badge/cloud%20dependencies-none-lightgrey)

> **Summer internship project, Center of Excellence (CoE) in e-Mobility, Department of Electronics and Communication Engineering, RV College of Engineering (RVCE), Bengaluru.**
>
> Authors: **Rohit R Iyer** and **Srividya P** (RVCE, ECE).
> Companion paper: *Explainable and LLM-Assisted CAN Bus Intrusion Detection: A Leakage-Audited Benchmark on the HCRL Car-Hacking Dataset* (IEEE Access-format manuscript).

---

## Table of contents

1. [Overview](#1-overview)
2. [Headline results](#2-headline-results)
3. [Why this project: the problem](#3-why-this-project-the-problem)
4. [Dataset and threat model](#4-dataset-and-threat-model)
5. [Pipeline architecture](#5-pipeline-architecture)
6. [Methodology in detail](#6-methodology-in-detail)
7. [Results in detail](#7-results-in-detail)
8. [Explainability (SHAP)](#8-explainability-shap)
9. [Local LLM incident reporting](#9-local-llm-incident-reporting)
10. [Stream simulator and dashboard](#10-stream-simulator-and-dashboard)
11. [Repository structure](#11-repository-structure)
12. [Installation](#12-installation)
13. [Running the project](#13-running-the-project)
14. [Configuration reference](#14-configuration-reference)
15. [LLM configuration reference](#15-llm-configuration-reference)
16. [Reproducing the paper's numbers](#16-reproducing-the-papers-numbers)
17. [Scope: what this work does and does not claim](#17-scope-what-this-work-does-and-does-not-claim)
18. [Limitations and future work](#18-limitations-and-future-work)
19. [Troubleshooting](#19-troubleshooting)
20. [References](#20-references)
21. [Acknowledgements](#21-acknowledgements)

---

## 1. Overview

The Controller Area Network (CAN) bus connects the 70 to 100+ electronic control units (ECUs) in a modern car, and it has **no authentication, no encryption and no sender identification**. Any node with bus access (a compromised infotainment unit, a telematics dongle, a device on the OBD-II port) can inject frames that every other ECU accepts as genuine.

Machine-learning intrusion detection systems (IDS) report near-perfect accuracy on public benchmarks, but two gaps persist:

1. **Leakage is rarely audited.** Most pipelines cut the message stream into *overlapping* sliding windows. A normal row-wise train/test split then places near-duplicate windows on both sides of the split.
2. **Alarms are not actionable.** A class label alone does not tell an analyst *why* a window was flagged, what mechanism it suggests, or what to do next.

This repository contains a **fully local, end-to-end pipeline** that addresses both gaps on the complete HCRL Car-Hacking dataset (~16.6 million CAN messages):

| Stage | What it does |
|---|---|
| Preprocessing | Dual-format parser, per-message labels, 100 ms / 50 ms sliding windows, 25 vehicle-agnostic behavioural features |
| Leakage-audited split | Grouped, non-overlapping 10 s time-block split, verified by two invariants |
| Detection | Class-balanced XGBoost (compared against Random Forest and LightGBM) |
| Explainability | Exact TreeSHAP, global and per-window, computed from the deployed booster |
| Incident reporting | Local LLM (Ollama + `qwen3:4b`) producing five-section reports, with a deterministic template fallback |
| Demonstration | Real-time stream simulator and an 11-panel Streamlit dashboard |

Everything runs on a laptop. **No cloud API is used at any stage and no data leaves the machine.**

---

## 2. Headline results

| Metric (held-out test, 33,247 windows) | Value |
|---|---|
| **Macro-F1** | **0.9990** |
| Accuracy | 0.9992 |
| Macro-precision | 0.9992 |
| Macro-recall | 0.9988 |
| Weighted-F1 | 0.9992 |
| Lowest per-class F1 (Gear) | 0.9983 |

**The leakage audit:**

| Invariant | Naive (row-wise) split | Grouped time-block split |
|---|---|---|
| Cross-partition group overlap | not controlled | **0** |
| Same-file < 100 ms test contamination `C` | **98.8 %** | **0.96 %** |
| Test macro-F1 (each on its own held-out test set) | 0.9990 | 0.9990 |

**Honest interpretation.** The grouped split removes the leakage structurally (98.8 % to 0.96 % contamination), yet the headline score is the same under both splits. On this dataset the leakage did **not** inflate the metric. The grouped split establishes *validity*, not a different number. This is reported as measured, not tuned (split strategy, seed and block size were never adjusted to move the score).

Other numbers worth knowing:

* 190,422 windows in total (123,928 balanced-train, 33,247 validation, 33,247 test).
* Best boosting iteration **114 of at most 300**, 575 trees persisted (115 rounds × 5 classes).
* Training time for the deployed XGBoost model on the balanced training set: ~3 s (validation-stage comparison).
* Incident report generation on an NVIDIA RTX 4070 (8 GB): low single-digit seconds per report.
* Representative DoS replay: **61.7 %** of emitted windows flagged as attack, consistent with the burst structure of the capture.

---

## 3. Why this project: the problem

* CAN was standardised in the 1980s for deterministic, priority-arbitrated, real-time delivery. It was never designed against an adversary.
* Koscher et al. (2010) and the Jeep Cherokee remote compromise by Miller and Valasek (2015) showed that CAN-level injection is enough for full vehicle takeover.
* CAN cannot be retrofitted with authentication without breaking compatibility with the installed ECU base, so **passive network-based intrusion detection** is the standard countermeasure.

The research gap this project targets: *to our knowledge, no prior work combines a leakage-audited CAN intrusion-detection benchmark with SHAP-grounded, fully local LLM incident reporting in a single reproducible pipeline* (positioning table in the paper, Table 1).

### Contributions

1. **Leakage-free evaluation methodology** for overlapping-window CAN IDS: characterise the leakage mode, introduce a grouped time-block split, verify it with two measured invariants.
2. **Reproducible benchmark** on the full HCRL Car-Hacking dataset with 25 generic behavioural features and XGBoost, measuring the effect of the leakage fix directly instead of assuming it.
3. **Complete local detection-to-response pipeline**: TreeSHAP attributions feed both a dashboard and a locally hosted LLM reporter, with a deterministic fallback.
4. **Explicit statement of scope**: what the results do *not* establish (cross-vehicle generalisation, physical hardware validation, architectural novelty).

---

## 4. Dataset and threat model

### Dataset

[HCRL Car-Hacking dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) (Seo, Song and Kim, 2018): CAN logs recorded from a **Hyundai YF Sonata** through the OBD-II port.

| Property | Value |
|---|---|
| Messages | ≈ 16.6 million |
| Classes | 5: Normal, DoS, Fuzzy, Gear spoofing, RPM spoofing |
| Ground truth | Per-message flag in the attack files: `R` (regular) or `T` (injected) |
| Attack structure | ≈ 300 discrete bursts of 3 to 5 s per attack file, interleaved with genuine traffic |
| Raw formats | **Two**: keyword-tagged whitespace-delimited (normal file) and comma-delimited (attack files) |

The loader detects the format **per line, not per file**, so mixed input can never be silently misparsed:

```text
normal_run_data : Timestamp: 1479121434.850202 ID: 0350 000 DLC: 8 05 28 84 66 6d 00 00 a2
attack files    : 1478195721.903877,0545,8,d8,00,00,8a,00,00,00,00,R
```

### Attack types

| Class | Mechanism |
|---|---|
| **DoS** | Floods the highest-priority ID (`0x000`) and monopolises arbitration; dominates its windows |
| **Fuzzy** | Injects randomised CAN IDs and payloads |
| **Gear** | Spoofed gear-state frames on a normal-priority ID |
| **RPM** | Spoofed engine-RPM frames on a normal-priority ID |

Gear and RPM spoofing frames contribute only about 20 to 25 % of an attack window's messages (Yagiz et al., 2024). They are never the majority, which motivates the window-labelling rule below.

### Threat model

An attacker with electrical access to the bus (a compromised ECU, a malicious OBD-II dongle, or a telematics unit with remote code execution) who can inject arbitrary frames with arbitrary timing. **Out of scope:** physical-layer attacks (bus jamming, voltage tampering) and attacks on cryptographic key material of higher-layer protocols such as AUTOSAR SecOC.

### Class distribution of the windowed dataset

![Class distribution](reports/figures/class_distribution.png)

| Class | Windows |
|---|---|
| Normal | 87,933 |
| Fuzzy | 31,659 |
| DoS | 28,105 |
| RPM | 21,404 |
| Gear | 21,321 |
| **Total** | **190,422** |

(Normal windows in the training split are undersampled, see [Section 6.4](#64-class-balancing-and-model-training).)

---

## 5. Pipeline architecture

```text
 Raw CAN logs (HCRL Car-Hacking)
            │
            ▼
 Preprocessing: dual-format parsing, per-message labels
            │
            ▼
 Feature extraction: 25 features per 100 ms window (50 ms stride)
            │
            ▼
 Grouped, non-overlapping 10 s time-block split (70 / 15 / 15)
            │
            ▼
 Class-balanced XGBoost training (compared with RF, LightGBM)
            │
            ▼
 Held-out evaluation (macro-F1, per-class metrics, ROC / PR)
            │
            ▼
 SHAP explanations (global and per-window)
            │
            ▼
 Local LLM incident report (Ollama + qwen3:4b, template fallback)
            │
            ▼
 Stream simulator + 11-panel Streamlit dashboard
```

Every stage **persists its own artifacts** (Parquet feature tables, the trained model, SHAP figures, metrics files), so any stage can be re-run or audited independently. **No stage refits on live data:** the per-ID timing baseline and the trained classifier are frozen after training and reused unchanged by the simulator and the dashboard.

---

## 6. Methodology in detail

### 6.1 Windowing

* Messages are parsed, labelled from the ground-truth flag, and sorted by timestamp **within each source file**. Windows are never formed across concatenated files, because each file is an independent recording session.
* Sliding window **W = 100 ms**, stride **S = 50 ms** (50 % overlap). Window *k* contains messages `{m_i : kS ≤ t_i < kS + W}`.
* Time-based windows keep features comparable between idle and busy bus states and let the *identical* extraction code run in the real-time simulator.
* Windows with fewer than **3 messages** are discarded as statistically unreliable.

### 6.2 The 25 vehicle-agnostic features

| Group | Features |
|---|---|
| **Timing (7)** | `mean_iat`, `std_iat`, `min_iat`, `max_iat`, `msg_rate`, `burstiness`, `id_periodicity_dev` |
| **ID distribution (5)** | `unique_id_count`, `id_diversity_ratio`, `dominant_id_ratio`, `id_entropy`, `unseen_id_ratio` |
| **DLC and entropy (5)** | `dlc_mean`, `dlc_std`, `dlc_entropy`, `payload_byte_entropy`, `zero_payload_ratio` |
| **Payload statistics (4)** | `payload_mean`, `payload_std`, `payload_byte0_mean`, `payload_byte7_mean` |
| **Hamming distance (3)** | `mean_hamming_same_id`, `max_hamming_same_id`, `hamming_std_same_id` (between consecutive same-ID payloads) |
| **Compound (1)** | `max_id_msg_rate` (message rate of the most frequent ID in the window) |

Shannon ID entropy:

```text
H_ID = − Σ_j  p_j · log2(p_j)        p_j = fraction of window messages carrying the j-th distinct ID
```

**Why "vehicle-agnostic" matters.** An earlier design used two features derived from vehicle-specific target arbitration IDs. They were **removed** after design review: CAN ID assignments are not standardised across vehicles, so such features amount to a lookup table for one capture. `max_id_msg_rate` replaces them without prior knowledge of any identifier. Feature groups can be toggled in `configs/config.yaml` under `features:`.

**Calibration baseline.** A one-time pass over the **first 60 s of normal-only traffic** builds a per-ID baseline (expected inter-arrival time and the set of known IDs) used by `id_periodicity_dev` and `unseen_id_ratio`. It is persisted to `models/id_baseline.pkl` and **never refit online**. The calibration slice is **excluded from all windowing**, so no training, validation or test window overlaps the data that fit the baseline (otherwise the baseline would be a second leakage channel).

### 6.3 Window labelling: "any attack present"

```text
            ⎧ argmax_{c ∈ A} n_{k,c}   if Σ_{c ∈ A} n_{k,c} > 0
   y_k  =   ⎨
            ⎩ Normal                    otherwise
```

`A` is the set of attack classes and `n_{k,c}` the number of class-*c* messages in window *k*. A window gets an attack label if **any** attack message is present, and the most frequent attack class wins when a window straddles several. Under majority-vote labelling, Gear and RPM windows (only 20 to 25 % attack messages) would collapse toward Normal; this rule corrects that without changing any feature definition. *(The sensitivity to this choice was not ablated, see [Limitations](#18-limitations-and-future-work).)*

### 6.4 Class balancing and model training

* Splitting is done on the **full, unbalanced** window table so group atomicity is established **before** any resampling.
* **Normal** is then undersampled **in the training partition only**, down to **3.0×** the mean attack-class count (`balance.multiplier`). Validation and test partitions are left untouched.
* The residual imbalance (about 3.8 : 1 Normal to Gear) is handled with inverse-frequency sample weights, applied to **both** training and validation partitions:

```text
w_c = N / (K · n_c)          N = #samples, K = 5 classes, n_c = count of class c
```

  (the `balanced` scheme of scikit-learn). Weighting the validation set matters: early stopping on an unweighted validation loss would be dominated by Normal and could favour a model that underperforms on minority attack classes.
* **Random Forest, XGBoost and LightGBM** are trained under identical weighted conditions and compared on the validation partition. The deployed architecture was **fixed in advance** to XGBoost, with the pre-registered rule that a competitor must beat it by more than **0.03** validation macro-F1 to override it. None did.
* **XGBoost settings:** `multi:softprob`, `max_depth 6`, `eta 0.1`, up to 300 rounds, early stopping with patience 20 on multiclass log-loss.
* **Early stopping uses an explicit `EarlyStopping(..., save_best=True)` callback.** A bare `early_stopping_rounds` argument halts training but *keeps the post-optimal rounds*; the callback prunes the persisted booster back to the best iteration.

### 6.5 The leakage audit and grouped split

With W = 100 ms and S = 50 ms, each window shares about half its raw messages with each immediate neighbour. A row-wise stratified split treats windows as independent rows and scatters near-duplicates across partitions.

**Contamination metric** of a test set 𝒯:

```text
C = (1 / |𝒯|) · Σ_{w ∈ 𝒯} c(w)

c(w) = 1 if the training partition contains a window w' from the SAME source file
         with |t_w − t_w'| < 100 ms, else 0
```

Under the naive split **C = 98.8 %**.

**The fix.** Windows are grouped into contiguous, **non-overlapping 10 s time blocks within each source file**. Each block is assigned **atomically** to one partition (70 / 15 / 15 at group level), stratified by the block's dominant window label. A window's temporal neighbours can fall in a different partition only at block boundaries.

**Two invariants verify the construction:**

1. Groups of train, validation and test are pairwise disjoint (confirmed on the full dataset).
2. Contamination `C` falls from **98.8 % to 0.96 %**; the residual is attributable to block-boundary effects.

Then the effect on the headline metric was measured by training and evaluating **both** the naive and the grouped pipelines end to end, each on its own held-out test partition. Both reach macro-F1 **0.9990**.

---

## 7. Results in detail

### 7.1 Model-family comparison (validation split, balanced sample weights)

| Model | Macro-F1 | Weighted-F1 | Train time (s) |
|---|---|---|---|
| Random Forest | 0.99911 | 0.99937 | 4.99 |
| **XGBoost (deployed)** | **0.99905** | **0.99931** | **3.02** |
| LightGBM | 0.99891 | 0.99922 | 4.88 |

All three lie within a 0.0002 macro-F1 band, so the **25-feature representation, not the learner, sets the ceiling**. Random Forest leads XGBoost by 0.00006, far below the 0.03 override margin. XGBoost is retained for training speed and best-iteration pruning.

### 7.2 Per-class held-out test performance (33,247 windows)

| Class | Precision | Recall | F1 | Support | ROC-AUC |
|---|---|---|---|---|---|
| DoS | 0.9986 | 0.9998 | 0.9992 | 4,249 | 1.0000 |
| Fuzzy | 0.9998 | 0.9985 | 0.9991 | 4,673 | 1.0000 |
| Gear | 0.9995 | 0.9970 | 0.9983 | 4,047 | 1.0000 |
| Normal | 0.9992 | 0.9999 | 0.9996 | 17,153 | 1.0000 |
| RPM | 0.9990 | 0.9990 | 0.9990 | 3,125 | 1.0000 |
| **Macro avg.** | **0.9992** | **0.9988** | **0.9990** | 33,247 | n/a |

Gear and RPM spoofing, which prior literature identifies as the hardest classes because injected messages never dominate their windows, both clear 0.998 F1. Full-precision values (including PR-AUC) are in `reports/metrics/evaluation_metrics.csv` and `reports/metrics/evaluation_summary.json`.

### 7.3 Composition of the partitions

| Class | Train (balanced) | Test |
|---|---|---|
| DoS | 19,560 | 4,249 |
| Fuzzy | 22,161 | 4,673 |
| Gear | 14,118 | 4,047 |
| Normal | 53,112 | 17,153 |
| RPM | 14,977 | 3,125 |
| **Total** | **123,928** | **33,247** |

Validation also holds 33,247 windows and is not modified by balancing.

### 7.4 Figures

**Normalised confusion matrix (held-out test)**

![Confusion matrix](reports/figures/eval_confusion_matrix.png)

**Per-class F1 (all classes exceed 0.998; bar labels are rounded to 1.00)**

![Per-class F1](reports/figures/eval_per_class_f1.png)

**One-vs-rest ROC curves (AUC ≈ 1.000 for every class)**

![ROC curves](reports/figures/eval_roc_curves.png)

**One-vs-rest precision-recall curves (AP ≈ 1.000 for every class)**

![PR curves](reports/figures/eval_pr_curves.png)

### 7.5 Training dynamics

Validation multiclass log-loss reaches its minimum of **0.00451 at round 114**; training stops after the 20-round patience window and the persisted model is pruned to the best iteration. Training and validation losses follow nearly identical trajectories.

| Training loss | Validation loss |
|---|---|
| ![Training loss](reports/figures/training_loss_curve.png) | ![Validation loss](reports/figures/validation_loss_curve.png) |

**Learning curve**

![Learning curve](reports/figures/learning_curve.png)

**Gain-based feature importance of the final booster.** Payload mean, byte-7 mean, `id_periodicity_dev` and `unseen_id_ratio` account for most of the total gain.

![Feature importance (gain)](reports/figures/feature_importance.png)

Raw loss history for every round: `reports/metrics/training_history.csv`.

---

## 8. Explainability (SHAP)

`shap.TreeExplainer` wraps the **deployed booster**, so global and local explanations derive from the actual model, not from a surrogate. For class *c*, the margin output decomposes additively:

```text
f_c(x) = φ_{0,c} + Σ_j φ_{j,c}(x)
```

A single function, `ShapExplainer.explain_prediction()` (in `explainability/shap_analysis.py`), returns the **top-k signed attributions** of a window together with the raw feature values. **Both the dashboard and the LLM prompt builder call it**, so the two consumers can never diverge.

### Global attributions (mean |SHAP|, stacked by class)

![SHAP global summary](reports/figures/shap_summary.png)

* `unseen_id_ratio` has the largest total attribution, mainly from **Fuzzy** (random identifiers absent from the normal-traffic baseline).
* `payload_byte7_mean` acts almost entirely on **Gear and RPM**.
* `dominant_id_ratio` acts on Gear, RPM and Normal.
* `id_periodicity_dev` is predominantly a **Normal** signal (legitimate ECU traffic is periodic).
* `payload_mean` and `zero_payload_ratio` act mainly on **DoS**.

**Gain vs SHAP.** Rankings differ: payload statistics lead in *gain* (they carry the early, coarse DoS split), while `unseen_id_ratio` leads in *mean |SHAP|*. They answer different questions. Features that rank high under both (`unseen_id_ratio`, `id_periodicity_dev`, byte-7 mean) are the most robust drivers of the model.

### Per-class SHAP summary plots

Each point is one window; horizontal position is the effect on the class margin and colour is the feature value (blue = low, red = high).

| DoS | Fuzzy | Gear |
|---|---|---|
| ![DoS](reports/figures/shap_class_DoS.png) | ![Fuzzy](reports/figures/shap_class_Fuzzy.png) | ![Gear](reports/figures/shap_class_Gear.png) |

| Normal | RPM |
|---|---|
| ![Normal](reports/figures/shap_class_Normal.png) | ![RPM](reports/figures/shap_class_RPM.png) |

**Reading the plots:**

* **DoS:** low `payload_mean` and high `zero_payload_ratio` push strongly toward DoS (flooding by high-priority all-zero frames).
* **Fuzzy:** high `unseen_id_ratio` is the dominant positive driver, supported by high ID entropy and unique-ID count and a low maximum per-ID rate.
* **Gear and RPM:** high `dominant_id_ratio` and `max_id_msg_rate` support the attack class; a high unseen-ID ratio argues *against* it, because spoofed frames reuse legitimate identifiers. The byte-7 mean acts in opposite directions for the two classes, so it discriminates Gear from RPM.
* **Normal:** low unseen-ID ratio, low dominant-ID ratio and low periodicity deviation support Normal; a high unseen-ID ratio is the strongest evidence against it.

---

## 9. Local LLM incident reporting

For each flagged window the reporter receives the **predicted class, its softmax confidence, and the top five signed SHAP features**, and produces a report with five fixed sections:

1. Incident Summary
2. Risk Level
3. Technical Analysis
4. Potential Impact
5. Recommended Mitigation

within a **200-word budget** (`llm.max_report_words`).

### Model and runtime

| Item | Value |
|---|---|
| Runtime | [Ollama](https://ollama.com), local HTTP API at `http://localhost:11434` |
| Model | `qwen3:4b` |
| Temperature | 0.2 |
| Request timeout | 60 s |
| Privacy | No cloud dependency, no data leaves the machine (matching automotive data-sensitivity practice) |
| Tested on | NVIDIA RTX 4070 (8 GB VRAM): the model loads fully on GPU and produces a report in low single-digit seconds |

### Three safety-relevant design decisions

**1. Grounded prompting.** The prompt restricts the model to the supplied class, confidence and features, and forbids invented CAN IDs, byte values or attacker identities.

**2. Programmatic risk level overrides the LLM.** During development, `qwen3:4b`'s free-form risk wording did not reliably track the confidence (e.g. writing "High" when the calibrated value was "Critical"). The risk label is therefore computed **in code** from the softmax confidence *p* and **overwrites** the model's own wording, so the dashboard caption and the report body can never disagree:

| Confidence *p* | Risk level |
|---|---|
| p ≥ 0.90 | Critical |
| 0.70 ≤ p < 0.90 | High |
| 0.50 ≤ p < 0.70 | Medium |
| p < 0.50 | Low (uncertain detection, verify before acting) |
| predicted class = Normal | Informational (always) |

**3. Deterministic template fallback (required).** If Ollama is unreachable (not running, model not pulled, HTTP error or timeout), `generate_report()` falls back to a fixed template with the *same five-section structure* that quotes the SHAP features verbatim. **The fallback is not a second LLM**, and the rest of the pipeline never depends on LLM availability. `generate_report()` never raises on Ollama being down; it returns `{"report", "source": "ollama" | "template", "risk_level"}`.

Other robustness details: `<think>…</think>` reasoning traces emitted by qwen3 are stripped before display, and reports are truncated to the word budget.

### Quick LLM smoke test (no Ollama needed)

```bash
python llm/incident_report.py
```

With Ollama stopped this prints a `source=template` report for a representative DoS detection; with Ollama running and `qwen3:4b` pulled it prints `source=ollama`.

---

## 10. Stream simulator and dashboard

### Stream simulator (`simulation/can_stream_simulator.py`)

Replays a recorded capture as a live message stream, scoring each window as it arrives. Two load-bearing design constraints:

* **No train/serve skew.** It uses the *same* extraction functions as training (`_prepare_arrays`, `extract_window_features`), driven incrementally.
* **Baseline reuse, never refit.** `models/id_baseline.pkl` is loaded as-is. Refitting on a live, possibly attacker-influenced stream would let the model gradually accept an ongoing attack as normal.

In a representative DoS replay on the RTX 4070 machine, **61.7 %** of emitted windows were flagged as attack windows.

### Streamlit dashboard (`dashboard/app.py`)

An **eleven-panel** dashboard containing **no analysis logic of its own**; it only imports and orchestrates the backing modules.

| # | Panel | # | Panel |
|---|---|---|---|
| 1 | Live CAN traffic feed | 7 | Training metrics |
| 2 | Current prediction | 8 | Loss curves |
| 3 | Attack status | 9 | SHAP explanations (Global / Local tabs) |
| 4 | Prediction confidence | 10 | AI incident report |
| 5 | Attack statistics | 11 | Historical attack timeline |
| 6 | Confusion matrix (held-out test) | | |

The sidebar lets you choose the replay scenario (`dos`, `fuzzy`, `gear`, `rpm`, `normal`) and the number of messages to replay (10,000 to 120,000, default 40,000), then press **Run detection stream**.

---

## 11. Repository structure

```text
.
├── configs/
│   └── config.yaml              # single source of truth for all stages
├── preprocessing/
│   ├── preprocess.py            # parsing, labelling, windowing, grouped split, balancing
│   └── feature_extractor.py     # 25 features, IDBaseline, window extraction
├── training/
│   └── train.py                 # model comparison, final XGBoost, training figures/metrics
├── evaluation/
│   └── evaluate.py              # held-out test metrics + eval_* figures
├── explainability/
│   └── shap_analysis.py         # ShapExplainer, global plots, explain_prediction()
├── llm/
│   ├── incident_report.py       # Ollama reporter + deterministic template fallback
│   └── incident_report_old.py   # earlier version (kept for reference)
├── simulation/
│   └── can_stream_simulator.py  # real-time replay through the training feature path
├── dashboard/
│   └── app.py                   # 11-panel Streamlit dashboard
├── models/
│   ├── xgb_model.json           # deployed booster (pruned to best iteration)
│   ├── feature_columns.json     # ordered list of the 25 features
│   ├── label_encoder.json       # {0: DoS, 1: Fuzzy, 2: Gear, 3: Normal, 4: RPM}
│   ├── id_baseline.pkl          # frozen per-ID timing / known-ID baseline
│   └── checkpoints/             # xgb_round_{25,50,75,100,125}.json audit trail
├── reports/
│   ├── figures/                 # all PNG figures used in this README
│   ├── metrics/                 # evaluation_*.csv/json, training_*.csv/json, model_comparison.csv
│   ├── paper/                   # RESULTS.md, CONTRIBUTION.md
│   ├── plotting.py              # shared publication-quality plotting helpers
│   ├── build_slides.py          # regenerates the internship deck from real metrics
│   └── CAN_IDS_Presentation.pptx
├── data/                        # NOT committed (git-ignored): raw/ interim/ processed/
├── maketree.py                  # helper that prints the project tree
├── oldcode.txt                  # notes / earlier code kept for reference
├── requirements.txt
├── .gitignore
└── README.md
```

`data/` and `logs/` are created at run time. The raw HCRL files are not redistributed here; download them from the dataset owners.

---

## 12. Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| Python | 3.11, 3.12 or 3.13 (developed and run on 3.13 and 3.11) |
| RAM | 8 GB or more recommended for full-dataset windowing |
| GPU | **Optional.** Only used by Ollama for the LLM stage; everything else is CPU |
| Ollama | **Optional.** Without it, reports use the template fallback |
| macOS only | `brew install libomp` (OpenMP runtime for XGBoost / LightGBM) |

### Steps

```bash
# 1. Clone
git clone https://github.com/rohitiyer1234/can-bus-intrusion-detection.git
cd can-bus-intrusion-detection

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Download the dataset

1. Download the **Car-Hacking Dataset** from the HCRL / OCSLab page: <https://ocslab.hksecurity.net/Datasets/car-hacking-dataset>
2. Place the five files in `data/raw/` using exactly the names expected by `configs/config.yaml`:

```text
data/
├── raw/
│   ├── normal_run_data.csv      # extract from normal_run_data.7z if you only have the archive
│   ├── DoS_dataset.csv
│   ├── Fuzzy_dataset.csv
│   ├── gear_dataset.csv
│   └── RPM_dataset.csv
├── interim/                     # created/used by the pipeline
└── processed/                   # train/val/test.parquet are written here by Step 1
```

The normal-traffic file is distributed as a `.7z` archive; extract it (for example with 7-Zip: `7z x normal_run_data.7z`) so that `normal_run_data.csv` sits in `data/raw/`. If your copy has a different name or extension, rename it to match `dataset.files.normal` in the config (or edit the config). Any extra helper files in `data/raw/` (for example a local line-counting script) are ignored by the pipeline. The loader handles the two raw line formats automatically.

---

## 13. Running the project

> **Run every command from the repository root.** The config uses relative paths (`data/raw`, `models`, `reports`).

All scripts take an optional config path as their first argument (default `configs/config.yaml`).

### Step 1: Preprocess (parse, label, window, extract features, grouped split)

```bash
python preprocessing/preprocess.py configs/config.yaml
```

Writes `data/processed/{train,val,test}.parquet` (with `group_id`, `window_start`, `window_end`, `source_file` columns kept for auditing), `models/label_encoder.json`, `models/feature_columns.json` and `models/id_baseline.pkl`.

### Step 2: Train (model comparison + final XGBoost)

```bash
python training/train.py configs/config.yaml
```

Writes `models/xgb_model.json`, `models/checkpoints/` (every 25 rounds), the training figures and `reports/metrics/training_*.{json,csv}` and `model_comparison.csv`.

### Step 3: Evaluate on the held-out test split

```bash
python evaluation/evaluate.py configs/config.yaml
```

Writes `reports/metrics/evaluation_summary.json`, `evaluation_metrics.csv` and the four `eval_*.png` figures. **The macro-F1 printed here is the headline number.**

### Step 4: SHAP explanations

```bash
python explainability/shap_analysis.py configs/config.yaml
```

Writes `reports/figures/shap_summary.png` and `shap_class_*.png` (computed on a deterministic 2,000-window sample of the test set) and prints an example local explanation.

### Step 5 (optional): Local LLM setup

```bash
# Install Ollama from https://ollama.com, then:
ollama pull qwen3:4b
ollama serve                 # skip if the Ollama desktop app / service is already running
curl http://localhost:11434/api/tags    # should list qwen3:4b
python llm/incident_report.py           # smoke test: expect source=ollama
```

### Step 6: Stream simulator (command line)

```bash
python simulation/can_stream_simulator.py configs/config.yaml dos
# scenario: dos | fuzzy | gear | rpm | normal
```

Prints windows by predicted class and the streaming accuracy against the replayed ground truth.

### Step 7: Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open the URL Streamlit prints (default <http://localhost:8501>), pick a scenario in the sidebar and click **Run detection stream**. Panel 10 shows `Source: ollama` when the LLM is used and `Source: template` (with an "Ollama unavailable" note) when the fallback is used.

### Step 8 (optional): Rebuild the presentation

```bash
python reports/build_slides.py
```

Regenerates `reports/CAN_IDS_Presentation.pptx` from the real metrics and figures.

### Run it all (quick reference)

```bash
python preprocessing/preprocess.py configs/config.yaml \
 && python training/train.py configs/config.yaml \
 && python evaluation/evaluate.py configs/config.yaml \
 && python explainability/shap_analysis.py configs/config.yaml \
 && streamlit run dashboard/app.py
```

### Using the pre-trained model without the raw data

`models/` ships a trained model and baseline. The **dashboard replay and simulator still need the raw capture files** in `data/raw/` (they replay real messages), and the confusion-matrix / metrics panels read the files in `reports/`.

---

## 14. Configuration reference

All stages read `configs/config.yaml`.

| Section | Key | Default | Meaning |
|---|---|---|---|
| `paths` | `raw_dir`, `interim_dir`, `processed_dir`, `models_dir`, `reports_dir`, `logs_dir` | `data/raw`, `data/interim`, `data/processed`, `models`, `reports`, `logs` | Directory layout |
| `dataset` | `files.{normal,dos,fuzzy,gear,rpm}` | `normal_run_data.csv`, `DoS_dataset.csv`, `Fuzzy_dataset.csv`, `gear_dataset.csv`, `RPM_dataset.csv` | Raw file names in `raw_dir` |
| `dataset` | `raw_columns` | `timestamp, can_id, dlc, d0..d7, flag` | Standardised output schema of the loader |
| `windowing` | `window_ms` / `stride_ms` | `100` / `50` | Window length and hop (50 % overlap) |
| `windowing` | `min_msgs_per_window` | `3` | Drop windows with fewer messages |
| `windowing` | `label_strategy` | `any_attack_present` | Window label rule |
| `features` | `timing`, `id_distribution`, `dlc_payload_entropy`, `payload_stats`, `hamming`, `compound` | all `true` | Toggle feature groups |
| `baseline` | `baseline_seconds` | `60` | Normal-only calibration slice (excluded from windowing) |
| `split` | `train` / `val` / `test` | `0.7` / `0.15` / `0.15` | Ratios (at group level) |
| `split` | `strategy` | `grouped_by_time_block` | `grouped_by_time_block` (correct) or `random` (leaky, comparison only) |
| `split` | `block_seconds` | `10` | Time-block width |
| `split` | `random_seed` | `42` | Seed |
| `balance` | `majority_class` / `multiplier` | `Normal` / `3.0` | Undersample Normal to *multiplier* × mean attack-class count (train only) |
| `model` | `objective` | `multi:softprob` | XGBoost objective |
| `model` | `num_class` | `5` | Checked against `label_encoder.json` at training time |
| `model` | `max_depth` / `eta` / `n_estimators` | `6` / `0.1` / `300` | Booster settings |
| `model` | `early_stopping_rounds` / `eval_metric` | `20` / `mlogloss` | Early stopping |
| `llm` | `provider` | `ollama` | LLM backend |
| `llm` | `model` | `qwen3:4b` | Ollama model tag |
| `llm` | `max_report_words` | `200` | Report word budget |
| top level | `random_seed` | `42` | Global seed |

---

## 15. LLM configuration reference

| Setting | Where | Default | Notes |
|---|---|---|---|
| Model tag | `configs/config.yaml` → `llm.model` | `qwen3:4b` | Must be pulled first: `ollama pull <tag>` |
| Word budget | `configs/config.yaml` → `llm.max_report_words` | `200` | Applied in the prompt *and* enforced by truncation |
| Ollama endpoint | `llm/incident_report.py` → `OLLAMA_URL` | `http://localhost:11434/api/generate` | Hard-coded constant; also used for the `/api/tags` availability check. Edit the constant to point at another host or port |
| Temperature | `llm/incident_report.py` → request `options` | `0.2` | Low temperature for stable, grounded wording |
| Timeout | `llm/incident_report.py` → `generate_report(timeout=60.0)` | `60 s` | After this the template fallback is used |
| Risk thresholds | `llm/incident_report.py` → `_risk_level()` | 0.90 / 0.70 / 0.50 | Programmatic, overrides the LLM wording |
| Fallback | automatic | n/a | Triggered on: Ollama not running, model not pulled, HTTP error, timeout, empty response |

### Switching to a different local model

```bash
ollama pull qwen3:8b            # example: any model tag Ollama can serve
```

```yaml
# configs/config.yaml
llm:
  provider: "ollama"
  model: "qwen3:8b"
  max_report_words: 200
```

> The project's frozen design is `qwen3:4b`. Other models are untested here. The availability check accepts any installed model whose name shares the family prefix (the part before `:`), but the request itself always uses the exact tag in the config. If that exact tag is not installed, the call fails and the template fallback is used. Reasoning-style models that emit `<think>…</think>` are handled; other models may need prompt adjustments in `build_prompt()`.

### Forcing the fallback (for testing)

Stop Ollama (or set `llm.model` to a tag you have not pulled) and run `python llm/incident_report.py`. The output must show `source=template`.

### Hardware guide

`qwen3:4b` fits entirely in 8 GB VRAM (tested on an RTX 4070). Without a GPU, Ollama runs on CPU; reports take longer, and if a generation exceeds the 60 s timeout the template fallback is used.

---

## 16. Reproducing the paper's numbers

All paper results use: grouped split, 10 s blocks, 70/15/15 at group level, seed 42, and the configuration in [Section 14](#14-configuration-reference).

| Claim | How to reproduce |
|---|---|
| Macro-F1 0.9990, accuracy 0.9992, per-class table | Steps 1 to 3 in [Section 13](#13-running-the-project); compare with `reports/metrics/evaluation_summary.json` |
| Model comparison (RF / XGB / LightGBM) | Printed by `training/train.py`; stored in `reports/metrics/model_comparison.csv` |
| Best iteration 114, 575 trees | `reports/metrics/training_metrics.json` → `final_training` |
| Naive-split macro-F1 = 0.9990 | Set `split.strategy: "random"` in `configs/config.yaml`, re-run Steps 1 to 3, then **restore** `grouped_by_time_block` |
| 98.8 % → 0.96 % contamination (Eq. 3) | Reported in the paper (Table 5) and the deck. The processed splits keep `group_id`, `window_start` and `source_file`, which is all Eq. (3) needs; a standalone audit script is not yet part of the repo |
| SHAP figures | Step 4 |
| 61.7 % flagged windows in a DoS replay | Step 6 / dashboard with scenario `dos` (the exact fraction depends on the replay length) |

Re-running the pipeline should reproduce the metrics to the reported precision on the same data and seed. Minor floating-point differences across library versions are possible.

---

## 17. Scope: what this work does and does not claim

**Claims**

* A verified leakage-free evaluation methodology for overlapping-window CAN IDS, with measured (not assumed) impact.
* An honest, reproducible full-dataset benchmark (macro-F1 0.9990) with explainability built in.
* A complete local-only pipeline: detection → SHAP → LLM incident report → real-time simulator → dashboard.

**Does not claim**

* **Cross-vehicle or cross-dataset generalisation.** All data come from one Hyundai YF Sonata.
* **Real-hardware validation.** The simulator replays recorded captures; no physical CAN bus or ECU was exercised and in-vehicle latency was not measured.
* **Architectural novelty.** XGBoost on engineered features is a deliberate, well-understood choice; the contribution is evaluation methodology and the explainable, local detection-to-response pipeline.
* **That leakage is always harmless.** It was negligible *on this dataset*; whether it inflates a benchmark depends on dataset, features and model.

**On near-ceiling scores.** A macro-F1 of 0.9990 means the four injected attack signatures in HCRL are almost linearly separable from normal traffic under these features. It does **not** mean in-vehicle intrusion detection is solved. More subtle or adaptive attacks (for example those in the ROAD dataset) are known to be considerably harder.

---

## 18. Limitations and future work

**Limitations and threats to validity**

* **Single vehicle.** Several of the most discriminative features are payload statistics (payload mean, byte-7 mean) whose value may depend on this vehicle's signal layout. Transfer is untested.
* **Block-level, not session-level, split.** Train and test blocks come from the same capture sessions. The dataset provides one file per attack type, so a file-level hold-out is not possible. The residual 0.96 % contamination arises at block boundaries.
* **Replayed captures only.** No physical testbed; real-time latency on in-vehicle hardware not measured.
* **Labelling scheme.** The any-attack-present rule is one of several plausible choices (e.g. proportion-based soft labels); its effect on Gear and RPM results was not ablated.
* **LLM risk wording.** The free-form risk wording of `qwen3:4b` did not reliably match the confidence-calibrated level, hence the programmatic override. Report quality was assessed qualitatively; a systematic faithfulness evaluation is future work.

**Future work**

1. Repeat the leakage-audited methodology on **ROAD** and on captures from additional vehicles (enabling file- or vehicle-level hold-out).
2. Validate on a **physical CAN testbed** and measure latency.
3. Calibrate LLM risk wording natively through fine-tuning or few-shot prompting.
4. Systematic **evaluation of report faithfulness** against the supplied attributions.
5. Ablate the window-labelling rule against soft, proportion-based labels.

---

## 19. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `FileNotFoundError` for a raw CSV | Files are not in `data/raw/` or names differ from `dataset.files` in the config |
| `ValueError` while parsing a raw file | The loader detects the format per line; a truncated or corrupted download is the usual cause, so re-download |
| XGBoost / LightGBM import error on macOS | `brew install libomp` |
| Dashboard panels show "awaiting stream" | Press **Run detection stream** in the sidebar; raw files must be present in `data/raw/` |
| Panel 10 says `Source: template` | Ollama is not running, the model is not pulled (`ollama pull qwen3:4b`), or generation timed out (CPU-only machines) |
| SHAP panel empty | `shap` not installed or model files missing; the dashboard shows the error and continues |
| Config path errors | Run from the repository root so relative paths resolve |
| Metrics differ slightly from the paper | Different library versions or a changed `split.strategy`, seed or block size |
| Pickle load warning | `models/id_baseline.pkl` is a Python pickle; only load pickles from sources you trust |

---

## 20. References

1. K. Koscher *et al.*, "Experimental security analysis of a modern automobile," *IEEE Symp. Security and Privacy*, 2010, pp. 447-462.
2. C. Miller and C. Valasek, "Remote exploitation of an unaltered passenger vehicle," *Black Hat USA*, 2015.
3. E. Seo, H. M. Song and H. K. Kim, "GIDS: GAN based intrusion detection system for in-vehicle network," *PST*, 2018, pp. 1-6.
4. H. M. Song, J. Woo and H. K. Kim, "In-vehicle network intrusion detection using deep convolutional neural network," *Veh. Commun.*, vol. 21, 100198, 2020.
5. T. Moulahi, S. Zidi, A. Alabdulatif and M. Atiquzzaman, "Comparative performance evaluation of intrusion detection based on machine learning in in-vehicle controller area network bus," *IEEE Access*, vol. 9, pp. 99595-99605, 2021.
6. M. D. Hossain *et al.*, "LSTM-based intrusion detection system for in-vehicle CAN bus communications," *IEEE Access*, vol. 8, pp. 185489-185502, 2020.
7. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, vol. 30, 2017, pp. 4765-4774.
8. S.-F. Lokman, A. T. Othman and M. H. Abu-Bakar, "Intrusion detection system for automotive controller area network (CAN) bus system: A review," *EURASIP J. Wireless Commun. Netw.*, 2019, p. 184.
9. M. A. Yagiz, P. MohajerAnsari, M. D. Pesé and P. Goktas, "Transforming in-vehicle network intrusion detection: VAE-based knowledge distillation meets explainable AI," *CPSIoTSec*, co-located with ACM CCS, 2024.
10. Y. Wang *et al.*, "Analysis of recent deep-learning-based intrusion detection methods for in-vehicle network," *IEEE Trans. Intell. Transp. Syst.*, vol. 24, p. 1843, 2023.
11. M. E. Verma *et al.*, "A comparative study of CAN-IDS using the ROAD dataset," *Int. J. Inf. Secur.*, 2024.
12. A. Alfardus and D. B. Rawat, "AI-driven intrusion detection systems (IDS) on the ROAD dataset: A comparative analysis for automotive CAN," arXiv:2408.17235, 2024.
13. V.-Q. Nguyen *et al.*, "A survey of learning-based intrusion detection systems for in-vehicle network," arXiv:2505.11551, 2025.
14. A. Habibzadeh, F. Feyzi and R. Ebrahimi Atani, "Large language models for security operations centers: A comprehensive survey," arXiv:2509.10858, 2025.
15. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016, pp. 785-794.

---

## 21. Acknowledgements

This work was supported by the **Center of Excellence (CoE) in e-Mobility** and carried out on behalf of the **Department of Electronics and Communication Engineering, RV College of Engineering, Bengaluru**, as part of the summer internship programme (7th semester, ECE). The HCRL Car-Hacking dataset is courtesy of the Hacking and Countermeasure Research Lab (HCRL), Korea University.

**Citation**

```bibtex
@article{iyer2026canids,
  title   = {Explainable and LLM-Assisted CAN Bus Intrusion Detection:
             A Leakage-Audited Benchmark on the HCRL Car-Hacking Dataset},
  author  = {Iyer, Rohit R and P, Srividya},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Manuscript}
}
```

**License.** Add a `LICENSE` file of your choice (for example MIT) before publishing; none is included yet.
