# CAN Bus IDS — Results & Methodology Report

*Generated from the leakage-fixed pipeline on the full HCRL Car-Hacking dataset
(~16.6M messages across Normal, DoS, Fuzzy, Gear-spoofing, RPM-spoofing captures).*

## 1. Headline result (honest, held-out test set)

| Metric | Value |
|---|---|
| **Macro F1 (test)** | **0.9990** |
| Accuracy | 0.9992 |
| Weighted F1 | 0.9992 |
| Test windows | 33,247 |
| Split strategy | grouped-by-time-block (10 s blocks, per source file) |

Per-class (held-out test):

| Class | Precision | Recall | F1 | Support | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| DoS | 0.9986 | 0.9998 | 0.9992 | 4249 | 1.0000 | 1.0000 |
| Fuzzy | 0.9998 | 0.9985 | 0.9991 | 4673 | 1.0000 | 1.0000 |
| Gear | 0.9995 | 0.9970 | 0.9983 | 4047 | 1.0000 | 0.9999 |
| Normal | 0.9992 | 0.9999 | 0.9996 | 17153 | 1.0000 | 1.0000 |
| RPM | 0.9990 | 0.9990 | 0.9990 | 3125 | 1.0000 | 0.9999 |

Model: XGBoost (`multi:softprob`), best iteration 114, balanced inverse-frequency
sample weighting. Random Forest and LightGBM were compared on validation
(0.99911 / 0.99891 macro-F1) and did not beat XGBoost by a deployable margin;
XGBoost remains the frozen deployment model.

## 2. The leakage audit and what actually happened

**Problem.** The preprocessing split was a plain, ungrouped stratified
`train_test_split` over the pooled window table. Because windows are 100 ms wide
with a 50 ms stride (50 % overlap), each window shares ~half its raw messages with
its temporal neighbours; random splitting scatters these near-duplicates across
train/val/test, so the "held-out" set is barely held out. A prior synthetic
reproduction measured 98.8 % of test windows having an overlapping neighbour in
train, and a modest macro-F1 drop (0.824 → 0.797) when the split was made grouped.

**Fix.** Windows are now bucketed into contiguous, non-overlapping 10-second time
blocks *per source file*, and whole blocks are assigned atomically to a single
split (stratified by each block's dominant class). This makes overlap leakage
structurally impossible across the train/val/test boundary.

**Verified invariants (full dataset):**
- Group overlap between splits: **0** (train∩val = train∩test = val∩test = ∅).
- Contamination — test windows with a train window <100 ms away in the same file:
  **0.96 %** (down from the audit's 98.8 %). The residual ~1 % is block-boundary
  edge effect, not scattering.

**Measured before/after inflation (apples-to-apples, both scored on their own
held-out test set, full real dataset):**

| Split strategy | Test macro-F1 |
|---|---|
| Random (ungrouped, leaky) | 0.9990 |
| Grouped-by-time-block (leakage-free) | 0.9990 |

**Finding.** On the real, full HCRL dataset the overlap leakage did **not**
materially inflate macro-F1 — both splits land at 0.9990. The synthetic
reproduction's modest drop did not reproduce on real data. The honest
interpretation is *not* "the model is inflated"; it is that the 25 generic
behavioral features separate these five classes almost perfectly even across
cleanly held-out time blocks, and the previously-reported 0.9988 was not driven
by contamination. The value of the fix is **methodological validity**: the
reported number is now a defensible generalization estimate on a genuinely
held-out set, not an unproven one. We did not tune the split, seed, or block size
to move the number in either direction.

## 3. Reproduction

```bash
python3.13 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
# (macOS: brew install libomp   — OpenMP runtime for xgboost/lightgbm)
cd CANIDS
../.venv/bin/python preprocessing/preprocess.py configs/config.yaml   # grouped split
../.venv/bin/python training/train.py            configs/config.yaml
../.venv/bin/python evaluation/evaluate.py       configs/config.yaml
../.venv/bin/python explainability/shap_analysis.py configs/config.yaml
```

Set `split.strategy: random` in `configs/config.yaml` to reproduce the leaky
baseline for comparison.

## 4. Figures

`reports/figures/`: `eval_confusion_matrix.png`, `eval_per_class_f1.png`,
`eval_roc_curves.png`, `eval_pr_curves.png`, `shap_summary.png`,
`shap_class_{DoS,Fuzzy,Gear,Normal,RPM}.png`, `training_loss_curve.png`,
`validation_loss_curve.png`, `learning_curve.png`, `feature_importance.png`,
`class_distribution.png`.
