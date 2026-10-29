# Research Contribution Statement

*A deliberately non-exaggerated statement of what this project does and does not
demonstrate. Written to survive scrutiny, not to maximize claims.*

## What we claim

1. **A leakage-free evaluation methodology for overlapping-window CAN IDS.**
   We identify and fix a train/val/test leakage mode specific to sliding-window
   CAN intrusion detection: with 50 %-overlapping windows, a naive row-wise split
   scatters near-duplicate windows across splits. We replace it with a grouped,
   time-block split (whole non-overlapping blocks per source file assigned
   atomically), and we *verify* the fix with two invariants — zero group overlap
   across splits, and a drop in same-file <100 ms contamination from 98.8 % to
   0.96 %. This is the primary methodological contribution.

2. **An honest, reproducible benchmark on the full HCRL Car-Hacking dataset.**
   Under the corrected split, generic vehicle-agnostic behavioral features
   (timing, ID-distribution, entropy, payload statistics, per-ID Hamming distance)
   classify Normal/DoS/Fuzzy/Gear/RPM at macro-F1 0.9990 on a genuinely held-out
   test set, including the historically-hard Gear and RPM spoofing classes
   (F1 0.9983 / 0.9990).

3. **An empirical result about the leakage itself.** We measured the actual
   before/after inflation on real data (both splits scored on their own held-out
   set): it is negligible here (0.9990 vs 0.9990). We report this as-is rather
   than presenting the fix as having "recovered" an inflated number. The
   contribution is that the benchmark is now *defensible*, not that it changed.

4. **A complete, explainable, local-only pipeline.** End-to-end: preprocessing →
   XGBoost detection → SHAP explanations (global + per-window local) → LLM-assisted
   incident reporting (Ollama + qwen3:4b, with a required deterministic template
   fallback) → real-time replay simulator → Streamlit dashboard.

## What we explicitly do NOT claim

- **Not cross-vehicle generalization.** The dataset is a single Hyundai Sonata
  capture. The feature set was *designed* to avoid vehicle-specific hardcoding
  (vehicle-specific CAN-ID features were removed after review), but this is a
  design intention, not empirical proof that the model transfers to other vehicles.
- **Not real-hardware validation.** The simulator replays recorded captures; there
  is no live CAN hardware in scope.
- **Not a novel model architecture.** XGBoost on engineered features is a
  deliberate, well-understood choice; the novelty is in methodology and
  explainability, not the classifier.
- **Not a claim that leakage is always harmless.** We found it negligible *on this
  dataset*; the correct split should still be used, because whether leakage
  inflates results is dataset-dependent and cannot be assumed away.

## Honesty notes

- Near-perfect scores on a single benchmark warrant caution: they reflect that
  HCRL's attack signatures are strongly separable under these features, not that
  the problem is "solved." Cross-dataset / cross-vehicle evaluation is the
  necessary next step and is stated as future work, not as an achieved result.
- No metric was obtained by tuning the split strategy, random seed, or block size
  to a favorable value.
