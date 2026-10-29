"""
app.py — CAN IDS Streamlit Dashboard (Phase 10)
================================================
Live demo + analysis dashboard with the 11 required sections (README Section 13):

  1. Live CAN Traffic Feed       7. Training Metrics
  2. Current Prediction          8. Loss Curves
  3. Attack Status               9. SHAP Explanations
  4. Prediction Confidence      10. AI Incident Report
  5. Attack Statistics          11. Historical Attack Timeline
  6. Confusion Matrix

Architecture (README Section 13): this file contains NO ML/analysis business
logic. It imports the backing modules and only orchestrates + renders:
  - simulation.can_stream_simulator -> live stream + per-window predictions
  - explainability.shap_analysis     -> SHAP local explanations
  - llm.incident_report              -> AI (or template-fallback) incident report
  - evaluation / training artifacts  -> confusion matrix, metrics, loss curves, figures

Run:  streamlit run dashboard/app.py

Dependencies
------------
streamlit, pandas, numpy  (+ the backing modules and their deps)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
for sub in ("preprocessing", "explainability", "llm", "simulation", "evaluation"):
    sys.path.insert(0, str(_ROOT / sub))

from feature_extractor import load_config  # noqa: E402

st.set_page_config(page_title="CAN Bus IDS", layout="wide", page_icon="🚗")

CONFIG_PATH = str(_ROOT / "configs" / "config.yaml")


# ------------------------------------------------------------------ #
# Cached loaders — all heavy objects come from the backing modules.
# ------------------------------------------------------------------ #
@st.cache_resource
def get_config() -> dict:
    return load_config(CONFIG_PATH)


@st.cache_resource
def get_simulator():
    from can_stream_simulator import CANStreamSimulator
    return CANStreamSimulator(get_config())


@st.cache_resource
def get_explainer():
    try:
        from shap_analysis import ShapExplainer
        return ShapExplainer.from_config(get_config())
    except Exception as e:  # SHAP optional at view time
        st.session_state["shap_error"] = str(e)
        return None


@st.cache_data
def load_metrics() -> dict:
    cfg = get_config()
    mdir = Path(cfg["paths"]["reports_dir"]) / "metrics"
    out = {}
    for name in ("training_metrics.json", "evaluation_summary.json"):
        p = mdir / name
        if p.exists():
            out[name] = json.loads(p.read_text())
    for name in ("training_history.csv",):
        p = mdir / name
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def fig_path(name: str) -> Path:
    return Path(get_config()["paths"]["reports_dir"]) / "figures" / name


def run_stream(file_key: str, max_messages: int) -> pd.DataFrame:
    sim = get_simulator()
    rows = []
    for wp in sim.stream_file(file_key, max_messages=max_messages):
        rows.append({
            "window": wp.window_index, "t_start": wp.window_start,
            "predicted": wp.predicted_class, "confidence": wp.confidence,
            "true": wp.true_label, **{f"p_{k}": v for k, v in wp.class_probabilities.items()},
            "_features": wp.features,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Sidebar — scenario controls
# ------------------------------------------------------------------ #
cfg = get_config()
st.sidebar.title("🚗 CAN Bus IDS")
st.sidebar.caption("Explainable + LLM-assisted intrusion detection")
scenario = st.sidebar.selectbox("Attack scenario (replay)", ["dos", "fuzzy", "gear", "rpm", "normal"], index=0)
max_msgs = st.sidebar.slider("Messages to replay", 10_000, 120_000, 40_000, step=10_000)
if st.sidebar.button("▶ Run detection stream", use_container_width=True):
    with st.spinner(f"Replaying {scenario} capture through the detector…"):
        st.session_state["stream"] = run_stream(scenario, max_msgs)
        st.session_state["scenario"] = scenario

stream: pd.DataFrame | None = st.session_state.get("stream")

st.title("CAN Bus Intrusion Detection — Live Dashboard")
if stream is None:
    st.info("Pick a scenario in the sidebar and click **Run detection stream** to populate the live sections.")

# ================================================================== #
# 1. Live CAN Traffic Feed
# ================================================================== #
st.header("1 · Live CAN Traffic Feed")
if stream is not None:
    feed = stream[["window", "t_start", "predicted", "confidence", "true"]].tail(200)
    st.dataframe(feed, use_container_width=True, height=240)
else:
    st.caption("— awaiting stream —")

# ================================================================== #
# 2. Current Prediction  &  3. Attack Status  &  4. Prediction Confidence
# ================================================================== #
c2, c3, c4 = st.columns(3)
if stream is not None and len(stream):
    last = stream.iloc[-1]
    with c2:
        st.header("2 · Current Prediction")
        st.metric("Latest window class", last["predicted"], help=f"window #{int(last['window'])}")
    with c3:
        st.header("3 · Attack Status")
        is_attack = last["predicted"] != "Normal"
        st.markdown(f"### {'🔴 ATTACK DETECTED' if is_attack else '🟢 Normal'}")
        st.caption(f"{(stream['predicted'] != 'Normal').mean():.1%} of windows flagged as attack this run")
    with c4:
        st.header("4 · Prediction Confidence")
        st.metric("Confidence", f"{last['confidence']:.2%}")
        st.progress(min(1.0, float(last["confidence"])))
else:
    for col, title in zip((c2, c3, c4), ("2 · Current Prediction", "3 · Attack Status", "4 · Prediction Confidence")):
        col.header(title)
        col.caption("— awaiting stream —")

# ================================================================== #
# 5. Attack Statistics
# ================================================================== #
st.header("5 · Attack Statistics")
if stream is not None:
    counts = stream["predicted"].value_counts()
    a, b = st.columns([1, 2])
    with a:
        st.dataframe(counts.rename("windows"), use_container_width=True)
    with b:
        st.bar_chart(counts)
    if stream["true"].notna().any():
        acc = (stream["predicted"] == stream["true"]).mean()
        st.metric("Streaming accuracy vs. replayed ground truth", f"{acc:.4f}")
else:
    st.caption("— awaiting stream —")

# ================================================================== #
# 6. Confusion Matrix  (from evaluation artifacts)
# ================================================================== #
st.header("6 · Confusion Matrix (held-out test set)")
cm_fig = fig_path("eval_confusion_matrix.png")
if cm_fig.exists():
    st.image(str(cm_fig), caption="Normalized confusion matrix — leakage-fixed test split")
else:
    st.warning("Run `evaluation/evaluate.py` to generate the confusion matrix.")

# ================================================================== #
# 7. Training Metrics
# ================================================================== #
st.header("7 · Training Metrics")
metrics = load_metrics()
ev = metrics.get("evaluation_summary.json")
if ev:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test macro-F1", f"{ev['macro_f1']:.4f}")
    m2.metric("Accuracy", f"{ev['accuracy']:.4f}")
    m3.metric("Weighted F1", f"{ev['weighted_f1']:.4f}")
    m4.metric("Test windows", f"{ev['n_test_windows']:,}")
    st.subheader("Per-class F1")
    per_class = pd.DataFrame(ev["per_class"]).T[["precision", "recall", "f1", "support"]]
    st.dataframe(per_class, use_container_width=True)
else:
    st.warning("No evaluation_summary.json yet — run the evaluation phase.")

# ================================================================== #
# 8. Loss Curves
# ================================================================== #
st.header("8 · Loss Curves")
hist = metrics.get("training_history.csv")
if hist is not None:
    st.line_chart(hist.set_index("round")[["train_mlogloss", "val_mlogloss"]])
else:
    for fig in ("training_loss_curve.png", "validation_loss_curve.png"):
        if fig_path(fig).exists():
            st.image(str(fig_path(fig)))

# ================================================================== #
# 9. SHAP Explanations
# ================================================================== #
st.header("9 · SHAP Explanations")
tab_global, tab_local = st.tabs(["Global", "Local (selected window)"])
with tab_global:
    if fig_path("shap_summary.png").exists():
        st.image(str(fig_path("shap_summary.png")), caption="Global feature importance (all classes)")
        cls = st.selectbox("Per-class SHAP", cfg_classes := [
            fp.stem.replace("shap_class_", "") for fp in fig_path(".").glob("shap_class_*.png")
        ] or ["(none)"])
        if fig_path(f"shap_class_{cls}.png").exists():
            st.image(str(fig_path(f"shap_class_{cls}.png")))
    else:
        st.warning("Run `explainability/shap_analysis.py` to generate SHAP figures.")
with tab_local:
    explainer = get_explainer()
    if stream is not None and len(stream) and explainer is not None:
        widx = st.slider("Window index", 0, len(stream) - 1, len(stream) - 1)
        feats = stream.iloc[widx]["_features"]
        local = explainer.explain_prediction(feats)
        st.write(f"**Predicted:** {local['predicted_class']}  ·  **confidence:** {local['confidence']:.2%}")
        st.dataframe(pd.DataFrame(local["top_features"]), use_container_width=True)
        st.session_state["last_local"] = local
    else:
        st.caption("Run a stream (and generate SHAP) to inspect a single window.")

# ================================================================== #
# 10. AI Incident Report
# ================================================================== #
st.header("10 · AI Incident Report")
if stream is not None and len(stream):
    from incident_report import generate_report
    local = st.session_state.get("last_local")
    explainer = get_explainer()
    # default to the highest-confidence attack window if no window was hand-picked
    if local is None:
        attacks = stream[stream["predicted"] != "Normal"]
        pick = (attacks if len(attacks) else stream).sort_values("confidence").iloc[-1]
        if explainer is not None:
            local = explainer.explain_prediction(pick["_features"])
        else:
            local = {"predicted_class": pick["predicted"], "confidence": float(pick["confidence"]), "top_features": []}
    rep = generate_report(local["predicted_class"], local["confidence"], local.get("top_features", []), cfg=cfg)
    st.caption(f"Source: **{rep['source']}** · Risk level: **{rep['risk_level']}**"
               + ("" if rep["source"] == "ollama" else "  (Ollama unavailable → deterministic template fallback)"))
    for line in rep["report"].split("\n"):
        if ":" in line:
            head, body = line.split(":", 1)
            st.markdown(f"**{head.strip()}:** {body.strip()}")
        elif line.strip():
            st.write(line)
else:
    st.caption("— awaiting stream —")

# ================================================================== #
# 11. Historical Attack Timeline
# ================================================================== #
st.header("11 · Historical Attack Timeline")
if stream is not None and len(stream):
    tl = stream[["t_start", "predicted", "confidence"]].copy()
    tl["is_attack"] = (tl["predicted"] != "Normal").astype(int)
    tl["rel_time_s"] = tl["t_start"] - tl["t_start"].iloc[0]
    st.area_chart(tl.set_index("rel_time_s")[["is_attack"]])
    st.caption("Attack-active windows over the replay timeline (1 = attack, 0 = normal). "
               "Reflects the discrete ~3–5s HCRL attack bursts.")
else:
    st.caption("— awaiting stream —")
