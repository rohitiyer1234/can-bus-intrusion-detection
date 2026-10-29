"""
incident_report.py
==================
Phase 8 LLM incident reporting. Turns one detection (predicted attack class +
confidence + SHAP top features from Phase 7) into a concise, human-readable
incident report with five sections:

    Incident Summary | Risk Level | Technical Analysis |
    Potential Impact | Recommended Mitigation

Frozen design (README Section 11): the LLM is Ollama + qwen3:4b, LOCAL ONLY --
no cloud/paid APIs, no alternative-LLM evaluation.

REQUIRED fallback: if Ollama is unavailable (not running, model missing, HTTP
error, timeout), generate a deterministic TEMPLATE-based report instead. The
fallback is explicitly NOT another LLM -- it is a fixed, auditable template.
The rest of the pipeline (detection, dashboard) must keep working with no LLM
present, so report generation never raises on Ollama being down.

Safety (README Section 11):
  - Do not fabricate technical claims beyond what the SHAP features support:
    the prompt is grounded ONLY in the passed features, and the fallback quotes
    the features verbatim.
  - Risk-level language is calibrated to the model's actual confidence, not
    uniformly alarmist -- see `_risk_level`.

Dependencies
------------
requests, pyyaml   (Ollama is optional at runtime; absence triggers the fallback)
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from feature_extractor import load_config  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"

# Per-attack context used by BOTH the LLM prompt and the deterministic fallback,
# so the two paths make the same (grounded, non-fabricated) claims.
ATTACK_CONTEXT = {
    "DoS": {
        "mechanism": "high-priority ID flooding that monopolizes bus arbitration",
        "impact": "legitimate ECMs are starved of bus access; safety-critical messages may be delayed or dropped",
    },
    "Fuzzy": {
        "mechanism": "injection of randomized CAN IDs and payloads",
        "impact": "unpredictable actuator behavior and diagnostic instability as spoofed frames are interpreted as valid",
    },
    "Gear": {
        "mechanism": "spoofed transmission/gear-state frames on a normal-priority ID",
        "impact": "false gear indication or shift behavior, a direct drivability and safety concern",
    },
    "RPM": {
        "mechanism": "spoofed engine-RPM frames on a normal-priority ID",
        "impact": "false tachometer/engine-speed signals that can mislead dependent control logic",
    },
    "Normal": {
        "mechanism": "no attack signature detected",
        "impact": "none; traffic is consistent with the calibrated baseline",
    },
}


def _risk_level(attack_class: str, confidence: float) -> str:
    """
    Calibrate risk language to confidence, not uniform alarm (README safety note).
    Normal is always Informational regardless of confidence.
    """
    if attack_class == "Normal":
        return "Informational"
    if confidence >= 0.90:
        return "Critical"
    if confidence >= 0.70:
        return "High"
    if confidence >= 0.50:
        return "Medium"
    return "Low (uncertain detection — verify before acting)"


def _format_features(top_features: list[dict]) -> str:
    return "; ".join(
        f"{f['feature']} ({f.get('value', float('nan')):.3f}, {f.get('direction', 'affects')} the score)"
        for f in top_features
    )


def build_prompt(attack_class: str, confidence: float, top_features: list[dict], max_words: int) -> str:
    ctx = ATTACK_CONTEXT.get(attack_class, ATTACK_CONTEXT["Normal"])
    feats = _format_features(top_features)
    return (
        "You are a vehicle cybersecurity analyst writing a CAN bus intrusion incident report.\n"
        "Ground every technical claim ONLY in the detector output below. Do not invent CAN IDs, "
        "byte values, or attacker identities that are not given.\n\n"
        f"Detection: class={attack_class}, model_confidence={confidence:.2%}\n"
        f"Attack mechanism (reference): {ctx['mechanism']}\n"
        f"Top contributing features (SHAP): {feats}\n\n"
        f"Write a report UNDER {max_words} words with EXACTLY these five sections, each on its own line "
        "prefixed by the section name and a colon:\n"
        "Incident Summary: ...\nRisk Level: ...\nTechnical Analysis: ...\n"
        "Potential Impact: ...\nRecommended Mitigation: ...\n"
        "Keep the risk level consistent with the stated confidence (do not overstate a low-confidence detection)."
    )


def template_report(attack_class: str, confidence: float, top_features: list[dict], max_words: int) -> str:
    """
    Deterministic, auditable fallback used when Ollama is unavailable. Quotes the
    SHAP features verbatim -- makes no claim the features don't support.
    """
    ctx = ATTACK_CONTEXT.get(attack_class, ATTACK_CONTEXT["Normal"])
    risk = _risk_level(attack_class, confidence)
    feats = _format_features(top_features) or "no dominant features reported"

    if attack_class == "Normal":
        summary = f"Traffic classified as Normal with {confidence:.1%} confidence; no intrusion signature detected."
        analysis = f"The classifier found behavior consistent with the calibrated baseline. Leading features: {feats}."
        impact = ctx["impact"].capitalize() + "."
        mitigation = "No action required. Continue monitoring; re-baseline only on trusted traffic."
    else:
        summary = (
            f"A {attack_class} attack was detected on the CAN bus with {confidence:.1%} model confidence."
        )
        analysis = (
            f"Detection is driven by {ctx['mechanism']}. The features most responsible for this "
            f"classification were: {feats}."
        )
        impact = f"Potential impact: {ctx['impact']}."
        mitigation = (
            "Isolate/rate-limit the offending traffic, verify affected ECMs against expected signals, "
            "and review bus logs around the detection window. "
            + ("Treat as urgent given high confidence." if confidence >= 0.9 else "Corroborate before escalating given the confidence level.")
        )

    report = (
        f"Incident Summary: {summary}\n"
        f"Risk Level: {risk}\n"
        f"Technical Analysis: {analysis}\n"
        f"Potential Impact: {impact}\n"
        f"Recommended Mitigation: {mitigation}"
    )
    return _truncate_words(report, max_words)


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",.;") + " …"


def _ollama_available(cfg: dict, timeout: float = 2.0) -> bool:
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=timeout)
        if tags.status_code != 200:
            return False
        model = cfg["llm"]["model"]
        names = [m.get("name", "") for m in tags.json().get("models", [])]
        # accept exact or family match (e.g. "qwen3:4b" vs "qwen3:4b-instruct")
        return any(n == model or n.startswith(model.split(":")[0]) for n in names)
    except (requests.RequestException, ValueError):
        return False


def generate_report(
    attack_class: str,
    confidence: float,
    top_features: list[dict],
    config_path: str = "configs/config.yaml",
    cfg: dict | None = None,
    timeout: float = 60.0,
) -> dict:
    """
    Generate an incident report. Returns {'report': str, 'source': 'ollama'|'template',
    'risk_level': str}. NEVER raises on Ollama being unavailable -- falls back to template.
    """
    cfg = cfg or load_config(config_path)
    max_words = cfg["llm"].get("max_report_words", 200)
    risk = _risk_level(attack_class, confidence)

    if _ollama_available(cfg):
        try:
            prompt = build_prompt(attack_class, confidence, top_features, max_words)
            resp = requests.post(
                OLLAMA_URL,
                json={"model": cfg["llm"]["model"], "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.2}},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            if text:
                return {"report": _truncate_words(text, max_words), "source": "ollama", "risk_level": risk}
        except (requests.RequestException, ValueError) as e:
            print(f"[llm][warn] Ollama call failed ({e}); using template fallback")

    return {"report": template_report(attack_class, confidence, top_features, max_words),
            "source": "template", "risk_level": risk}


if __name__ == "__main__":
    # Smoke test: exercise the fallback with a representative detection.
    demo_features = [
        {"feature": "max_id_msg_rate", "value": 850.0, "direction": "increases"},
        {"feature": "id_entropy", "value": 0.42, "direction": "decreases"},
        {"feature": "mean_iat", "value": 0.0004, "direction": "decreases"},
    ]
    out = generate_report("DoS", 0.987, demo_features)
    print(f"[source={out['source']}] risk={out['risk_level']}\n")
    print(out["report"])
