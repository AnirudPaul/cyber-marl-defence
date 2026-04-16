"""
forensic_response.py

Phase 2.0 — Adaptive Forensic Intelligence:
  - Score multi-source explainability outputs (SHAP, LIME, IG)
  - Decide action {allow, inspect, block} based on severity, cost budget, and past FP/FN stats
  - Persist incident logs for offline retraining / auditing
  - Offer helper to export feedback dataset for retraining

Usage:
  from src.forensic_response import decide_action_and_log
  result = decide_action_and_log(sample_id, features, xai_out, agent_meta, request_meta)
"""

import json
import os
import math
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# constants / config (tweakable)
LOG_DIR = Path("results/forensics")
LOG_DIR.mkdir(parents=True, exist_ok=True)
INCIDENT_LOG = LOG_DIR / "incidents.jsonl"               # newline-delimited JSON
FEEDBACK_CSV = LOG_DIR / "forensics_feedback.csv"        # saved examples for retrain
MAX_INSPECTION_RATE = 0.2    # maximum fraction of flows we are willing to inspect
BLOCK_THRESHOLD = 0.75       # severity >= => block
INSPECT_THRESHOLD = 0.4      # severity in [INSPECT_THRESHOLD, BLOCK_THRESHOLD) => inspect
ALLOW_THRESHOLD = 0.0        # severity < INSPECT_THRESHOLD => allow

# cost model (higher = more costly)
COST_BLOCK = 2.0     # blocking cost (possible collateral)
COST_INSPECT = 0.5   # cost for deep inspection
COST_ALLOW = 0.0     # cost for allow

# basic moving-window counters (kept in-memory for runtime decisions)
_runtime_counters = {
    "total": 0,
    "inspected": 0,
    "blocked": 0,
    "allow": 0,
    "false_positives": 0,   # to be filled by offline labeling / feedback
    "false_negatives": 0
}


def _safe_mean(a):
    try:
        return float(np.nanmean(np.asarray(a).astype(float)))
    except Exception:
        return 0.0


def score_from_shap(shap_dict: Dict[str, Any]) -> float:
    """
    Compute a normalized severity contribution from SHAP output.
    Expected shap_dict: {'mean_abs': <array or list>}
    Return: [0,1] severity contribution
    """
    if not shap_dict:
        return 0.0
    try:
        mean_abs = shap_dict.get("mean_abs", None)
        if mean_abs is None:
            return 0.0
        arr = np.asarray(mean_abs, dtype=float)
        # Collapse over features and actions (support multi-output)
        val = np.nanmean(np.abs(arr))
        # normalize using robust scaling (empirically tuned)
        # use log scaling to compress large values (avoids numeric overflow)
        score = math.tanh(val / (1.0 + np.median(arr) + 1e-6))
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0


def score_from_lime(lime_dict: Dict[str, Any]) -> float:
    """
    Compute severity contribution from LIME.
    Expected lime_dict: {'lime_results': [...]} or {'lime_features': ...}
    We use sum of absolute weights normalized by a heuristic.
    """
    if not lime_dict:
        return 0.0
    try:
        # support multiple output formats
        entries = []
        if "lime_results" in lime_dict:
            for r in lime_dict["lime_results"]:
                feats = r.get("lime_features", [])
                entries.append(sum(abs(w) for (_, w) in feats))
        elif "lime_features" in lime_dict:
            feats = lime_dict["lime_features"]
            entries.append(sum(abs(w) for (_, w) in feats))
        else:
            return 0.0
        val = np.mean(entries) if entries else 0.0
        # normalize
        score = math.tanh(val / (1.0 + np.median(entries) if entries else 1.0))
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0


def score_from_ig(ig_dict: Dict[str, Any]) -> float:
    """
    Integrated Gradients severity: mean absolute attribution magnitude.
    """
    if not ig_dict:
        return 0.0
    try:
        vals = ig_dict.get("ig_values", None) or ig_dict.get("integrated_gradients", None)
        if vals is None:
            return 0.0
        arr = np.asarray(vals, dtype=float)
        val = np.nanmean(np.abs(arr))
        score = math.tanh(val / (1.0 + np.median(np.abs(arr)) + 1e-6))
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0


def aggregate_severity(xai_out: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> float:
    """
    Combine SHAP / LIME / IG scores into a single severity in [0,1].
    weights default: shap=0.5, lime=0.3, ig=0.2
    """
    if weights is None:
        weights = {"shap": 0.5, "lime": 0.3, "ig": 0.2}
    s_shap = score_from_shap(xai_out.get("shap", {}))
    s_lime = score_from_lime(xai_out.get("lime", {}))
    s_ig = score_from_ig(xai_out.get("integrated_gradients", {}))
    total = (weights["shap"] * s_shap) + (weights["lime"] * s_lime) + (weights["ig"] * s_ig)
    # final clamp
    return float(np.clip(total, 0.0, 1.0))


def resource_safe_action(severity: float, feature_cost_estimate: float = 1.0) -> str:
    """
    Decide action purely from severity and simple resource constraint.
    feature_cost_estimate is a future hook for cost-per-feature computing.
    Returns 'allow'|'inspect'|'block'
    """
    # enforce global inspect rate budget
    inspect_budget_ok = (_runtime_counters["inspected"] / max(1, _runtime_counters["total"])) < MAX_INSPECTION_RATE

    if severity >= BLOCK_THRESHOLD:
        _runtime_counters["blocked"] += 1
        return "block"
    if severity >= INSPECT_THRESHOLD and inspect_budget_ok:
        _runtime_counters["inspected"] += 1
        return "inspect"
    _runtime_counters["allow"] += 1
    return "allow"


def log_incident(sample_id: str,
                 features,
                 xai_out: Dict[str, Any],
                 action: str,
                 severity: float,
                 agent_meta: Optional[Dict[str, Any]] = None,
                 request_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Persist a JSONL incident record for auditing / retraining.
    Returns the record dict.
    """
    rec = {
        "ts": time.time(),
        "id": sample_id,
        "action": action,
        "severity": float(severity),
        "agent_meta": agent_meta or {},
        "request_meta": request_meta or {},
        "features": np.asarray(features).tolist() if features is not None else None,
        "xai": xai_out
    }
    # append to incidents file
    try:
        with INCIDENT_LOG.open("a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        # best-effort
        pass

    # also save a CSV feedback row (feature vector + assigned action + severity)
    try:
        import csv
        header = None
        features_list = np.asarray(features).tolist() if features is not None else []
        # header creation (first time)
        if not FEEDBACK_CSV.exists():
            header = ["ts", "id"] + [f"f{i}" for i in range(len(features_list))] + ["action", "severity"]
            with FEEDBACK_CSV.open("w", newline="", encoding="utf8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
        # append row
        row = [time.time(), sample_id] + features_list + [action, severity]
        with FEEDBACK_CSV.open("a", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception:
        pass

    # bookkeeping
    _runtime_counters["total"] += 1

    return rec


def decide_action_and_log(sample_id: str,
                          features,
                          xai_out: Dict[str, Any],
                          agent_meta: Optional[Dict[str, Any]] = None,
                          request_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    End-to-end: compute severity, choose action, log incident and return structured decision.
    Returns:
      {
        'action': 'inspect'|'block'|'allow',
        'severity': float,
        'record': {...}   # persisted JSON record
      }
    """
    try:
        severity = aggregate_severity(xai_out)
        action = resource_safe_action(severity)
        rec = log_incident(sample_id, features, xai_out, action, severity, agent_meta=agent_meta, request_meta=request_meta)
        return {"action": action, "severity": float(severity), "record": rec}
    except Exception as e:
        # fallback: allow, and log the error for review
        rec = {
            "ts": time.time(),
            "id": sample_id,
            "error": str(e),
            "xai": xai_out
        }
        try:
            with INCIDENT_LOG.open("a", encoding="utf8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        _runtime_counters["total"] += 1
        return {"action": "allow", "severity": 0.0, "record": rec}


# utilities
def get_counters() -> Dict[str, int]:
    return dict(_runtime_counters)


def reset_counters():
    for k in _runtime_counters:
        _runtime_counters[k] = 0


def export_feedback_for_retrain(dest: Optional[str] = None) -> str:
    """
    Return path to feedback CSV for use in retraining pipelines.
    """
    return str(FEEDBACK_CSV if dest is None else Path(dest).resolve())
