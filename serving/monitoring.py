"""Production monitoring: feature drift, prediction logging, delayed-label performance.

Strategy
--------
1. **Feature drift (PSI)** — incoming feature batches are compared to the training
   reference distribution (decile bins). PSI > 0.2 = moderate shift, > 0.25 = significant.
   Drift is a *batch* statistic, so it runs over the recent prediction log or an explicit batch.
2. **Prediction logging** — every request/response is appended to a ring buffer (and optional
   JSONL file) for audit, drift, and prediction-distribution tracking.
3. **Delayed-label performance** — incident labels only materialise after a forecast window
   closes. Outcomes are submitted to `/monitoring/outcome`; rolling precision/recall/PR-AUC are
   computed per window once both classes are present.
4. **Retraining triggers** — significant feature drift OR a drop in rolling PR-AUC below a
   floor should trigger retraining (the day-30 snapshot also rolls forward monthly).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
LOG_PATH = Path(os.getenv("PREDICTION_LOG", "logs/predictions.jsonl"))
BUFFER_SIZE = int(os.getenv("MONITOR_BUFFER", "5000"))

PSI_MODERATE = 0.20
PSI_SIGNIFICANT = 0.25
# global fallback floor; per-window baseline-relative floors are loaded from serving_meta.json
PRAUC_FLOOR_DEFAULT = float(os.getenv("PRAUC_FLOOR", "0.25"))


def _psi(actual_values: np.ndarray, bin_edges: list[float],
         expected_fracs: list[float]) -> float:
    """PSI between an incoming batch and the training reference.

    Compares the actual per-bin proportions against the *stored training* proportions
    (`expected_fracs`) — not a uniform assumption, since decile edges collapse for skewed
    or binary features and leave unequal-mass bins.
    """
    edges = np.array(bin_edges, dtype=float)
    if len(edges) < 2 or not expected_fracs:
        return 0.0
    clipped = np.clip(actual_values, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    actual_fracs = counts / max(counts.sum(), 1)
    psi = 0.0
    for a, e in zip(actual_fracs, expected_fracs):
        a = max(float(a), 1e-6)
        e = max(float(e), 1e-6)
        psi += (a - e) * math.log(a / e)
    return float(psi)


class Monitor:
    def __init__(self) -> None:
        ref_path = ARTIFACTS_DIR / "reference_stats.json"
        self.reference = json.loads(ref_path.read_text()) if ref_path.exists() else {}

        # per-window baseline-relative PR-AUC floors (retraining trigger)
        meta_path = ARTIFACTS_DIR / "serving_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.pr_auc_floor: dict[str, float] = meta.get("pr_auc_floor", {})

        self.predictions: deque = deque(maxlen=BUFFER_SIZE)
        self.latencies: deque = deque(maxlen=BUFFER_SIZE)
        self.tier_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        # outcomes[window] = list of (predicted_prob, actual_int)
        self.outcomes: dict[str, list[tuple[float, int]]] = {}
        self._last_drift_status = "ok"
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ----- logging -----
    def log_prediction(self, features: dict, response: dict, latency_ms: float) -> None:
        self.predictions.append(features)
        self.latencies.append(latency_ms)
        for fc in response.get("forecasts", []):
            self.tier_counts[fc["risk_tier"]] = self.tier_counts.get(fc["risk_tier"], 0) + 1
        record = {"ts": time.time(), "features": features, "response": response,
                  "latency_ms": latency_ms}
        try:
            with LOG_PATH.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass  # logging must never break serving

    # ----- drift -----
    def drift_report(self, batch: list[dict] | None = None) -> dict:
        samples = batch if batch is not None else list(self.predictions)
        if not samples or not self.reference:
            return {"n_samples": len(samples), "n_features_checked": 0,
                    "drifted_features": [], "overall_status": "ok"}

        drifted = []
        checked = 0
        for feat, ref in self.reference.items():
            if not ref.get("bin_edges"):
                continue  # feature constant in training — not tracked
            vals = np.array([float(s.get(feat, 0.0)) for s in samples], dtype=float)
            checked += 1
            psi = _psi(vals, ref["bin_edges"], ref.get("expected_fracs", []))
            if psi >= PSI_MODERATE:
                severity = "significant" if psi >= PSI_SIGNIFICANT else "moderate"
                drifted.append({"feature": feat, "psi": round(psi, 4), "severity": severity})

        drifted.sort(key=lambda d: d["psi"], reverse=True)
        n_sig = sum(1 for d in drifted if d["severity"] == "significant")
        status = "alert" if n_sig else ("warning" if drifted else "ok")
        self._last_drift_status = status
        return {"n_samples": len(samples), "n_features_checked": checked,
                "drifted_features": drifted, "overall_status": status}

    # ----- delayed-label performance -----
    def record_outcome(self, window: str, predicted_probability: float,
                       incident_occurred: bool) -> dict:
        self.outcomes.setdefault(window, []).append(
            (float(predicted_probability), int(incident_occurred))
        )
        return self.performance().get(window, {})

    def performance(self, threshold: float = 0.5) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for window, rows in self.outcomes.items():
            if len(rows) < 10:
                continue
            probs = np.array([r[0] for r in rows])
            actual = np.array([r[1] for r in rows])
            preds = (probs >= threshold).astype(int)
            metrics = {
                "n": float(len(rows)),
                "precision": float(precision_score(actual, preds, zero_division=0)),
                "recall": float(recall_score(actual, preds, zero_division=0)),
                "positive_rate": float(actual.mean()),
            }
            if actual.min() != actual.max():  # both classes present
                floor = self.pr_auc_floor.get(window, PRAUC_FLOOR_DEFAULT)
                metrics["pr_auc"] = float(average_precision_score(actual, probs))
                metrics["pr_auc_floor"] = float(floor)
                metrics["below_floor"] = metrics["pr_auc"] < floor
            out[window] = metrics
        return out

    def metrics(self) -> dict:
        return {
            "n_predictions": len(self.predictions),
            "n_outcomes": sum(len(v) for v in self.outcomes.values()),
            "avg_latency_ms": round(float(np.mean(self.latencies)), 2) if self.latencies else 0.0,
            "prediction_rate_by_tier": dict(self.tier_counts),
            "performance": self.performance(),
            "drift_status": self._last_drift_status,
        }
