"""Model loading and inference for the risk and incident-type models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "2024-snapshot-v1")


class ModelBundle:
    """Loads the dynamic (risk) models, the Stage-3 type model, and serving metadata."""

    def __init__(self) -> None:
        self.meta = json.loads((ARTIFACTS_DIR / "serving_meta.json").read_text())
        self.windows: list[str] = self.meta["windows"]
        self.risk_tiers: dict = self.meta["risk_tiers"]
        self.mean_cost: float = self.meta["mean_cost_per_incident"]
        self.cost_map: dict = self.meta["cost_map"]

        self.dyn_features: dict = joblib.load(MODELS_DIR / "dynamic_feature_cols.pkl")
        self.dyn_models = {
            w: joblib.load(MODELS_DIR / f"dynamic_{w}.pkl") for w in self.windows
        }

        # thin (day-3 intake) models — early triage; uses facility baseline risk
        self.thin_features: list[str] = joblib.load(MODELS_DIR / "stage1_feature_cols.pkl")
        self.thin_windows = ["6m", "12m"]
        self.thin_models = {
            w: joblib.load(MODELS_DIR / f"stage1_propensity_{w}.pkl") for w in self.thin_windows
        }
        # per-window facility incident-rate map (target-encoded on training data)
        self.facility_risk = json.loads((ARTIFACTS_DIR / "facility_risk.json").read_text())

        self.type_model = joblib.load(MODELS_DIR / "stage3_incident_type.pkl")
        self.type_features: list[str] = joblib.load(MODELS_DIR / "stage3_feature_cols.pkl")
        self.type_classes = [str(c) for c in self.type_model.classes_]

        # per-resident feature matrix for demo lookups
        self.serving_features = pd.read_parquet(ARTIFACTS_DIR / "serving_features.parquet")

    # ----- helpers -----
    def _row(self, features: dict[str, float], cols: list[str]) -> pd.DataFrame:
        """Build a single-row frame in the exact training column order; missing -> 0."""
        return pd.DataFrame([{c: float(features.get(c, 0.0)) for c in cols}])[cols]

    def _tier(self, window: str, prob: float) -> str:
        cuts = self.risk_tiers[window]
        if prob >= cuts["high"]:
            return "high"
        if prob >= cuts["medium"]:
            return "medium"
        return "low"

    # ----- public API -----
    def predict_risk(self, features: dict[str, float]) -> dict:
        forecasts = []
        prob_6m = None
        for w in self.windows:
            cols = self.dyn_features[w]
            p = float(self.dyn_models[w].predict_proba(self._row(features, cols))[0, 1])
            forecasts.append({"window": w, "probability": p, "risk_tier": self._tier(w, p)})
            if w == "6m":
                prob_6m = p
        if prob_6m is None:  # fall back to the longest window present
            prob_6m = forecasts[-1]["probability"]
        return {
            "forecasts": forecasts,
            "expected_cost": round(prob_6m * self.mean_cost, 2),
            "model_version": MODEL_VERSION,
        }

    def predict_admission_risk(self, features: dict[str, float],
                               facility_id: str | None = None) -> dict:
        """Day-3 intake triage from the thin model. Injects the per-window facility
        baseline-risk feature from facility_id (falls back to the global rate if unknown)."""
        forecasts = []
        prob_6m = None
        for w in self.thin_windows:
            fac = self.facility_risk["windows"][w]
            fr = fac["rates"].get(facility_id, fac["global"]) if facility_id else fac["global"]
            feat = {**features, "facility_risk": fr}
            p = float(self.thin_models[w].predict_proba(self._row(feat, self.thin_features))[0, 1])
            forecasts.append({"window": w, "probability": p})
            if w == "6m":
                prob_6m = p
        if prob_6m is None:
            prob_6m = forecasts[0]["probability"]
        return {
            "forecasts": forecasts,
            "expected_cost": round(prob_6m * self.mean_cost, 2),
            "model_version": MODEL_VERSION,
        }

    def predict_incident_type(self, features: dict[str, float]) -> dict:
        proba = self.type_model.predict_proba(self._row(features, self.type_features))[0]
        type_probs = {c: float(p) for c, p in zip(self.type_classes, proba)}
        exp_cost = sum(type_probs[c] * self.cost_map.get(c, self.mean_cost) for c in type_probs)
        return {
            "type_probabilities": type_probs,
            "expected_cost_given_incident": round(exp_cost, 2),
            "model_version": MODEL_VERSION,
        }

    def lookup_features(self, resident_id: str) -> dict[str, float] | None:
        if resident_id not in self.serving_features.index:
            return None
        return self.serving_features.loc[resident_id].to_dict()
