"""Pydantic request/response models — these drive the OpenAPI schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: list[str]
    n_serving_residents: int


class RiskRequest(BaseModel):
    """Feature payload for a single resident at the day-30 snapshot.

    In production these come from the feature store / batch pipeline that
    materialises the same vectors used in training. Missing features default to 0
    (matching the training-time fill).
    """

    resident_id: str | None = Field(default=None, description="Optional identifier for logging")
    features: dict[str, float] = Field(..., description="Dynamic feature name -> value")

    model_config = {
        "json_schema_extra": {
            "example": {
                "resident_id": "demo-001",
                "features": {
                    "age_at_admission": 84.2,
                    "prior_incident_count": 2,
                    "prior_transfer_count": 1,
                    "total_diagnoses": 12,
                    "active_diagnoses": 9,
                    "vital_std_bp_-_systolic": 14.3,
                },
            }
        }
    }


class HorizonForecast(BaseModel):
    window: str = Field(..., description="Forecast horizon, e.g. '20d'")
    probability: float = Field(..., description="P(incident within the window)")
    risk_tier: str = Field(..., description="low | medium | high")


class RiskResponse(BaseModel):
    resident_id: str | None
    forecasts: list[HorizonForecast]
    expected_cost: float = Field(..., description="P(incident, 6m) x mean cost per incident (USD)")
    model_version: str


class AdmissionRiskRequest(BaseModel):
    """Intake (day-3) feature payload. `facility_id` drives the facility baseline-risk feature."""

    resident_id: str | None = Field(default=None, description="Optional identifier for logging")
    facility_id: str | None = Field(
        default=None, description="Facility id → facility baseline risk; global rate if unknown"
    )
    features: dict[str, float] = Field(..., description="Intake feature name -> value")

    model_config = {
        "json_schema_extra": {
            "example": {
                "resident_id": "demo-001",
                "facility_id": "0240d706-3348-5117-8d03-b06c5141e8c0",
                "features": {
                    "age_at_admission": 84,
                    "prior_incident_count": 2,
                    "days_since_last_discharge": 30,
                    "dx_fall_hist": 1,
                    "dx_dementia": 1,
                    "total_diagnoses": 11,
                },
            }
        }
    }


class AdmissionForecast(BaseModel):
    window: str = Field(..., description="Forecast horizon, e.g. '6m'")
    probability: float = Field(..., description="P(incident within the window)")


class AdmissionRiskResponse(BaseModel):
    resident_id: str | None
    forecasts: list[AdmissionForecast]
    expected_cost: float = Field(..., description="P(incident, 6m) x mean cost per incident (USD)")
    model_version: str
    note: str = Field(
        default="Day-3 intake triage (thin model, ROC-AUC ~0.74-0.75). "
        "Use /predict/risk once the resident reaches day 30 for the stronger forecast.",
        description="Context on the thin model",
    )


class IncidentTypeRequest(BaseModel):
    resident_id: str | None = None
    features: dict[str, float] = Field(..., description="Stage-3 feature name -> value")

    model_config = {
        "json_schema_extra": {
            "example": {
                "resident_id": "demo-001",
                "features": {
                    "age_at_incident": 85,
                    "prior_fall_count": 2,
                    "prior_wound_count": 0,
                    "prior_altercation_count": 0,
                    "dx_chap_M": 3,
                    "dx_chap_L": 1,
                },
            }
        }
    }


class IncidentTypeResponse(BaseModel):
    resident_id: str | None
    type_probabilities: dict[str, float]
    expected_cost_given_incident: float = Field(..., description="sum_type P(type) x cost[type] (USD)")
    model_version: str


class DriftRequest(BaseModel):
    """Optional explicit batch; if omitted, the recent prediction log is used."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [
                    {"age_at_admission": 84, "prior_incident_count": 2},
                    {"age_at_admission": 79, "prior_incident_count": 0},
                ]
            }
        }
    }

    features: list[dict[str, float]] | None = Field(
        default=None, description="Batch of feature payloads to test for drift"
    )


class FeatureDrift(BaseModel):
    feature: str
    psi: float
    severity: str = Field(..., description="none | moderate | significant")


class DriftResponse(BaseModel):
    n_samples: int
    n_features_checked: int
    drifted_features: list[FeatureDrift]
    overall_status: str = Field(..., description="ok | warning | alert")


class OutcomeRequest(BaseModel):
    """Submitted once a forecast window closes and the true outcome is known."""

    resident_id: str
    window: str = Field(..., description="Forecast window the prediction was for, e.g. '6m'")
    predicted_probability: float
    incident_occurred: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "resident_id": "demo-001",
                "window": "6m",
                "predicted_probability": 0.42,
                "incident_occurred": True,
            }
        }
    }


class OutcomeResponse(BaseModel):
    window: str
    rolling_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Rolling precision/recall/PR-AUC for the window (once enough labels exist)",
    )


class MetricsResponse(BaseModel):
    n_predictions: int
    n_outcomes: int
    avg_latency_ms: float
    prediction_rate_by_tier: dict[str, int]
    performance: dict[str, dict[str, float]]
    drift_status: str
