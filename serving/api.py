"""FastAPI app exposing the Tricura incident-risk models with OpenAPI docs.

Run locally:   uvicorn serving.api:app --reload
OpenAPI docs:  http://localhost:8000/docs   (Swagger UI)  /  /redoc  /  /openapi.json
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .inference import MODEL_VERSION, ModelBundle
from .monitoring import Monitor
from .schemas import (
    AdmissionRiskRequest,
    AdmissionRiskResponse,
    DriftRequest,
    DriftResponse,
    HealthResponse,
    IncidentTypeRequest,
    IncidentTypeResponse,
    MetricsResponse,
    OutcomeRequest,
    OutcomeResponse,
    RiskRequest,
    RiskResponse,
)

bundle: ModelBundle
monitor: Monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle, monitor
    bundle = ModelBundle()
    monitor = Monitor()
    yield


app = FastAPI(
    title="Tricura Incident-Risk API",
    description=(
        "Forecasts resident incident risk (multiple horizons), predicts likely incident "
        "type → claim cost, and exposes production monitoring (drift, performance)."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        models_loaded=[f"dynamic_{w}" for w in bundle.windows] + ["stage3_incident_type"],
        n_serving_residents=int(bundle.serving_features.shape[0]),
    )


@app.post("/predict/risk", response_model=RiskResponse, tags=["forecast"])
def predict_risk(req: RiskRequest) -> RiskResponse:
    t0 = time.perf_counter()
    result = bundle.predict_risk(req.features)
    latency_ms = (time.perf_counter() - t0) * 1000
    monitor.log_prediction(req.features, result, latency_ms)
    return RiskResponse(resident_id=req.resident_id, **result)


@app.post("/predict/admission-risk", response_model=AdmissionRiskResponse, tags=["forecast"])
def predict_admission_risk(req: AdmissionRiskRequest) -> AdmissionRiskResponse:
    """Stage-1 day-3 intake triage (thin model). Uses facility baseline risk; see the `note`."""
    result = bundle.predict_admission_risk(req.features, req.facility_id)
    return AdmissionRiskResponse(resident_id=req.resident_id, **result)


@app.get("/predict/{resident_id}", response_model=RiskResponse, tags=["forecast"])
def predict_by_id(resident_id: str) -> RiskResponse:
    """Demo convenience: look up a resident's precomputed feature vector and forecast."""
    features = bundle.lookup_features(resident_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"resident_id '{resident_id}' not found")
    t0 = time.perf_counter()
    result = bundle.predict_risk(features)
    monitor.log_prediction(features, result, (time.perf_counter() - t0) * 1000)
    return RiskResponse(resident_id=resident_id, **result)


@app.post("/predict/incident-type", response_model=IncidentTypeResponse, tags=["forecast"])
def predict_incident_type(req: IncidentTypeRequest) -> IncidentTypeResponse:
    result = bundle.predict_incident_type(req.features)
    return IncidentTypeResponse(resident_id=req.resident_id, **result)


@app.post("/monitoring/drift", response_model=DriftResponse, tags=["monitoring"])
def drift(req: DriftRequest) -> DriftResponse:
    report = monitor.drift_report(req.features)
    return DriftResponse(**report)


@app.post("/monitoring/outcome", response_model=OutcomeResponse, tags=["monitoring"])
def submit_outcome(req: OutcomeRequest) -> OutcomeResponse:
    """Submit a realised outcome once a forecast window has closed (delayed labels)."""
    metrics = monitor.record_outcome(req.window, req.predicted_probability, req.incident_occurred)
    return OutcomeResponse(window=req.window, rolling_metrics=metrics)


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
def metrics() -> MetricsResponse:
    return MetricsResponse(**monitor.metrics())
