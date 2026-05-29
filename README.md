# Tricura — Incident Risk & Severity Modeling


## What was built

A three-part scoring system. All models share one algorithm (**CatBoost**) to keep the system
simple to maintain, and all use **time-based validation** (train on earlier admissions, test on
later ones) to mirror real deployment. Because incidents are rare and recall matters more than
raw accuracy, the primary metric is **PR-AUC** (average precision).

| Stage | Model | Snapshot | Horizons | Purpose |
|---|---|---|---|---|
| **1 — Thin** | CatBoost (binary) | Day-3 intake | 6m, 12m | Early triage after the admission workup |
| **2 — Dynamic** | CatBoost (binary) | Day 30, refreshed monthly | 20d, 30d, 6m | Ongoing risk; short horizons drive prevention timing |
| **3 — Type/Cost** | CatBoost (multiclass) | At incident | Fall / Wound / Altercation / Other | Which incident type → maps to claim cost |

The stages compose a risk score for intervention targeting:

```
expected_cost(resident) = P(incident in window)   [Stage 1/2]
                        × E[cost | incident]       [cost per incident]
```

**As implemented:** `/predict/risk` returns `P(incident, 6m) × mean_cost_per_incident` (a
population-average cost, ~$3.5k), and `/predict/incident-type` separately returns the
resident-specific `E[cost | incident] = Σ P(type) × cost[type]` from Stage 3. The
per-resident Stage-3 cost into the risk endpoint is left as a refinement — given the narrow
per-type cost spread, the population average is a close approximation and most of the dollar
signal comes from P(incident).

## Data

17 anonymized parquet tables (~280 MB): 3,000 residents, 3,578 incidents, plus diagnoses,
medications, vitals, physician orders, needs, care plans, labs, hospital admissions/transfers,
and therapy. Joined on `resident_id` / `incident_id`. See `eda.ipynb` for the full exploration;
`modeling.ipynb` contains the models.

## Key modeling decisions

- **Forecast windows.** Targets are "incident within N days of the snapshot date," bounded by
  discharge — Tricura's exposure ends when a resident leaves. This avoids mislabelling short-stay
  residents as low-risk.

- **Intake (day-3) vs day-30 snapshots.** Little data exists exactly at admission, so the thin
  model scores at a 3-day intake window (after the admission workup) and leans on facility
  baseline risk. The day-30 dynamic model is the workhorse; the thin model is the early triage net.

- **Strict leakage control.** Every feature is reconstructed *as of the snapshot date* — e.g. a
  diagnosis counts as "active" only if unresolved *at the snapshot*, not as of today. This avoids leakage of future states.
  
- **Dropped weak/sparse data.** GG and ADL assessments (<5% coverage), document tags, and raw
  vitals as static means were excluded. Vitals re-entered the dynamic model as 30-day
  trend/variability features. Data-entry artifacts (`has_missing_icd`, `is_outpatient`,
  clinically irrelevant ICD chapters) were removed.

- **Imbalance handling.** `auto_class_weights="Balanced"`; rare incident types folded into "Other."

- **Per-window feature policy.** Monitoring features (`vital_count_*`) help the 20-day
  window but add noise over longer horizons, so they're dropped from 30d/6m. SHAP-based feature
  selection is applied only where there are enough positives to be reliable (6m), 
  so 20 and 30 day window keeps the full feature set, 30 day also keeps the vitals data.

## Results
**Thin model (day-3 intake):** ROC-AUC ~0.74–0.75 (PR-AUC ~0.43). A pure day-0 model is
near-random (~0.52) because little clinical data exists at admission; three changes fix it —
a **3-day intake window** (captures the admission workup), **facility baseline risk**
(target-encoded per facility, the single strongest feature), and **fall-risk ICD flags**
(history of falling, dementia, gait, Parkinson's). A usable early-triage tool.

**Dynamic model (day-30) — the primary product:**

| Horizon | Test PR-AUC | Test ROC-AUC |
|---|---|---|
| 20 days | 0.365 | 0.807 |
| 30 days | 0.342 | 0.780 |
| 6 months | 0.359 | 0.801 |

Consistent 0.78–0.81 ROC-AUC, well above the ~8–18% positive base rates. Strongest predictors:
prior incidents, prior hospital transfers, diagnosis burden, and vital-sign variability.

**Type/Cost model (Stage 3):** Macro-F1 0.44 (Fall F1 0.78, Wound 0.43, Altercation 0.37). It
trades raw accuracy (0.67 vs a 0.72 always-Fall baseline) to actually catch minority classes —
48% of Wounds, 40% of Altercations — so it can direct *which* prevention a resident needs.

## limitations

- **RTH ($20k, the costliest category) is out of scope** — it lives in `hospital_transfers`, not
  `incidents`, so the type model can't price it. Expected-cost differentiation is therefore narrow
  ($2.5k–$4.3k); most of the dollar signal comes from Stage 1/2's probability, not the type.

- **Short windows are sparse** (20d ≈ 7% positives), so those models are the least stable.

- **A single facility-year of data** limits seasonal and cross-site generalization.
- A real deployment needs **probability calibration** and a **decision threshold chosen to the
  intervention budget**, plus monthly retraining as the dynamic snapshot rolls forward.

## Serving & monitoring

The models are served behind a **FastAPI** app (auto-generated OpenAPI docs), Dockerized.

```bash
docker compose up --build      # → Swagger UI at http://localhost:8000/docs
curl -X POST localhost:8000/predict/risk -H "Content-Type: application/json" \
  -d '{"features": {"age_at_admission": 88, "prior_incident_count": 3}}'
```

**Endpoints**

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | Liveness + which models loaded |
| POST | `/predict/admission-risk` | Stage-1 day-3 intake triage: P(incident) for 6m/12m (uses `facility_id`) |
| POST | `/predict/risk` | Stage-2 day-30 forecast: P(incident) for 20d/30d/6m + expected cost |
| GET  | `/predict/{resident_id}` | Demo: look up a resident's precomputed features and forecast |
| POST | `/predict/incident-type` | Stage-3: incident-type probabilities + E[cost \| incident] |
| POST | `/monitoring/drift` | PSI feature-drift report (explicit batch, or the recent prediction log) |
| POST | `/monitoring/outcome` | Submit a realised outcome once a window closes (delayed labels) |
| GET  | `/metrics` | Prediction counts, latency, tier mix, rolling performance, drift status |

In production a feature store / batch pipeline materialises the same training feature vectors and
posts them to `/predict/risk`; `/predict/{resident_id}` is a demo convenience backed by a bundled
per-resident snapshot.

**Monitoring strategy** — four complementary signals against silent model decay:

1. **Feature drift (PSI).** Incoming features are binned against the training reference
   distribution (deciles in `artifacts/reference_stats.json`). PSI ≥ 0.20 moderate, ≥ 0.25
   significant. Drift is a batch statistic over the recent prediction log or an explicit batch.
2. **Prediction logging.** Every request/response is appended to `logs/predictions.jsonl`
   (mounted volume) for audit and offline analysis.
3. **Prediction-distribution tracking.** `/metrics` reports the low/medium/high tier mix and
   latency — a sudden tier shift flags change before labels arrive.
4. **Delayed-label performance.** Incident labels only exist *after* a window closes. Outcomes
   submitted to `/monitoring/outcome` drive rolling precision/recall/PR-AUC per window.

**Retraining triggers** (any of): significant feature drift (PSI ≥ 0.25 on key features); rolling
PR-AUC below a **per-window, baseline-relative floor** (`baseline + 0.5×(launch − baseline)`,
where baseline = prevalence = a random model's PR-AUC — floors: 20d 0.22, 30d 0.22, 6m 0.27); or
the monthly cadence as the day-30 snapshot rolls forward. In a full deployment `/metrics` and
`/monitoring/drift` would be scraped into Prometheus/Grafana with alerting on these thresholds.

**Configuration (env vars):** `MODELS_DIR` (`models`), `ARTIFACTS_DIR` (`artifacts`),
`MODEL_VERSION`, `PREDICTION_LOG` (`logs/predictions.jsonl`), `PRAUC_FLOOR` (fallback floor if a
window is missing from metadata).

## Repository

```
eda.ipynb        # exploratory data analysis
modeling.ipynb   # Parts I–VI: features, thin/dynamic/type models, artifact export
models/          # serialized CatBoost models + feature lists + cost map
artifacts/       # serving feature matrix + drift reference stats + metadata
serving/         # FastAPI app (api.py), inference, monitoring — Dockerized
score_example.py # score the models directly, without the API
data/            # source parquet tables
```

## Running the project

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/) (Python 3.12 is installed by uv), and
Docker (only for the serving step).

> The dataset is **not** included in this repo (`data/`). The only manual step is
> to drop the provided parquet files into a `data/` folder at the repo root — everything else
> (models, serving artifacts) is regenerated by step 3.

```
data/
├── residents.parquet
├── incidents.parquet
├── diagnoses.parquet
└── ... (all 17 tables)
```

### 1. Install dependencies

```bash
uv sync
```

### 2. Explore the data (optional)

```bash
uv run jupyter lab eda.ipynb        # or open in VS Code / Jupyter
```

### 3. Train the models and export serving artifacts

Runs the full pipeline (feature engineering → thin, dynamic, and type models) and writes
`models/` and `artifacts/`:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace modeling.ipynb
```

This single run trains **and evaluates all three stages** and writes `models/` + `artifacts/`.
It is required before serving — the API loads the files it produces. Takes a few minutes.

### 3a. Evaluate the models

Evaluation lives in the executed `modeling.ipynb` — open it after step 3. Each stage reports
held-out metrics (PR-AUC primary, ROC-AUC, confusion matrix, feature importance) on a time-based
test split:

| Stage | Notebook part | What to look at |
|---|---|---|
| Thin (day-3 intake) | II–III | per-window PR/ROC, confusion, feature importance (facility risk on top) |
| Dynamic (day-30) | IV | per-window CV + test PR/ROC, feature importance |
| Type/Cost | V | macro-F1, per-class report, confusion matrix, cost mapping |

The two experiment sections near the end show the before/after for the thin-model improvement
and the (negative) facility-risk test on the dynamic model. The top "Key findings" cell
summarises the headline numbers.

### 4. Serve the API

```bash
docker compose up --build           # → http://localhost:8000
```

- Swagger UI: http://localhost:8000/docs
- Health check: `curl localhost:8000/health`

Or run it locally without Docker:

```bash
uv run uvicorn serving.api:app --reload
```

### 5. Make a prediction

```bash
# by feature payload
curl -X POST localhost:8000/predict/risk -H "Content-Type: application/json" \
  -d '{"features": {"age_at_admission": 88, "prior_incident_count": 3}}'

# by resident id (demo lookup against the bundled snapshot)
curl localhost:8000/predict/<resident_id>
```

### Score without the API (direct Python)

The same models can be scored directly — no server needed. `score_example.py` at the repo root
loads the artifacts from step 3 and runs all three predictions (risk forecast by payload, by
resident id, and incident-type/cost):

```bash
uv run python score_example.py
```

It uses the shared inference layer (`serving/inference.py`), the same one the API calls:

```python
from serving.inference import ModelBundle

bundle = ModelBundle()
risk = bundle.predict_risk({"age_at_admission": 88, "prior_incident_count": 3})
print(risk["forecasts"], risk["expected_cost"])
```

The notebook (`modeling.ipynb`) also evaluates every model on the held-out test set with full
metrics and plots.

---

