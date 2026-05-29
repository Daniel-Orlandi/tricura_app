"""Score the trained models directly, without the API.

Run from the repo root (after `modeling.ipynb` has produced models/ and artifacts/):

    uv run python score_example.py

Demonstrates all three predictions via the shared `ModelBundle` inference layer.
"""

from __future__ import annotations

import json

import pandas as pd

from serving.inference import ModelBundle


def main() -> None:
    bundle = ModelBundle()

    print("=" * 60)
    print("1. Risk forecast from a feature payload")
    print("=" * 60)
    # missing features default to 0 (matching training-time fill)
    risk = bundle.predict_risk(
        {
            "age_at_admission": 88,
            "prior_incident_count": 3,
            "prior_transfer_count": 2,
            "active_diagnoses": 11,
        }
    )
    for fc in risk["forecasts"]:
        print(f"  {fc['window']:>4}: P(incident)={fc['probability']:.3f}  tier={fc['risk_tier']}")
    print(f"  expected_cost: ${risk['expected_cost']:,.2f}")

    print("\n" + "=" * 60)
    print("2. Risk forecast by resident_id (bundled demo snapshot)")
    print("=" * 60)
    serving = pd.read_parquet("artifacts/serving_features.parquet")
    rid = serving.index[0]
    by_id = bundle.predict_risk(bundle.lookup_features(rid))
    print(f"  resident_id: {rid}")
    for fc in by_id["forecasts"]:
        print(f"  {fc['window']:>4}: P(incident)={fc['probability']:.3f}  tier={fc['risk_tier']}")

    print("\n" + "=" * 60)
    print("3. Admission (day-3 intake) triage — thin model + facility risk")
    print("=" * 60)
    fac_id = pd.read_parquet("data/residents.parquet")["facility_id"].iloc[0] \
        if __import__("pathlib").Path("data/residents.parquet").exists() else None
    adm = bundle.predict_admission_risk(
        {"age_at_admission": 84, "prior_incident_count": 2, "dx_fall_hist": 1, "dx_dementia": 1},
        facility_id=fac_id,
    )
    for fc in adm["forecasts"]:
        print(f"  {fc['window']:>4}: P(incident)={fc['probability']:.3f}")

    print("\n" + "=" * 60)
    print("4. Incident type -> expected cost given an incident")
    print("=" * 60)
    typ = bundle.predict_incident_type(
        {"prior_fall_count": 2, "prior_wound_count": 0, "dx_chap_M": 3, "age_at_incident": 85}
    )
    print("  type probabilities:")
    for cls, p in sorted(typ["type_probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"    {cls:<12} {p:.3f}")
    print(f"  E[cost | incident]: ${typ['expected_cost_given_incident']:,.2f}")


if __name__ == "__main__":
    main()
