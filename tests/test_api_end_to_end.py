"""End-to-end HTTP API test using FastAPI's TestClient. The LLM interpreter is
monkeypatched to return each case's ground-truth directives directly, so this
suite verifies request/response schema compliance and pipeline wiring without
needing a real API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import guardrails, llm_interpreter
from app.main import app

SAMPLES_PATH = Path(__file__).parent.parent / "sample_cases" / "public_sample_cases.json"


def load_cases():
    with open(SAMPLES_PATH) as f:
        data = json.load(f)
    return data["cases"]


CASES = load_cases()

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_redirects_to_docs_instead_of_404():
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/docs"


def test_malformed_json_returns_400():
    resp = client.post(
        "/optimize-energy",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_missing_fields_returns_400():
    resp = client.post("/optimize-energy", json={"scenario_id": "X"})
    assert resp.status_code == 400


def test_infeasible_scenario_returns_controlled_500(monkeypatch):
    """A scenario the optimizer genuinely cannot satisfy must fail safely
    (500, no stack trace, no crash) rather than return an invalid schedule."""
    case = CASES[0]
    inp = dict(case["input"])
    inp["battery"] = {
        "capacity_kwh": 50.0,
        "initial_energy_kwh": 25.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 5.0,
        "max_discharge_kwh_per_hour": 5.0,
    }
    inp["hours"] = [
        {"hour": h, "demand_kwh": 500.0, "solar_kwh": 0.0, "tariff_bdt_per_kwh": 10.0} for h in range(24)
    ]

    def fake_interpret_notes(operator_notes, battery_capacity_kwh, guardrail_check):
        # Force a hard grid cap of 0 with far too little solar/battery to
        # cover demand -- genuinely unsatisfiable energy balance.
        entries = [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": list(range(24)), "max_grid_kwh": 0.0},
                "explanation": "test",
            }
        ]
        entries += [
            {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "test",
            }
            for i in range(1, len(operator_notes))
        ]
        return entries

    monkeypatch.setattr("app.pipeline.llm_interpreter.interpret_notes", fake_interpret_notes)

    resp = client.post("/optimize-energy", json=inp)
    assert resp.status_code == 500
    body = resp.json()
    assert "error" in body
    assert "Traceback" not in json.dumps(body)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_full_pipeline_with_ground_truth_llm(case, monkeypatch):
    inp = case["input"]
    expected = case["expected_output"]

    _, _, ground_truth_directives = guardrails.validate_directive_interpretation(
        expected["directive_interpretation"],
        inp["operator_notes"],
        inp["battery"]["capacity_kwh"],
    )

    def fake_interpret_notes(operator_notes, battery_capacity_kwh, guardrail_check):
        is_valid, errors, normalized = guardrail_check(
            [
                {
                    "note_index": d["note_index"],
                    "applies": d["applies"],
                    "directive_type": d["directive_type"],
                    "structured_adjustment": d["structured_adjustment"],
                    "explanation": d["explanation"],
                }
                for d in ground_truth_directives
            ]
        )
        assert is_valid, errors
        return normalized

    monkeypatch.setattr(llm_interpreter, "interpret_notes", fake_interpret_notes)
    monkeypatch.setattr("app.pipeline.llm_interpreter.interpret_notes", fake_interpret_notes)

    resp = client.post("/optimize-energy", json=inp)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["scenario_id"] == case["id"]
    assert len(body["directive_interpretation"]) == len(inp["operator_notes"])
    assert len(body["hourly_plan"]) == 24
    assert body["total_grid_kwh"] > 0
    assert body["peak_grid_kwh"] == max(h["grid_kwh"] for h in body["hourly_plan"])

    recalculated_cost = sum(
        h["grid_kwh"] * hr["tariff_bdt_per_kwh"]
        for h, hr in zip(body["hourly_plan"], sorted(inp["hours"], key=lambda x: x["hour"]))
    )
    assert abs(recalculated_cost - body["total_cost_bdt"]) < 0.05
