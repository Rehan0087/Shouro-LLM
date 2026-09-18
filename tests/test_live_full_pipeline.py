"""Runs the REAL end-to-end pipeline (live LLM -> guardrails -> optimizer ->
replay validator) against all 10 public sample cases via the actual HTTP API,
and compares recalculated cost against the reference optimal cost. This is
the strongest signal we have that numeric extraction (hours, factor,
minimum_energy_kwh, max_grid_kwh) is correct, not just directive_type.

Requires a configured LLM API key; auto-skips otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLES_PATH = Path(__file__).parent.parent / "sample_cases" / "public_sample_cases.json"

provider = os.environ.get("LLM_PROVIDER", "openai").lower()
has_key = (
    (provider == "openai" and os.environ.get("OPENAI_API_KEY"))
    or (provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"))
)

pytestmark = pytest.mark.skipif(not has_key, reason="no LLM API key configured for this provider")


def load_cases():
    with open(SAMPLES_PATH) as f:
        data = json.load(f)
    return data["cases"]


CASES = load_cases()

client = TestClient(app)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_live_pipeline_end_to_end(case):
    inp = case["input"]
    expected = case["expected_output"]

    resp = client.post("/optimize-energy", json=inp)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["scenario_id"] == case["id"]
    assert len(body["hourly_plan"]) == 24

    ref_applies = {e["note_index"]: (e["applies"], e["directive_type"]) for e in expected["directive_interpretation"]}
    for entry in body["directive_interpretation"]:
        exp_applies, exp_type = ref_applies[entry["note_index"]]
        assert entry["applies"] == exp_applies, f"note {entry['note_index']}: applies mismatch"
        assert entry["directive_type"] == exp_type, f"note {entry['note_index']}: directive_type mismatch"

    ref_cost = expected["total_cost_bdt"]
    our_cost = body["total_cost_bdt"]
    # Allow a bit more slack here than the ground-truth-only optimizer test,
    # since this also depends on the LLM extracting exact numeric values.
    assert our_cost <= ref_cost + 5.0, f"{case['id']}: cost {our_cost} much worse than reference {ref_cost}"
