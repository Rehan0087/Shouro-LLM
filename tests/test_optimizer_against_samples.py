"""Validates the guardrail + optimizer + replay-validator pipeline against all
10 public sample cases, using each case's ground-truth directive_interpretation
directly (bypassing the LLM so this test needs no API key).

This is the fast, LLM-free way to catch bugs in the deterministic math before
ever spending an LLM call on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import guardrails, optimizer, replay_validator

SAMPLES_PATH = Path(__file__).parent.parent / "sample_cases" / "public_sample_cases.json"


def load_cases():
    with open(SAMPLES_PATH) as f:
        data = json.load(f)
    return data["cases"]


CASES = load_cases()


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_guardrails_accept_ground_truth(case):
    inp = case["input"]
    expected = case["expected_output"]
    is_valid, errors, normalized = guardrails.validate_directive_interpretation(
        expected["directive_interpretation"],
        inp["operator_notes"],
        inp["battery"]["capacity_kwh"],
    )
    assert is_valid, f"{case['id']}: guardrails rejected ground-truth directives: {errors}"


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_optimizer_produces_valid_schedule(case):
    inp = case["input"]
    expected = case["expected_output"]

    _, _, directives = guardrails.validate_directive_interpretation(
        expected["directive_interpretation"],
        inp["operator_notes"],
        inp["battery"]["capacity_kwh"],
    )

    hours = [h for h in inp["hours"]]
    result = optimizer.optimize_schedule(hours, inp["battery"], directives)

    errors = replay_validator.replay_schedule(hours, inp["battery"], directives, result["hourly_plan"])
    assert not errors, f"{case['id']}: replay validation failed: {errors}"


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_optimizer_cost_matches_reference(case):
    inp = case["input"]
    expected = case["expected_output"]

    _, _, directives = guardrails.validate_directive_interpretation(
        expected["directive_interpretation"],
        inp["operator_notes"],
        inp["battery"]["capacity_kwh"],
    )

    result = optimizer.optimize_schedule(inp["hours"], inp["battery"], directives)

    ref_cost = expected["total_cost_bdt"]
    our_cost = result["total_cost_bdt"]
    # Our LP finds a global optimum; the reference is also claimed optimal, so
    # costs should match closely. Generous tolerance for solver-level ties.
    assert our_cost <= ref_cost + 1.0, (
        f"{case['id']}: our cost {our_cost} is worse than reference optimal {ref_cost}"
    )
    assert abs(our_cost - ref_cost) < 5.0, (
        f"{case['id']}: cost differs a lot from reference (ours={our_cost}, ref={ref_cost}) -- investigate"
    )
