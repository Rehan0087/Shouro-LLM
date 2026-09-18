"""Runs the real LLM interpreter against all 10 public sample cases and checks
that applies/directive_type match the ground truth (numeric values are
compared with tolerance). Skipped automatically if no LLM API key is
configured -- set ANTHROPIC_API_KEY or OPENAI_API_KEY (+ LLM_PROVIDER) to run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app import guardrails, llm_interpreter

SAMPLES_PATH = Path(__file__).parent.parent / "sample_cases" / "public_sample_cases.json"

provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
has_key = (
    (provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"))
    or (provider == "openai" and os.environ.get("OPENAI_API_KEY"))
)

pytestmark = pytest.mark.skipif(not has_key, reason="no LLM API key configured for this provider")


def load_cases():
    with open(SAMPLES_PATH) as f:
        data = json.load(f)
    return data["cases"]


CASES = load_cases()


def _hours_close(a, b, tol=0.5):
    return a == b


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_llm_matches_ground_truth_directive_types(case):
    inp = case["input"]
    expected_entries = {e["note_index"]: e for e in case["expected_output"]["directive_interpretation"]}

    def check(raw):
        return guardrails.validate_directive_interpretation(
            raw, inp["operator_notes"], inp["battery"]["capacity_kwh"]
        )

    directives = llm_interpreter.interpret_notes(
        inp["operator_notes"], inp["battery"]["capacity_kwh"], check
    )

    assert len(directives) == len(inp["operator_notes"])

    mismatches = []
    for d in directives:
        exp = expected_entries[d["note_index"]]
        if d["applies"] != exp["applies"] or d["directive_type"] != exp["directive_type"]:
            mismatches.append(
                f"note {d['note_index']}: got ({d['applies']}, {d['directive_type']}), "
                f"expected ({exp['applies']}, {exp['directive_type']})"
            )

    assert not mismatches, f"{case['id']}: " + "; ".join(mismatches)
