"""Adversarial paraphrase robustness tests for the LLM interpreter.

None of these notes are drawn from the public sample pack -- they are
hand-written, hand-verified against the Problem Statement's directive rules,
to check the interpreter generalizes beyond memorized public wording. This is
exactly the kind of robustness the hidden judge set is designed to probe
(Problem Statement section 11.4: "the same underlying rule may be written
differently in hidden cases").

Requires a configured LLM API key; auto-skips otherwise.
"""

from __future__ import annotations

import os

import pytest

from app import guardrails, llm_interpreter

CAPACITY = 200.0

provider = os.environ.get("LLM_PROVIDER", "openai").lower()
has_key = (
    (provider == "openai" and os.environ.get("OPENAI_API_KEY"))
    or (provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"))
)

pytestmark = pytest.mark.skipif(not has_key, reason="no LLM API key configured for this provider")

SINGLE_NOTE_CASES = [
    (
        "Cloud cover this afternoon between 2 and 4 PM will cut solar generation roughly in half.",
        dict(applies=True, directive_type="solar_reduction", hours=[14, 15], factor=0.5),
    ),
    (
        "Between 09:00 and 11:00 the array will be offline for inspection, so treat solar as zero.",
        dict(applies=True, directive_type="solar_reduction", hours=[9, 10], factor=0.0),
    ),
    (
        "From 4 to 6 in the afternoon expect only a third of normal panel output due to dust buildup.",
        dict(applies=True, directive_type="solar_reduction", hours=[16, 17], factor=1 / 3),
    ),
    (
        "The battery must never fall under a quarter of its full capacity between 10 PM and midnight.",
        dict(applies=True, directive_type="minimum_battery_reserve", hours=[22, 23], minimum_energy_kwh=50.0),
    ),
    (
        "Facilities wants at least 75 kWh held back as backup power from 5 AM to 7 AM.",
        dict(applies=True, directive_type="minimum_battery_reserve", hours=[5, 6], minimum_energy_kwh=75.0),
    ),
    (
        "The charging circuit breaker will be locked out for safety inspection between 3 and 4 AM.",
        dict(applies=True, directive_type="no_charge_window", hours=[3]),
    ),
    (
        "Please suspend all battery charging from 9 PM to 11 PM while the inverter firmware updates.",
        dict(applies=True, directive_type="no_charge_window", hours=[21, 22]),
    ),
    (
        "Engineers are running a load-bank test on the battery bus from 10 to 11 AM, "
        "so it must not feed power out during that time.",
        dict(applies=True, directive_type="no_discharge_window", hours=[10]),
    ),
    (
        "Utility has asked us to stay under 90 kWh of grid draw each hour between 7 and 9 AM "
        "during peak network stress.",
        dict(applies=True, directive_type="max_grid_window", hours=[7, 8], max_grid_kwh=90.0),
    ),
    ("The library will extend its hours until 11 PM starting next Monday.", dict(applies=False, directive_type="no_op")),
    ("Parking permits for the east lot increase to 500 taka starting next semester.", dict(applies=False, directive_type="no_op")),
    ("IT will restart the campus wifi routers between 2 and 3 AM tonight.", dict(applies=False, directive_type="no_op")),
    ("The vice chancellor's office moved tomorrow's 10 AM meeting to 1 PM.", dict(applies=False, directive_type="no_op")),
    # --- regression: windows ending at "10 PM" (hour 22) were consistently
    # truncated by one hour live in production -- found by re-testing the
    # official public sample pack (SAMPLE-07, SAMPLE-10) against the
    # deployed service, not by the original stress-test set. ---
    (
        "Keep at least 90 kWh in the battery from 6 PM until 10 PM for emergency services.",
        dict(applies=True, directive_type="minimum_battery_reserve", hours=[18, 19, 20, 21], minimum_energy_kwh=90.0),
    ),
    (
        "Grid intake must stay at or below 190 kWh from 7 PM until 10 PM while the substation is constrained.",
        dict(applies=True, directive_type="max_grid_window", hours=[19, 20, 21], max_grid_kwh=190.0),
    ),
    # Note: the "X PM and midnight" / cross-midnight-boundary regression is
    # already covered by the "quarter of its full capacity" case above.
    # --- directive-type discrimination: worded to sound like a different
    # directive than the correct one, so the model can't pattern-match on
    # surface keywords (e.g. "battery" -> always reserve). ---
    (
        "To protect battery warranty terms, avoid drawing any extra power from it during the "
        "6 to 7 PM evening peak.",
        dict(applies=True, directive_type="no_discharge_window", hours=[18]),
    ),
    (
        "Overcast skies are expected all afternoon, cutting panel output by around 30% "
        "from 1 to 5 PM.",
        dict(applies=True, directive_type="solar_reduction", hours=[13, 14, 15, 16], factor=0.7),
    ),
    (
        "Ensure the campus never pulls more than 300 kWh straight from the utility line "
        "during the 5 to 7 PM evening rush.",
        dict(applies=True, directive_type="max_grid_window", hours=[17, 18], max_grid_kwh=300.0),
    ),
    (
        "Starting next month, unused rooftop solar capacity will be sold back to the grid "
        "under a new net-metering agreement.",
        dict(applies=False, directive_type="no_op"),
    ),
    (
        "The battery charger tripped its breaker and will stay offline for charging purposes "
        "from 2 AM to 4 AM until a technician resets it.",
        dict(applies=True, directive_type="no_charge_window", hours=[2, 3]),
    ),
    (
        # Cross-midnight window: "11 PM to 1 AM" covers hour 23 and hour 0.
        # Guardrails require ascending order, so the correct representation
        # is [0, 23], not chronological [23, 0].
        "Security wants 40 kWh guaranteed available in the battery at all times between "
        "11 PM and 1 AM in case of a blackout.",
        dict(applies=True, directive_type="minimum_battery_reserve", hours=[0, 23], minimum_energy_kwh=40.0),
    ),
]


def _assert_matches(got: dict, expected: dict) -> None:
    assert got["applies"] == expected["applies"], f"applies: got {got['applies']}, want {expected['applies']}"
    assert got["directive_type"] == expected["directive_type"], (
        f"type: got {got['directive_type']}, want {expected['directive_type']}"
    )
    if not expected["applies"]:
        return
    adj = got["structured_adjustment"] or {}
    if "hours" in expected:
        assert adj.get("hours") == expected["hours"], f"hours: got {adj.get('hours')}, want {expected['hours']}"
    for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if key in expected:
            got_val = adj.get(key)
            assert got_val is not None and abs(got_val - expected[key]) <= 0.02, (
                f"{key}: got {got_val}, want {expected[key]}"
            )


@pytest.mark.parametrize("note,expected", SINGLE_NOTE_CASES, ids=[c[0][:40] for c in SINGLE_NOTE_CASES])
def test_single_note_paraphrase(note, expected):
    def check(raw):
        return guardrails.validate_directive_interpretation(raw, [note], CAPACITY)

    directives = llm_interpreter.interpret_notes([note], CAPACITY, check)
    _assert_matches(directives[0], expected)


MULTI_NOTE_CASES = [
    (
        [
            "Solar panels will run at only 60% capacity from noon to 2 PM for a firmware update.",
            "Keep half of the battery in reserve from midnight to 2 AM for emergency dispatch readiness.",
            "The vending machines in building C are being restocked with energy drinks tomorrow.",
        ],
        [
            dict(applies=True, directive_type="solar_reduction", hours=[12, 13], factor=0.6),
            dict(applies=True, directive_type="minimum_battery_reserve", hours=[0, 1], minimum_energy_kwh=120.0),
            dict(applies=False, directive_type="no_op"),
        ],
        240.0,
    ),
    (
        [
            "Grid supply from the substation is limited to 150 units per hour between 6 and 8 PM tonight.",
            "The annual energy fair poster competition submission deadline is next Friday.",
        ],
        [
            dict(applies=True, directive_type="max_grid_window", hours=[18, 19], max_grid_kwh=150.0),
            dict(applies=False, directive_type="no_op"),
        ],
        240.0,
    ),
    (
        [
            "Battery charging is completely disabled from 1 AM to 3 AM tonight for cell balancing maintenance.",
            "Discharging from the battery must also stop between 1 AM and 3 AM during the same maintenance.",
            "Electricity bills for the semester will be emailed to department heads next week.",
        ],
        [
            dict(applies=True, directive_type="no_charge_window", hours=[1, 2]),
            dict(applies=True, directive_type="no_discharge_window", hours=[1, 2]),
            dict(applies=False, directive_type="no_op"),
        ],
        240.0,
    ),
]


@pytest.mark.parametrize(
    "notes,expected_list,capacity", MULTI_NOTE_CASES, ids=[f"multi-note-{i}" for i in range(len(MULTI_NOTE_CASES))]
)
def test_multi_note_request(notes, expected_list, capacity):
    def check(raw):
        return guardrails.validate_directive_interpretation(raw, notes, capacity)

    directives = llm_interpreter.interpret_notes(notes, capacity, check)
    directives.sort(key=lambda d: d["note_index"])
    for got, expected in zip(directives, expected_list):
        _assert_matches(got, expected)
