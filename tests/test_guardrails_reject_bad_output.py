"""Guardrails must reject malformed/unsupported LLM output rather than
silently accepting it (Problem Statement section 08, 'SAFE FAILURE')."""

from __future__ import annotations

from app.guardrails import validate_directive_interpretation

NOTES = ["note A", "note B"]
CAPACITY = 200.0


def _run(directives):
    return validate_directive_interpretation(directives, NOTES, CAPACITY)


def test_unsupported_directive_type_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": True, "directive_type": "grid_export", "structured_adjustment": {"hours": [1]}, "explanation": "x"},
            {"note_index": 1, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid
    assert any("not a supported directive type" in e for e in errors)


def test_no_op_with_applies_true_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": True, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
            {"note_index": 1, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid


def test_missing_note_index_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid
    assert any("missing directive_interpretation entries" in e for e in errors)


def test_duplicate_note_index_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
            {"note_index": 0, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid


def test_unsorted_hours_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": True, "directive_type": "no_charge_window", "structured_adjustment": {"hours": [5, 3, 4]}, "explanation": "x"},
            {"note_index": 1, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid
    assert any("ascending order" in e for e in errors)


def test_out_of_range_hour_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": True, "directive_type": "no_charge_window", "structured_adjustment": {"hours": [24]}, "explanation": "x"},
            {"note_index": 1, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid


def test_solar_factor_out_of_range_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": True, "directive_type": "solar_reduction", "structured_adjustment": {"hours": [1], "factor": 1.5}, "explanation": "x"},
            {"note_index": 1, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid
    assert any("factor must be between 0 and 1" in e for e in errors)


def test_reserve_above_capacity_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve", "structured_adjustment": {"hours": [1], "minimum_energy_kwh": 999}, "explanation": "x"},
            {"note_index": 1, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid


def test_negative_max_grid_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": True, "directive_type": "max_grid_window", "structured_adjustment": {"hours": [1], "max_grid_kwh": -5}, "explanation": "x"},
            {"note_index": 1, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid


def test_missing_structured_adjustment_keys_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": True, "directive_type": "max_grid_window", "structured_adjustment": {"hours": [1]}, "explanation": "x"},
            {"note_index": 1, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid
    assert any("missing required keys" in e for e in errors)


def test_wrong_count_rejected():
    is_valid, errors, _ = _run(
        [
            {"note_index": 0, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
        ]
    )
    assert not is_valid


def test_valid_minimal_case_accepted():
    is_valid, errors, normalized = _run(
        [
            {"note_index": 0, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "x"},
            {"note_index": 1, "applies": True, "directive_type": "no_discharge_window", "structured_adjustment": {"hours": [18, 19]}, "explanation": "x"},
        ]
    )
    assert is_valid, errors
    assert len(normalized) == 2
