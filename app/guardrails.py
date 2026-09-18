"""Deterministic validation of LLM-produced directive interpretations.

Nothing from the LLM is trusted until it passes every check here (Problem
Statement section 08). On failure we return a list of human-readable error
strings describing exactly what is wrong, which the caller can feed back to
the LLM for a corrective retry.
"""

from __future__ import annotations

from typing import Any

ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

REQUIRED_KEYS = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
}


def _check_hours(hours: Any, errors: list[str], ctx: str) -> bool:
    if not isinstance(hours, list) or len(hours) == 0:
        errors.append(f"{ctx}: 'hours' must be a non-empty list")
        return False
    if not all(isinstance(h, int) and not isinstance(h, bool) for h in hours):
        errors.append(f"{ctx}: 'hours' must contain only integers")
        return False
    if any(h < 0 or h > 23 for h in hours):
        errors.append(f"{ctx}: 'hours' entries must be within 0-23")
        return False
    if len(set(hours)) != len(hours):
        errors.append(f"{ctx}: 'hours' must not contain duplicates")
        return False
    if list(hours) != sorted(hours):
        errors.append(f"{ctx}: 'hours' must be in ascending order")
        return False
    return True


def validate_directive_interpretation(
    directives: list[dict[str, Any]],
    operator_notes: list[str],
    battery_capacity_kwh: float,
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    """Validate LLM output against the guardrails in Problem Statement section 08.

    Returns (is_valid, errors, normalized_directives). normalized_directives
    is only meaningful when is_valid is True; entries are sorted by note_index
    and numeric fields are coerced to float where the schema calls for numbers.
    """
    errors: list[str] = []
    n = len(operator_notes)

    if not isinstance(directives, list):
        return False, ["directive_interpretation must be a list"], []

    if len(directives) != n:
        errors.append(
            f"expected exactly {n} directive_interpretation entries (one per note), got {len(directives)}"
        )

    seen_indices: set[int] = set()
    normalized: list[dict[str, Any]] = []

    for i, d in enumerate(directives):
        ctx = f"entry {i}"
        if not isinstance(d, dict):
            errors.append(f"{ctx}: must be an object")
            continue

        note_index = d.get("note_index")
        if not isinstance(note_index, int) or isinstance(note_index, bool):
            errors.append(f"{ctx}: note_index must be an integer")
            continue
        if note_index < 0 or note_index >= n:
            errors.append(f"{ctx}: note_index {note_index} does not reference an existing note")
            continue
        if note_index in seen_indices:
            errors.append(f"{ctx}: duplicate note_index {note_index}")
            continue
        seen_indices.add(note_index)
        ctx = f"note_index {note_index}"

        directive_type = d.get("directive_type")
        if directive_type not in ALLOWED_TYPES:
            errors.append(f"{ctx}: directive_type '{directive_type}' is not a supported directive type")
            continue

        applies = d.get("applies")
        if not isinstance(applies, bool):
            errors.append(f"{ctx}: applies must be a boolean")
            continue

        structured_adjustment = d.get("structured_adjustment")
        explanation = d.get("explanation", "")
        if not isinstance(explanation, str):
            errors.append(f"{ctx}: explanation must be a string")
            continue

        if directive_type == "no_op":
            if applies is not False:
                errors.append(f"{ctx}: no_op must have applies = false")
                continue
            if structured_adjustment is not None:
                errors.append(f"{ctx}: no_op must have structured_adjustment = null")
                continue
            normalized.append(
                {
                    "note_index": note_index,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": explanation,
                }
            )
            continue

        # Every non-no_op directive
        if applies is not True:
            errors.append(f"{ctx}: non-no_op directives must have applies = true")
            continue
        if not isinstance(structured_adjustment, dict):
            errors.append(f"{ctx}: structured_adjustment must be an object for directive_type '{directive_type}'")
            continue

        required_keys = REQUIRED_KEYS[directive_type]
        missing = required_keys - structured_adjustment.keys()
        if missing:
            errors.append(f"{ctx}: structured_adjustment missing required keys {sorted(missing)}")
            continue

        if not _check_hours(structured_adjustment.get("hours"), errors, ctx):
            continue

        adj: dict[str, Any] = {"hours": list(structured_adjustment["hours"])}

        if directive_type == "solar_reduction":
            factor = structured_adjustment.get("factor")
            if not isinstance(factor, (int, float)) or isinstance(factor, bool):
                errors.append(f"{ctx}: factor must be a number")
                continue
            if not (0.0 <= float(factor) <= 1.0):
                errors.append(f"{ctx}: factor must be between 0 and 1 inclusive")
                continue
            adj["factor"] = float(factor)

        elif directive_type == "minimum_battery_reserve":
            min_energy = structured_adjustment.get("minimum_energy_kwh")
            if not isinstance(min_energy, (int, float)) or isinstance(min_energy, bool):
                errors.append(f"{ctx}: minimum_energy_kwh must be a number")
                continue
            min_energy = float(min_energy)
            if not (min_energy == min_energy and min_energy not in (float("inf"), float("-inf"))):
                errors.append(f"{ctx}: minimum_energy_kwh must be finite")
                continue
            if min_energy < 0 or min_energy > battery_capacity_kwh:
                errors.append(f"{ctx}: minimum_energy_kwh must be between 0 and battery capacity")
                continue
            adj["minimum_energy_kwh"] = min_energy

        elif directive_type == "max_grid_window":
            max_grid = structured_adjustment.get("max_grid_kwh")
            if not isinstance(max_grid, (int, float)) or isinstance(max_grid, bool):
                errors.append(f"{ctx}: max_grid_kwh must be a number")
                continue
            max_grid = float(max_grid)
            if not (max_grid == max_grid and max_grid not in (float("inf"), float("-inf"))):
                errors.append(f"{ctx}: max_grid_kwh must be finite")
                continue
            if max_grid < 0:
                errors.append(f"{ctx}: max_grid_kwh must be non-negative")
                continue
            adj["max_grid_kwh"] = max_grid

        # no_charge_window / no_discharge_window need only 'hours', already set.

        normalized.append(
            {
                "note_index": note_index,
                "applies": True,
                "directive_type": directive_type,
                "structured_adjustment": adj,
                "explanation": explanation,
            }
        )

    if len(seen_indices) != n:
        missing_notes = set(range(n)) - seen_indices
        if missing_notes:
            errors.append(f"missing directive_interpretation entries for note_index {sorted(missing_notes)}")

    is_valid = len(errors) == 0 and len(normalized) == n
    normalized.sort(key=lambda d: d["note_index"])
    return is_valid, errors, normalized
