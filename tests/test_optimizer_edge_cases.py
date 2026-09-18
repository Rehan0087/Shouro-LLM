"""Boundary-condition tests for the optimizer, independent of the LLM. These
probe values a hidden case is likely to specifically target: zero/extreme
directive values, a battery starting full, and a genuinely infeasible
scenario that must fail safely rather than return an invalid schedule.
"""

from __future__ import annotations

from app import optimizer, replay_validator
from app.optimizer import InfeasibleScenarioError

import pytest


def make_hours(demand=100.0, solar=0.0, tariff=10.0, n=24):
    return [
        {"hour": h, "demand_kwh": demand, "solar_kwh": solar, "tariff_bdt_per_kwh": tariff}
        for h in range(n)
    ]


def make_battery(capacity=200.0, initial=100.0, minimum=20.0, max_charge=50.0, max_discharge=50.0):
    return {
        "capacity_kwh": capacity,
        "initial_energy_kwh": initial,
        "minimum_energy_kwh": minimum,
        "max_charge_kwh_per_hour": max_charge,
        "max_discharge_kwh_per_hour": max_discharge,
    }


def test_max_grid_zero_forces_battery_and_solar():
    hours = make_hours(demand=50.0, solar=30.0)
    battery = make_battery(initial=100.0)
    directives = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": [10], "max_grid_kwh": 0.0},
            "explanation": "test",
        }
    ]
    result = optimizer.optimize_schedule(hours, battery, directives)
    hour_10 = next(p for p in result["hourly_plan"] if p["hour"] == 10)
    assert hour_10["grid_kwh"] == 0.0
    errors = replay_validator.replay_schedule(hours, battery, directives, result["hourly_plan"])
    assert not errors, errors


def test_solar_reduction_factor_zero():
    hours = make_hours(demand=50.0, solar=40.0)
    battery = make_battery()
    directives = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [5], "factor": 0.0},
            "explanation": "test",
        }
    ]
    result = optimizer.optimize_schedule(hours, battery, directives)
    hour_5 = next(p for p in result["hourly_plan"] if p["hour"] == 5)
    assert hour_5["solar_used_kwh"] == 0.0
    errors = replay_validator.replay_schedule(hours, battery, directives, result["hourly_plan"])
    assert not errors, errors


def test_solar_reduction_factor_one_is_a_no_op_numerically():
    hours = make_hours(demand=50.0, solar=40.0)
    battery = make_battery()
    no_directive_result = optimizer.optimize_schedule(hours, battery, [])
    full_factor_directive = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": list(range(24)), "factor": 1.0},
            "explanation": "test",
        }
    ]
    with_directive_result = optimizer.optimize_schedule(hours, battery, full_factor_directive)
    assert abs(no_directive_result["total_cost_bdt"] - with_directive_result["total_cost_bdt"]) < 0.01


def test_minimum_reserve_equal_to_capacity_forces_full_battery():
    hours = make_hours(demand=50.0, solar=0.0)
    battery = make_battery(capacity=200.0, initial=100.0, minimum=20.0, max_charge=200.0, max_discharge=200.0)
    directives = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [12], "minimum_energy_kwh": 200.0},
            "explanation": "test",
        }
    ]
    result = optimizer.optimize_schedule(hours, battery, directives)
    hour_12 = next(p for p in result["hourly_plan"] if p["hour"] == 12)
    assert abs(hour_12["battery_energy_after_kwh"] - 200.0) < 0.01
    errors = replay_validator.replay_schedule(hours, battery, directives, result["hourly_plan"])
    assert not errors, errors


def test_battery_starts_full_cannot_overcharge():
    hours = make_hours(demand=10.0, solar=100.0)  # excess solar every hour
    battery = make_battery(capacity=150.0, initial=150.0, minimum=10.0)
    result = optimizer.optimize_schedule(hours, battery, [])
    for p in result["hourly_plan"]:
        assert p["battery_energy_after_kwh"] <= 150.0 + 0.01
    errors = replay_validator.replay_schedule(hours, battery, [], result["hourly_plan"])
    assert not errors, errors


def test_stacked_no_charge_and_reserve_same_hour():
    # An hour that's both a no-charge window AND has a raised minimum reserve
    # must still be satisfiable by having charged the energy in BEFORE that hour.
    hours = make_hours(demand=50.0, solar=0.0)
    battery = make_battery(capacity=200.0, initial=100.0, minimum=20.0, max_charge=100.0, max_discharge=100.0)
    directives = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [10]},
            "explanation": "test",
        },
        {
            "note_index": 1,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [10], "minimum_energy_kwh": 150.0},
            "explanation": "test",
        },
    ]
    result = optimizer.optimize_schedule(hours, battery, directives)
    hour_10 = next(p for p in result["hourly_plan"] if p["hour"] == 10)
    assert hour_10["battery_action"] != "charge"
    assert hour_10["battery_energy_after_kwh"] >= 150.0 - 0.01
    errors = replay_validator.replay_schedule(hours, battery, directives, result["hourly_plan"])
    assert not errors, errors


def test_genuinely_infeasible_scenario_raises_controlled_error():
    # Demand far exceeds grid+solar+battery capability with a zero grid cap
    # and no solar: impossible to satisfy energy balance.
    hours = make_hours(demand=500.0, solar=0.0)
    battery = make_battery(capacity=50.0, initial=25.0, minimum=0.0, max_charge=10.0, max_discharge=10.0)
    directives = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": list(range(24)), "max_grid_kwh": 0.0},
            "explanation": "test",
        }
    ]
    with pytest.raises(InfeasibleScenarioError):
        optimizer.optimize_schedule(hours, battery, directives)
