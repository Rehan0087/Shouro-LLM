"""Defensive replay check (Problem Statement section 09 & 11).

The optimizer already enforces these as hard constraints, but we replay the
returned hourly_plan independently before responding, mirroring the judge's
own "Final Validator" stage in the architecture diagram. If this ever fails
it means the optimizer has a bug -- better to catch it here than ship an
invalid schedule.
"""

from __future__ import annotations

from typing import Any

TOL = 0.01


def replay_schedule(
    hours: list[dict[str, Any]],
    battery: dict[str, Any],
    directives: list[dict[str, Any]],
    hourly_plan: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    hours_by_h = {h["hour"]: h for h in hours}
    plan_by_h = {p["hour"]: p for p in hourly_plan}

    if sorted(plan_by_h.keys()) != list(range(24)):
        return ["hourly_plan must contain exactly one entry for each hour 0-23"]

    effective_solar = {h: float(hours_by_h[h]["solar_kwh"]) for h in range(24)}
    grid_cap: dict[int, float] = {}
    charge_allowed = {h: True for h in range(24)}
    discharge_allowed = {h: True for h in range(24)}
    min_energy = {h: float(battery["minimum_energy_kwh"]) for h in range(24)}

    for d in directives:
        if not d["applies"]:
            continue
        adj = d["structured_adjustment"]
        dtype = d["directive_type"]
        for h in adj["hours"]:
            if dtype == "solar_reduction":
                effective_solar[h] *= adj["factor"]
            elif dtype == "minimum_battery_reserve":
                min_energy[h] = max(min_energy[h], adj["minimum_energy_kwh"])
            elif dtype == "no_charge_window":
                charge_allowed[h] = False
            elif dtype == "no_discharge_window":
                discharge_allowed[h] = False
            elif dtype == "max_grid_window":
                grid_cap[h] = adj["max_grid_kwh"] if h not in grid_cap else min(grid_cap[h], adj["max_grid_kwh"])

    capacity = float(battery["capacity_kwh"])
    initial_energy = float(battery["initial_energy_kwh"])
    max_charge = float(battery["max_charge_kwh_per_hour"])
    max_discharge = float(battery["max_discharge_kwh_per_hour"])

    energy = initial_energy
    for h in range(24):
        p = plan_by_h[h]
        demand = float(hours_by_h[h]["demand_kwh"])
        grid = float(p["grid_kwh"])
        solar_used = float(p["solar_used_kwh"])
        action = p["battery_action"]
        battery_kwh = float(p["battery_kwh"])

        if grid < -TOL:
            errors.append(f"hour {h}: grid_kwh is negative")
        if solar_used < -TOL or solar_used > effective_solar[h] + TOL:
            errors.append(f"hour {h}: solar_used_kwh {solar_used} exceeds effective solar {effective_solar[h]}")
        if action not in ("charge", "discharge", "idle"):
            errors.append(f"hour {h}: invalid battery_action '{action}'")
        if action == "idle" and abs(battery_kwh) > TOL:
            errors.append(f"hour {h}: battery_kwh must be 0 when idle")
        if action == "charge" and battery_kwh > max_charge + TOL:
            errors.append(f"hour {h}: charge {battery_kwh} exceeds max_charge_kwh_per_hour {max_charge}")
        if action == "discharge" and battery_kwh > max_discharge + TOL:
            errors.append(f"hour {h}: discharge {battery_kwh} exceeds max_discharge_kwh_per_hour {max_discharge}")
        if action == "charge" and not charge_allowed[h]:
            errors.append(f"hour {h}: charging is not allowed (no_charge_window directive active)")
        if action == "discharge" and not discharge_allowed[h]:
            errors.append(f"hour {h}: discharging is not allowed (no_discharge_window directive active)")
        if h in grid_cap and grid > grid_cap[h] + TOL:
            errors.append(f"hour {h}: grid_kwh {grid} exceeds max_grid_window cap {grid_cap[h]}")

        charge = battery_kwh if action == "charge" else 0.0
        discharge = battery_kwh if action == "discharge" else 0.0
        expected_balance = grid + solar_used + discharge
        required_balance = demand + charge
        if abs(expected_balance - required_balance) > TOL:
            errors.append(
                f"hour {h}: energy balance violated ({expected_balance} != {required_balance})"
            )

        energy = energy + charge - discharge
        reported_after = float(p["battery_energy_after_kwh"])
        if abs(reported_after - energy) > TOL:
            errors.append(f"hour {h}: battery_energy_after_kwh mismatch (reported {reported_after}, expected {energy})")
        if energy < min_energy[h] - TOL:
            errors.append(f"hour {h}: battery energy {energy} below required minimum {min_energy[h]}")
        if energy > capacity + TOL:
            errors.append(f"hour {h}: battery energy {energy} exceeds capacity {capacity}")

    if abs(energy - initial_energy) > TOL:
        errors.append(f"end-of-day battery neutrality violated: final={energy}, initial={initial_energy}")

    return errors
