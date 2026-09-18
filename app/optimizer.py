"""Linear-programming energy optimizer.

Decision variables per hour h = 0..23:
    g[h] >= 0   grid_kwh
    s[h] >= 0   solar_used_kwh   (<= effective solar after solar_reduction)
    c[h] >= 0   battery charge amount this hour
    d[h] >= 0   battery discharge amount this hour

charge/discharge are kept as separate non-negative variables (rather than a
signed net + binary action flag) which keeps the whole problem a pure LP.
Since neither variable is penalized in the objective, an optimal solver never
benefits from running both in the same hour; even so we net them defensively
when building the response, which is always safe because
    net = c - d,  |net| <= max(c, d) <= the relevant rate limit.

Hard constraints:
    energy balance:      g + s + d - c = demand                (every hour)
    solar limit:          0 <= s <= effective_solar             (bounds)
    charge rate limit:    0 <= c <= max_charge (0 if no_charge_window)
    discharge rate limit: 0 <= d <= max_discharge (0 if no_discharge_window)
    grid cap:              g <= max_grid_kwh where max_grid_window applies
    battery bounds:        min_energy[h] <= E[h] <= capacity     (cumulative)
    end-of-day neutrality: E[23] = initial_energy_kwh

Objective: minimize sum(tariff[h] * g[h])
"""

from __future__ import annotations

from typing import Any

from scipy.optimize import linprog

ZERO_TOL = 1e-6


class InfeasibleScenarioError(Exception):
    pass


def _apply_directives(
    hours: list[dict[str, Any]],
    battery: dict[str, Any],
    directives: list[dict[str, Any]],
) -> tuple[list[float], list[float | None], list[bool], list[bool], list[float]]:
    n = 24
    effective_solar = [float(h["solar_kwh"]) for h in hours]
    grid_cap: list[float | None] = [None] * n
    charge_allowed = [True] * n
    discharge_allowed = [True] * n
    min_energy = [float(battery["minimum_energy_kwh"])] * n

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
                cap = adj["max_grid_kwh"]
                grid_cap[h] = cap if grid_cap[h] is None else min(grid_cap[h], cap)

    return effective_solar, grid_cap, charge_allowed, discharge_allowed, min_energy


def optimize_schedule(
    hours: list[dict[str, Any]],
    battery: dict[str, Any],
    directives: list[dict[str, Any]],
) -> dict[str, Any]:
    n = 24
    hours_sorted = sorted(hours, key=lambda h: h["hour"])
    demand = [float(h["demand_kwh"]) for h in hours_sorted]
    tariff = [float(h["tariff_bdt_per_kwh"]) for h in hours_sorted]

    effective_solar, grid_cap, charge_allowed, discharge_allowed, min_energy = _apply_directives(
        hours_sorted, battery, directives
    )

    capacity = float(battery["capacity_kwh"])
    initial_energy = float(battery["initial_energy_kwh"])
    max_charge = float(battery["max_charge_kwh_per_hour"])
    max_discharge = float(battery["max_discharge_kwh_per_hour"])

    # Variable layout: [g_0..23, s_0..23, c_0..23, d_0..23]
    num_vars = 4 * n

    def idx_g(h: int) -> int:
        return h

    def idx_s(h: int) -> int:
        return n + h

    def idx_c(h: int) -> int:
        return 2 * n + h

    def idx_d(h: int) -> int:
        return 3 * n + h

    c_obj = [0.0] * num_vars
    for h in range(n):
        c_obj[idx_g(h)] = tariff[h]

    bounds: list[tuple[float, float | None]] = [(0.0, None)] * num_vars
    for h in range(n):
        bounds[idx_g(h)] = (0.0, grid_cap[h])
        bounds[idx_s(h)] = (0.0, max(effective_solar[h], 0.0))
        bounds[idx_c(h)] = (0.0, max_charge if charge_allowed[h] else 0.0)
        bounds[idx_d(h)] = (0.0, max_discharge if discharge_allowed[h] else 0.0)

    A_eq: list[list[float]] = []
    b_eq: list[float] = []

    # Energy balance per hour: g + s + d - c = demand
    for h in range(n):
        row = [0.0] * num_vars
        row[idx_g(h)] = 1.0
        row[idx_s(h)] = 1.0
        row[idx_d(h)] = 1.0
        row[idx_c(h)] = -1.0
        A_eq.append(row)
        b_eq.append(demand[h])

    # End-of-day neutrality: sum(c) - sum(d) = 0
    row = [0.0] * num_vars
    for h in range(n):
        row[idx_c(h)] = 1.0
        row[idx_d(h)] = -1.0
    A_eq.append(row)
    b_eq.append(0.0)

    A_ub: list[list[float]] = []
    b_ub: list[float] = []

    # Cumulative battery energy bounds: min_energy[h] <= initial + sum_{k<=h}(c-d) <= capacity
    for h in range(n):
        row = [0.0] * num_vars
        for k in range(h + 1):
            row[idx_c(k)] = 1.0
            row[idx_d(k)] = -1.0
        # upper: sum <= capacity - initial
        A_ub.append(row)
        b_ub.append(capacity - initial_energy)
        # lower: -sum <= initial - min_energy[h]  (i.e. sum >= min_energy[h] - initial)
        neg_row = [-v for v in row]
        A_ub.append(neg_row)
        b_ub.append(initial_energy - min_energy[h])

    result = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise InfeasibleScenarioError(
            f"optimizer could not find a feasible schedule: {result.message}"
        )

    x = result.x
    hourly_plan = []
    energy = initial_energy
    for h in range(n):
        g = max(x[idx_g(h)], 0.0)
        s = max(x[idx_s(h)], 0.0)
        c = x[idx_c(h)]
        d = x[idx_d(h)]
        net = c - d
        if net > ZERO_TOL:
            action = "charge"
            battery_kwh = net
        elif net < -ZERO_TOL:
            action = "discharge"
            battery_kwh = -net
        else:
            action = "idle"
            battery_kwh = 0.0
        energy = energy + net
        hourly_plan.append(
            {
                "hour": h,
                "grid_kwh": round(g, 6),
                "solar_used_kwh": round(s, 6),
                "battery_action": action,
                "battery_kwh": round(battery_kwh, 6),
                "battery_energy_after_kwh": round(energy, 6),
            }
        )

    total_grid_kwh = sum(entry["grid_kwh"] for entry in hourly_plan)
    total_cost_bdt = sum(entry["grid_kwh"] * tariff[h] for h, entry in enumerate(hourly_plan))
    peak_grid_kwh = max(entry["grid_kwh"] for entry in hourly_plan)

    return {
        "hourly_plan": hourly_plan,
        "total_grid_kwh": round(total_grid_kwh, 6),
        "total_cost_bdt": round(total_cost_bdt, 6),
        "peak_grid_kwh": round(peak_grid_kwh, 6),
    }
