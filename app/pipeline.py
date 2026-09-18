"""End-to-end pipeline: LLM Interpreter -> Guardrail Validator -> Math
Optimizer -> Final Validator -> response assembly (Problem Statement section 03).
"""

from __future__ import annotations

from typing import Any

from app import guardrails, llm_interpreter, optimizer, replay_validator


class PipelineError(Exception):
    """Raised for any controlled failure that should become an HTTP 500."""


def _guardrail_check(operator_notes: list[str], battery_capacity_kwh: float):
    def check(raw_directives: list[dict[str, Any]]):
        return guardrails.validate_directive_interpretation(
            raw_directives, operator_notes, battery_capacity_kwh
        )

    return check


def run_pipeline(request_data: dict[str, Any]) -> dict[str, Any]:
    scenario_id = request_data["scenario_id"]
    operator_notes = request_data["operator_notes"]
    hours = request_data["hours"]
    battery = request_data["battery"]

    try:
        directives = llm_interpreter.interpret_notes(
            operator_notes,
            battery["capacity_kwh"],
            _guardrail_check(operator_notes, battery["capacity_kwh"]),
        )
    except llm_interpreter.LLMInterpreterError as exc:
        raise PipelineError(f"operator note interpretation failed: {exc}") from exc

    try:
        opt_result = optimizer.optimize_schedule(hours, battery, directives)
    except optimizer.InfeasibleScenarioError as exc:
        raise PipelineError(str(exc)) from exc

    replay_errors = replay_validator.replay_schedule(hours, battery, directives, opt_result["hourly_plan"])
    if replay_errors:
        raise PipelineError("final schedule replay failed: " + "; ".join(replay_errors))

    plan_summary = _build_plan_summary(directives, opt_result)

    return {
        "scenario_id": scenario_id,
        "directive_interpretation": directives,
        "hourly_plan": opt_result["hourly_plan"],
        "total_grid_kwh": opt_result["total_grid_kwh"],
        "total_cost_bdt": opt_result["total_cost_bdt"],
        "peak_grid_kwh": opt_result["peak_grid_kwh"],
        "plan_summary": plan_summary,
    }


def _build_plan_summary(directives: list[dict[str, Any]], opt_result: dict[str, Any]) -> str:
    applied = [d["directive_type"] for d in directives if d["applies"]]
    if applied:
        directive_text = "applies " + ", ".join(sorted(set(applied)))
    else:
        directive_text = "applies no operator directives (all notes were no_op)"
    return (
        f"Optimized 24-hour schedule {directive_text}; total grid cost "
        f"{opt_result['total_cost_bdt']:.2f} BDT with peak hourly draw "
        f"{opt_result['peak_grid_kwh']:.2f} kWh, while restoring the battery to its "
        f"starting energy level by hour 23."
    )
