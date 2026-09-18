"""LLM-backed operator-note interpreter.

The language model is the ONLY thing that decides what each operator note
means. Its raw output is treated as untrusted structured data and is always
run through guardrails.validate_directive_interpretation before use (Problem
Statement section 08). If the first attempt fails guardrails, we send the
model a corrective follow-up naming the exact errors and ask it to resubmit
once before giving up.

Supports two providers, selected via LLM_PROVIDER env var:
  - "openai" (default): uses Structured Outputs (response_format
    json_schema, strict=True) -- OpenAI enforces the schema at generation
    time, so malformed JSON shape is not even possible.
  - "anthropic": uses tool-use with a forced tool_choice.

Both providers are forced into a fixed JSON schema at the API level, which is
the strongest guardrail we can put in front of the guardrail module itself.
"""

from __future__ import annotations

import json
import os
from typing import Any

from app.guardrails import REQUIRED_KEYS

SYSTEM_PROMPT = """You are the operator-note interpreter for GridWise, a smart-campus energy \
scheduling system. You will be given a battery's capacity and a numbered list of 1-3 short \
operator notes describing temporary conditions for the next 24-hour period. Convert EVERY note \
into exactly one structured directive from the fixed set below. Never invent a directive type \
outside this set, and never invent or alter demand, tariff, or battery numbers that were not \
stated in the note.

SUPPORTED DIRECTIVE TYPES
1. solar_reduction - {"hours": [int...], "factor": number}
   factor is the USABLE FRACTION THAT REMAINS, not the reduction amount.
   Example: "output drops to 20%" -> factor 0.2. "an 80% reduction" -> factor 0.2 (100%-80%=20% remains).
2. minimum_battery_reserve - {"hours": [int...], "minimum_energy_kwh": number}
   If the note states a percentage of capacity, convert it to an absolute kWh value using the
   battery capacity given to you. Example: 50% of a 200 kWh battery -> minimum_energy_kwh 100.
3. no_charge_window - {"hours": [int...]}
4. no_discharge_window - {"hours": [int...]}
5. max_grid_window - {"hours": [int...], "max_grid_kwh": number}
6. no_op - structured_adjustment must be null. Use this whenever a note does not affect today's
   24-hour energy schedule (unrelated announcements, other departments, future dates, non-energy
   topics, or anything not about today's solar/battery/grid operation).

TIME WINDOW RULE
Hours are integers 0-23. A stated clock range is start-inclusive and end-exclusive.
"1 PM to 3 PM" -> hours [13, 14] (NOT 15). "6 PM until 9 PM" -> hours [18, 19, 20].
Convert 12-hour clock references exactly: 12 AM = 0, 12 PM = 12, 1 PM = 13, etc. 24-hour clock
references (e.g. "13:00") map directly to that hour.

INTERPRETATION RULES
- Every note produces exactly one entry, using its 0-based position in the note list as note_index.
- Only mark applies = true for one of the 5 non-no_op directive types; use applies = false only
  for no_op.
- Notes may be phrased in many different ways (percentages, fractions, different clock formats,
  indirect language). Interpret the underlying operational meaning, not literal keywords -- the
  same rule can be worded very differently across scenarios.
- Do not modify demand, tariff, or battery configuration values; only produce the directive.
- hours arrays must contain unique integers 0-23 in strictly ascending order.
- structured_adjustment always carries all four fields (hours, factor, minimum_energy_kwh,
  max_grid_kwh). Set the fields that do not apply to the chosen directive_type to null, and for
  no_op set structured_adjustment itself to null. Example: no_charge_window sets hours to the
  real list and leaves factor, minimum_energy_kwh, max_grid_kwh as null.

Respond ONLY with the structured directive_interpretation, one entry per note, in note_index order."""

TOOL_SCHEMA = {
    "name": "submit_interpretation",
    "description": "Submit the structured directive interpretation for every operator note.",
    "input_schema": {
        "type": "object",
        "properties": {
            "directive_interpretation": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "note_index": {"type": "integer"},
                        "applies": {"type": "boolean"},
                        "directive_type": {
                            "type": "string",
                            "enum": [
                                "solar_reduction",
                                "minimum_battery_reserve",
                                "no_charge_window",
                                "no_discharge_window",
                                "max_grid_window",
                                "no_op",
                            ],
                        },
                        "structured_adjustment": {
                            "type": ["object", "null"],
                            "properties": {
                                "hours": {"type": "array", "items": {"type": "integer"}},
                                "factor": {"type": "number"},
                                "minimum_energy_kwh": {"type": "number"},
                                "max_grid_kwh": {"type": "number"},
                            },
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "note_index",
                        "applies",
                        "directive_type",
                        "structured_adjustment",
                        "explanation",
                    ],
                },
            }
        },
        "required": ["directive_interpretation"],
    },
}


# OpenAI Structured Outputs (strict=True) requires every property to be listed
# in "required" and forbids additionalProperties, so structured_adjustment is
# expressed as a flat object with every possible field present but nullable.
# Guardrails only reads the fields relevant to the chosen directive_type, so
# the irrelevant nulled-out fields are simply ignored downstream.
OPENAI_JSON_SCHEMA = {
    "name": "directive_interpretation_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "directive_interpretation": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "note_index": {"type": "integer"},
                        "applies": {"type": "boolean"},
                        "directive_type": {
                            "type": "string",
                            "enum": [
                                "solar_reduction",
                                "minimum_battery_reserve",
                                "no_charge_window",
                                "no_discharge_window",
                                "max_grid_window",
                                "no_op",
                            ],
                        },
                        "structured_adjustment": {
                            "type": ["object", "null"],
                            "properties": {
                                "hours": {"type": ["array", "null"], "items": {"type": "integer"}},
                                "factor": {"type": ["number", "null"]},
                                "minimum_energy_kwh": {"type": ["number", "null"]},
                                "max_grid_kwh": {"type": ["number", "null"]},
                            },
                            "required": ["hours", "factor", "minimum_energy_kwh", "max_grid_kwh"],
                            "additionalProperties": False,
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "note_index",
                        "applies",
                        "directive_type",
                        "structured_adjustment",
                        "explanation",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["directive_interpretation"],
        "additionalProperties": False,
    },
}


class LLMInterpreterError(Exception):
    """Raised when the LLM cannot be reached or never produces valid output."""


def _build_user_message(operator_notes: list[str], battery_capacity_kwh: float) -> str:
    notes_block = "\n".join(f"{i}: {note}" for i, note in enumerate(operator_notes))
    return (
        f"Battery capacity_kwh: {battery_capacity_kwh}\n\n"
        f"Operator notes:\n{notes_block}\n\n"
        "Call submit_interpretation with exactly one entry per note, in note_index order."
    )


def _call_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_interpretation"},
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_interpretation":
            return block.input.get("directive_interpretation", [])

    raise LLMInterpreterError("model did not call submit_interpretation")


def _call_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    openai_messages.extend(messages)

    response = client.chat.completions.create(
        model=model,
        messages=openai_messages,
        response_format={"type": "json_schema", "json_schema": OPENAI_JSON_SCHEMA},
    )

    message = response.choices[0].message
    if message.refusal:
        raise LLMInterpreterError(f"model refused: {message.refusal}")
    if not message.content:
        raise LLMInterpreterError("model returned empty content")

    args = json.loads(message.content)
    raw_entries = args.get("directive_interpretation", [])

    # Flatten structured_adjustment (all-fields-present, strict-mode shape)
    # back down to only the keys relevant to each entry's directive_type.
    normalized_entries = []
    for entry in raw_entries:
        adj = entry.get("structured_adjustment")
        directive_type = entry.get("directive_type")
        if directive_type == "no_op" or adj is None:
            entry["structured_adjustment"] = None
        else:
            relevant_keys = {"hours"} | (REQUIRED_KEYS.get(directive_type, set()) - {"hours"})
            entry["structured_adjustment"] = {
                k: v for k, v in adj.items() if k in relevant_keys and v is not None
            }
        normalized_entries.append(entry)

    return normalized_entries


def _call_llm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    try:
        if provider == "openai":
            return _call_openai(messages)
        elif provider == "anthropic":
            return _call_anthropic(messages)
        else:
            raise LLMInterpreterError(f"unsupported LLM_PROVIDER '{provider}'")
    except LLMInterpreterError:
        raise
    except Exception as exc:  # network/auth/provider errors
        raise LLMInterpreterError(f"LLM call failed: {exc}") from exc


def interpret_notes(
    operator_notes: list[str],
    battery_capacity_kwh: float,
    guardrail_check,
) -> list[dict[str, Any]]:
    """Interpret operator notes via the LLM, validating with `guardrail_check`.

    guardrail_check(directives) -> (is_valid, errors, normalized). On first
    failure, one corrective retry is attempted with the specific errors fed
    back to the model. Raises LLMInterpreterError if still invalid.
    """
    user_message = _build_user_message(operator_notes, battery_capacity_kwh)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    raw = _call_llm(messages)
    is_valid, errors, normalized = guardrail_check(raw)
    if is_valid:
        return normalized

    # One corrective retry with the exact guardrail errors.
    correction = (
        "Your previous submission failed validation with these errors:\n"
        + "\n".join(f"- {e}" for e in errors)
        + "\n\nResubmit a corrected, complete directive_interpretation covering every note."
    )
    messages.append({"role": "assistant", "content": json.dumps({"directive_interpretation": raw})})
    messages.append({"role": "user", "content": correction})

    raw2 = _call_llm(messages)
    is_valid2, errors2, normalized2 = guardrail_check(raw2)
    if is_valid2:
        return normalized2

    raise LLMInterpreterError(
        "LLM output failed guardrail validation after retry: " + "; ".join(errors2)
    )
