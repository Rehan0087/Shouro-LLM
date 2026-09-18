# GridWise — LLM-Assisted Campus Energy Optimization

BUP CSE Fest 2026 Hackathon · Online Preliminary · Team submission

An HTTP API that interprets natural-language campus operator notes with an
LLM, deterministically validates the extracted directives, then solves a
linear program to produce a minimum-cost 24-hour grid/solar/battery schedule.

## Architecture

```
Energy Data + Operator Notes
        │
        ▼
┌───────────────────┐   forced tool-call JSON, one entry per note
│   LLM Interpreter  │──────────────────────────────────────────┐
└───────────────────┘                                            │
        │ untrusted structured output                            │
        ▼                                                        │
┌───────────────────┐   invalid? → 1 corrective retry with        │
│ Guardrail Validator│     the exact errors fed back to the LLM ──┘
└───────────────────┘
        │ validated directives
        ▼
┌───────────────────┐   linear program (scipy HiGHS): minimize
│    Math Optimizer  │   Σ grid_kwh[h] · tariff[h] subject to
└───────────────────┘   energy balance, battery, and directive constraints
        │ hourly_plan
        ▼
┌───────────────────┐   independent replay: recomputes energy balance,
│  Final Validator    │   battery bounds, and every active directive
└───────────────────┘
        │
        ▼
   API Response (directive_interpretation + hourly_plan + totals)
```

- `app/llm_interpreter.py` — calls the LLM with a forced structured tool
  call so every response is well-formed JSON; on a guardrail failure it
  retries once with the specific errors.
- `app/guardrails.py` — deterministic validation of every field the Problem
  Statement requires (allowed types, note mapping, hour ordering, numeric
  ranges, `applies`/`no_op` semantics). Nothing from the LLM reaches the
  optimizer unvalidated.
- `app/optimizer.py` — builds and solves a linear program (grid/solar/charge/
  discharge per hour) with `scipy.optimize.linprog` (HiGHS solver). All six
  directive types are encoded as hard constraints.
- `app/replay_validator.py` — independently replays the returned schedule
  hour-by-hour to catch any optimizer bug before it reaches the response.
- `app/pipeline.py` / `app/main.py` — wires the stages together behind
  `GET /health` and `POST /optimize-energy`.

## LLM / model used

- **Provider:** configurable via `LLM_PROVIDER` — `openai` (default) or `anthropic`.
- **OpenAI (default):** `OPENAI_MODEL` (default `gpt-4o-mini`), called with
  [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
  (`response_format: {"type": "json_schema", "json_schema": {..., "strict": true}}`).
  OpenAI enforces the schema during generation, so the response is guaranteed
  to be well-formed JSON matching our shape before it ever reaches guardrails.
- **Anthropic (alternate):** `ANTHROPIC_MODEL` (default `claude-haiku-4-5-20251001`),
  called with a forced tool call (`tool_choice: {"type": "tool", "name": "submit_interpretation"}`).
- Either way, the LLM is the sole source of the `directive_interpretation` —
  it is not used only for `plan_summary` or cosmetic text — and its output is
  still fully re-validated by `app/guardrails.py` regardless of provider-level
  schema enforcement.

## Setup (local quickstart)

```bash
# 1. Clone / enter the repo
cd "BUP HAack"

# 2. Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# edit .env and set LLM_PROVIDER + the matching API key
#   LLM_PROVIDER=anthropic
#   ANTHROPIC_API_KEY=sk-ant-...
set -a; source .env; set +a

# 5. Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Verify it's alive

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Run a public sample case

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @<(python3 -c "import json; print(json.dumps(json.load(open('sample_cases/public_sample_cases.json'))['cases'][0]['input']))")
```

## Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `LLM_PROVIDER` | no (default `anthropic`) | `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | if provider is `anthropic` | Anthropic API key |
| `ANTHROPIC_MODEL` | no | defaults to `claude-haiku-4-5-20251001` |
| `OPENAI_API_KEY` | if provider is `openai` | OpenAI API key |
| `OPENAI_MODEL` | no | defaults to `gpt-4o-mini` |
| `PORT` | no | port for local `uvicorn` runs (default 8000) |

No secret values are committed anywhere in this repository; `.env` is
git-ignored and `.env.example` contains placeholders only.

## Optimizer / solver

Energy scheduling is modeled as a linear program over 96 variables
(`grid`, `solar_used`, `battery_charge`, `battery_discharge` × 24 hours) and
solved with `scipy.optimize.linprog` (`method="highs"`). All GridWise rules
(energy balance, solar/charge/discharge bounds, battery capacity, end-of-day
neutrality) and all six directive types are encoded as linear constraints, so
the solution returned is a global optimum for the given (directive-adjusted)
problem.

## Testing

```bash
source .venv/bin/activate

# Deterministic math pipeline against all 10 public sample cases (no API key needed)
pytest tests/test_optimizer_against_samples.py -v

# Guardrail rejection behavior on malformed/unsupported output
pytest tests/test_guardrails_reject_bad_output.py -v

# Full HTTP API, LLM mocked with ground-truth directives (no API key needed)
pytest tests/test_api_end_to_end.py -v

# Real LLM interpretation vs. ground truth (requires an API key; auto-skips otherwise)
pytest tests/test_llm_interpretation.py -v

# Everything
pytest -v
```

## Docker

```bash
docker build -t gridwise-llm .
docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=anthropic \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  gridwise-llm

curl http://localhost:8000/health
```

The image exposes port 8000, binds to `0.0.0.0`, and contains no baked-in
credentials — all keys are supplied at `docker run` time via `-e`.

## Known limitations

- Requires a reachable LLM provider at request time; the team is responsible
  for API key validity, quota, and rate limits during judging (no local/
  offline model is bundled).
- One corrective retry is attempted on a guardrail failure; if the model
  still returns invalid structured output after that, the request fails
  safely with HTTP 500 rather than guessing a directive.
- The optimizer assumes organizer-valid scenarios are feasible per the
  Problem Statement; a scenario with genuinely contradictory hard directives
  will raise a controlled error instead of returning an invalid schedule.

## Dependencies

- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) — HTTP service
- [Pydantic](https://docs.pydantic.dev/) — request/response schema validation
- [SciPy](https://scipy.org/) (`linprog`, HiGHS) — LP energy optimizer
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) / [OpenAI Python SDK](https://github.com/openai/openai-python) — LLM interpretation
- [pytest](https://pytest.org/) — test suite
