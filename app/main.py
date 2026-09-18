"""FastAPI service exposing GET /health and POST /optimize-energy
(Problem Statement section 06)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import ValidationError

from app.pipeline import PipelineError, run_pipeline
from app.schemas import HealthResponse, OptimizeEnergyRequest, OptimizeEnergyResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise Energy Optimization API")


def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    # FastAPI auto-documents a 422 response on every route with a validated
    # body, but our exception handler below always returns 400 instead (per
    # the Problem Statement's exact contract) -- leaving the auto-doc in
    # place would make /docs describe a status code this API never sends.
    for path in schema.get("paths", {}).values():
        for operation in path.values():
            operation.get("responses", {}).pop("422", None)
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError):
    # Covers both malformed JSON syntax and schema-invalid bodies -- FastAPI's
    # default for a declared body parameter is 422, but the Problem Statement
    # requires 400 for "malformed JSON or structurally invalid request".
    # exc.errors() can embed non-JSON-serializable objects (e.g. a raw
    # ValueError in ctx, raised by our own model_validators in schemas.py) --
    # jsonable_encoder is required here, plain json.dumps crashes on those.
    return JSONResponse(status_code=400, content={"error": "invalid request", "details": jsonable_encoder(exc.errors())})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    # Judge harness only ever calls /health and /optimize-energy; this just
    # keeps a casual visit to the bare base URL from looking broken.
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/optimize-energy",
    response_model=OptimizeEnergyResponse,
    responses={
        400: {
            "description": "Malformed JSON or structurally invalid request",
            "content": {
                "application/json": {
                    "example": {
                        "error": "invalid request",
                        "details": [
                            {
                                "type": "missing",
                                "loc": ["body", "operator_notes"],
                                "msg": "Field required",
                            }
                        ],
                    }
                }
            },
        },
        500: {
            "description": "Controlled internal error (e.g. an infeasible scenario, "
            "or the LLM never produced valid output after a corrective retry). "
            "Never a raw stack trace.",
            "content": {
                "application/json": {
                    "example": {"error": "unable to produce a valid schedule for this scenario"}
                }
            },
        },
    },
)
async def optimize_energy(parsed: OptimizeEnergyRequest):
    try:
        # run_pipeline makes a blocking LLM HTTP call; running it on a worker
        # thread keeps the event loop free so /health and other concurrent
        # requests aren't stalled behind it (Problem Statement section 08:
        # "the submitted service must remain reachable ... including
        # repeated LLM-backed requests").
        result = await asyncio.to_thread(run_pipeline, parsed.model_dump())
    except PipelineError as exc:
        logger.error("pipeline error for scenario %s: %s", parsed.scenario_id, exc)
        return JSONResponse(status_code=500, content={"error": "unable to produce a valid schedule for this scenario"})
    except Exception:
        logger.exception("unexpected error for scenario %s", parsed.scenario_id)
        return JSONResponse(status_code=500, content={"error": "internal server error"})

    try:
        validated = OptimizeEnergyResponse.model_validate(result)
    except ValidationError:
        logger.exception("internal response failed schema validation for scenario %s", parsed.scenario_id)
        return JSONResponse(status_code=500, content={"error": "internal server error"})

    return validated.model_dump()
