"""FastAPI service exposing GET /health and POST /optimize-energy
(Problem Statement section 06)."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.pipeline import PipelineError, run_pipeline
from app.schemas import HealthResponse, OptimizeEnergyRequest, OptimizeEnergyResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise Energy Optimization API")


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/optimize-energy")
async def optimize_energy(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "malformed JSON body"})

    try:
        parsed = OptimizeEnergyRequest.model_validate(body)
    except ValidationError as exc:
        return JSONResponse(status_code=400, content={"error": "invalid request schema", "details": exc.errors()})

    try:
        result = run_pipeline(parsed.model_dump())
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
