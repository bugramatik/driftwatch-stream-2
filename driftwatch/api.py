"""FastAPI app serving drift status, alert history and reference profiles."""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any, Deque

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from driftwatch import __version__
from driftwatch.config import Settings, load_settings
from driftwatch.drift import evaluate_model
from driftwatch.profiles import ProfileStore

log = logging.getLogger("driftwatch.api")

CONFIG_PATH = os.environ.get("DRIFTWATCH_CONFIG", "config/driftwatch.yaml")
settings: Settings = load_settings(CONFIG_PATH)
store = ProfileStore(settings.reference_dir)

app = FastAPI(title="Driftwatch Stream", version=__version__)

# Alerts published by the consumer are mirrored here so the dashboard can list them
# without a second Kafka client. The consumer appends through record_alert().
_alerts: Deque[dict[str, Any]] = deque(maxlen=500)


class ReferenceRequest(BaseModel):
    """A reference profile posted by the Spark job or by hand."""

    version: str = Field(min_length=1)
    features: dict[str, list[float]] = Field(min_length=1)


class AlertIn(BaseModel):
    """One alert as the consumer publishes it."""

    model: str
    version: str = "unknown"
    feature: str
    psi: float
    ks: float
    samples: int = 0
    raised_at: int = 0


def record_alert(alert: dict[str, Any]) -> None:
    """Keep the newest alerts in memory for the dashboard."""
    _alerts.appendleft(alert)


@app.get("/health")

def health() -> dict[str, Any]:
    """Liveness plus the number of models with a stored profile."""
    return {"status": "ok", "version": __version__, "models": len(store.models())}


@app.get("/api/models")

def list_models() -> list[dict[str, Any]]:
    """Every model that has at least one reference profile."""
    out: list[dict[str, Any]] = []
    for model in store.models():
        versions = store.versions(model)
        out.append(
            {
                "model": model,
                "versions": versions,
                "latest": versions[0] if versions else None,
            }
        )
    return out


@app.get("/api/models/{model}/status")

def model_status(model: str, samples: int = Query(default=0, ge=0)) -> dict[str, Any]:
    """Compare the latest reference profile with a current sample when one is posted.

    Without a current sample the endpoint reports the stored profile and the
    thresholds in force, which is what the dashboard shows before the first window.
    """
    try:
        version, reference = store.latest(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    payload: dict[str, Any] = {
        "model": model,
        "version": version,
        "features": sorted(reference),
        "psi_threshold": settings.psi_threshold,
        "ks_threshold": settings.ks_threshold,
        "window_size": settings.window_size,
    }
    if samples:
        payload["reference_samples"] = {name: len(values) for name, values in reference.items()}
    return payload


@app.get("/api/alerts")

def list_alerts(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
    """The newest alerts first, at most ``limit`` of them."""
    return list(_alerts)[:limit]


@app.post("/api/alerts")

def post_alert(alert: AlertIn) -> dict[str, Any]:
    """Accept an alert from the consumer so the dashboard can show it."""
    record_alert(alert.model_dump())
    return {"accepted": True, "pending": len(_alerts)}


@app.post("/api/models/{model}/reference")

def post_reference(model: str, body: ReferenceRequest) -> dict[str, Any]:
    """Store a reference profile for a model version."""
    try:
        path: Path = store.save(model, body.version, body.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"model": model, "version": body.version, "path": str(path)}


@app.get("/api/models/{model}/evaluate")

def evaluate(model: str, current: str = Query(description="path to a JSON file of current values")) -> dict[str, Any]:
    """Evaluate a JSON file of current feature values against the latest profile."""
    try:
        version, reference = store.latest(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    source = Path(current)
    if not source.is_file():
        raise HTTPException(status_code=400, detail=f"no such file: {current}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    import numpy as np

    current_values = {
        str(name): np.asarray(values, dtype=float)
        for name, values in (raw.get("features") or raw).items()
        if isinstance(values, list) and values
    }
    verdict = evaluate_model(
        model,
        version,
        reference,
        current_values,
        settings.psi_threshold,
        settings.ks_threshold,
    )
    return verdict.model_dump()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))


if __name__ == "__main__":
    main()
