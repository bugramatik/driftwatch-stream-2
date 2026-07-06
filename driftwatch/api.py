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

