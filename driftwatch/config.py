"""Typed settings loaded from the YAML configuration file."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    """Everything the consumer and the API need to run."""

    kafka_bootstrap: str = "localhost:9092"
    input_topic: str = "predictions"
    alert_topic: str = "drift.alerts"
    group_id: str = "driftwatch"
    window_size: int = Field(default=5000, ge=10)
    min_samples: int = Field(default=500, ge=10)
    psi_threshold: float = Field(default=0.2, gt=0.0)
    ks_threshold: float = Field(default=0.1, gt=0.0)
    cooldown_seconds: int = Field(default=900, ge=0)
    reference_dir: str = "./profiles"

    @field_validator("window_size", "min_samples")
    @classmethod
    def _window_fits(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("window values must be positive")
        return value

    def model_post_init(self, __context: object) -> None:
        if self.min_samples > self.window_size:
            raise ValueError("min_samples cannot exceed window_size")


def load_settings(path: str) -> Settings:
    """Read the YAML file at ``path`` and flatten it into a Settings object.

    The file groups keys under ``kafka``, ``drift`` and ``store``; a missing
    file falls back to the defaults so the API can start on its own.
    """
    file = Path(path)
    if not file.is_file():
        return Settings()

    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must hold a mapping at the top level")

    kafka = raw.get("kafka") or {}
    drift = raw.get("drift") or {}
    store = raw.get("store") or {}

    return Settings(
        kafka_bootstrap=str(kafka.get("bootstrap", Settings.model_fields["kafka_bootstrap"].default)),
        input_topic=str(kafka.get("input_topic", Settings.model_fields["input_topic"].default)),
        alert_topic=str(kafka.get("alert_topic", Settings.model_fields["alert_topic"].default)),
        group_id=str(kafka.get("group_id", Settings.model_fields["group_id"].default)),
        window_size=int(drift.get("window_size", Settings.model_fields["window_size"].default)),
        min_samples=int(drift.get("min_samples", Settings.model_fields["min_samples"].default)),
        psi_threshold=float(drift.get("psi_threshold", Settings.model_fields["psi_threshold"].default)),
        ks_threshold=float(drift.get("ks_threshold", Settings.model_fields["ks_threshold"].default)),
        cooldown_seconds=int(drift.get("cooldown_seconds", Settings.model_fields["cooldown_seconds"].default)),
        reference_dir=str(store.get("reference_dir", Settings.model_fields["reference_dir"].default)),
    )
