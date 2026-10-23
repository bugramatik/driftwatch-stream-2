"""Tests for the consumer's window handling, alerts and cooldown."""

import numpy as np
import pytest

from driftwatch.config import Settings
from driftwatch.consumer import DriftConsumer
from driftwatch.profiles import ProfileStore


class FakeProducer:
    """Collects produced messages instead of talking to Kafka."""

    def __init__(self):
        self.sent = []

    def send(self, topic, value):
        self.sent.append((topic, value))

    def flush(self, timeout=0):
        return 0


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        kafka_bootstrap="localhost:9092",
        input_topic="features",
        alert_topic="alerts",
        group_id="driftwatch-test",
        window_size=200,
        psi_threshold=0.2,
        ks_threshold=0.1,
        cooldown_seconds=60,
        reference_dir=str(tmp_path / "refs"),
        min_samples=50,
    )

