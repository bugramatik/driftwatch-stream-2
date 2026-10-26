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


def _seed(store, model="churn", version="v1", shift=0.0, n=400):
    rng = np.random.default_rng(42)
    profile = {
        "amount": rng.normal(shift, 1.0, n).tolist(),
        "tenure": rng.normal(0.0, 1.0, n).tolist(),
    }
    store.save(model, version, profile)


def _events(model="churn", version="v1", shift=0.0, count=120):
    rng = np.random.default_rng(43)
    for _ in range(count):
        yield {
            "model": model,
            "version": version,
            "features": {
                "amount": float(rng.normal(shift, 1.0)),
                "tenure": float(rng.normal(0.0, 1.0)),
            },
            "prediction": float(rng.random()),
        }


def test_handle_event_emits_alert(settings):
    store = ProfileStore(settings.reference_dir)
    _seed(store, shift=0.0)
    producer = FakeProducer()
    consumer = DriftConsumer(settings, store, producer)

    alerts = []
    for event in _events(shift=2.0):
        alerts.extend(consumer.handle_event(event))

    assert alerts, "a shifted feature should raise an alert"
    assert producer.sent, "the alert should be published"
    topic, payload = producer.sent[0]
    assert topic == settings.alert_topic
    assert payload["model"] == "churn"
    assert any(a["feature"] == "amount" for a in alerts)


def test_cooldown_suppresses_repeat(settings):
    store = ProfileStore(settings.reference_dir)
    _seed(store, shift=0.0)
    producer = FakeProducer()
    consumer = DriftConsumer(settings, store, producer)

    first = []
    for event in _events(shift=2.0):
        first.extend(consumer.handle_event(event))
    assert first

    second = []
    for event in _events(shift=2.0):
        second.extend(consumer.handle_event(event))
    assert second == [], "the cooldown should hold back a repeat alert"


def test_unknown_model_is_ignored(settings):
    store = ProfileStore(settings.reference_dir)
    _seed(store, model="churn")
    producer = FakeProducer()
    consumer = DriftConsumer(settings, store, producer)

    alerts = []
    for event in _events(model="unseen", shift=3.0):
        alerts.extend(consumer.handle_event(event))

    assert alerts == []
    assert producer.sent == []
