"""Kafka consumer loop: fill rolling windows, evaluate drift, publish alerts."""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from typing import Any, Protocol

from confluent_kafka import Consumer, KafkaError, Producer

from driftwatch.config import Settings, load_settings
from driftwatch.drift import FeatureVerdict, evaluate_feature
from driftwatch.profiles import ProfileStore
from driftwatch.window import RollingWindow

log = logging.getLogger("driftwatch.consumer")


class ProducerLike(Protocol):
    def send(self, topic: str, value: dict[str, Any]) -> None: ...


class DriftConsumer:
    """Turns a stream of feature and prediction events into drift verdicts."""

    def __init__(self, settings: Settings, store: ProfileStore, producer: ProducerLike) -> None:
        self.settings = settings
        self.store = store
        self.producer = producer
        self.windows: dict[str, RollingWindow] = {}
        self.references: dict[str, dict[str, list[float]]] = {}
        self.last_alert: dict[tuple[str, str], float] = {}
        self.verdicts: dict[str, list[FeatureVerdict]] = {}

    def _window(self, model: str) -> RollingWindow:
        window = self.windows.get(model)
        if window is None:
            window = RollingWindow(self.settings.window_size)
            self.windows[model] = window
        return window

    def _reference(self, model: str) -> dict[str, list[float]] | None:
        if model in self.references:
            return self.references[model]
        try:
            version, profile = self.store.latest(model)
        except FileNotFoundError:
            log.warning("no reference profile for model %s, skipping", model)
            return None
        log.info("loaded reference %s for model %s", version, model)
        self.references[model] = profile
        return profile

    def handle_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Add one event to its model's window and return any alerts raised."""
        model = event.get("model")
        features = event.get("features")
        if not model or not isinstance(features, dict):
            return []

        window = self._window(model)
        window.add(event)
        if not window.ready(self.settings.window_size):
            return []

        reference = self._reference(model)
        if reference is None:
            return []

        current = window.features()
        now = time.time()
        alerts: list[dict[str, Any]] = []
        verdicts: list[FeatureVerdict] = []

        for name, ref_values in reference.items():
            values = current.get(name)
            if values is None or values.size == 0:
                continue
            verdict = evaluate_feature(
                name,
                ref_values,
                values,
                self.settings.psi_threshold,
                self.settings.ks_threshold,
            )
            verdicts.append(verdict)
            if not verdict.drifted:
                continue
            key = (model, name)
            last = self.last_alert.get(key, 0.0)
            if now - last < self.settings.cooldown_seconds:
                continue
            self.last_alert[key] = now
            alert = {
                "model": model,
                "version": event.get("version", "unknown"),
                "feature": name,
                "psi": round(verdict.psi, 4),
                "ks": round(verdict.ks, 4),
                "samples": int(values.size),
                "raised_at": int(now),
            }
            alerts.append(alert)
            self.producer.send(self.settings.alert_topic, alert)

        self.verdicts[model] = verdicts
        return alerts

    def status(self) -> dict[str, Any]:
        """Snapshot of the current verdicts, used by the API when it shares a process."""
        return {
            model: [v.model_dump() for v in verdicts]
            for model, verdicts in self.verdicts.items()
        }

    def run(self, stop: threading.Event | None = None) -> None:
        consumer = Consumer(
            {
                "bootstrap.servers": self.settings.kafka_bootstrap,
                "group.id": self.settings.group_id,
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
            }
        )
        consumer.subscribe([self.settings.input_topic])
        stop = stop or threading.Event()

        def _shutdown(*_: Any) -> None:
            stop.set()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        log.info("consuming %s as %s", self.settings.input_topic, self.settings.group_id)
        try:
            while not stop.is_set():
                msg = consumer.poll(0.5)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    log.error("kafka error: %s", msg.error())
                    continue
                try:
                    event = json.loads(msg.value())
                except (TypeError, ValueError):
                    log.warning("dropping malformed event on %s", msg.topic())
                    continue
                for alert in self.handle_event(event):
                    log.info("drift alert %s/%s psi=%.3f", alert["model"], alert["feature"], alert["psi"])
        finally:
            consumer.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings("config/driftwatch.yaml")
    store = ProfileStore(settings.reference_dir)
    producer = Producer({"bootstrap.servers": settings.kafka_bootstrap})

    class _Producer:
        def send(self, topic: str, value: dict[str, Any]) -> None:
            producer.produce(topic, json.dumps(value).encode("utf-8"))
            producer.poll(0)

    DriftConsumer(settings, store, _Producer()).run()
    producer.flush(5)


if __name__ == "__main__":
    main()
