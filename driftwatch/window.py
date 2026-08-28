"""Rolling per-model window of feature values and predictions."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict

import numpy as np


class RollingWindow:
    """Keeps the last ``size`` events for one model in memory.

    Each event is a dict with a ``features`` mapping of name to number and an
    optional ``prediction`` number. The window drops the oldest event once it is
    full, so memory stays bounded no matter how long the consumer runs.
    """

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("window size must be positive")
        self.size = size
        self._events: Deque[Dict[str, Any]] = deque(maxlen=size)

    def add(self, event: Dict[str, Any]) -> None:
        """Append one event, keeping only its features and prediction."""
        features = event.get("features") or {}
        if not isinstance(features, dict):
            raise TypeError("event features must be a mapping")
        cleaned: Dict[str, float] = {}
        for name, value in features.items():
            if value is None:
                continue
            try:
                cleaned[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        prediction = event.get("prediction")
        try:
            pred = float(prediction) if prediction is not None else None
        except (TypeError, ValueError):
            pred = None
        self._events.append({"features": cleaned, "prediction": pred})

    def features(self) -> Dict[str, np.ndarray]:
        """Return one array per feature name, in first-seen order."""
        names: list[str] = []
        seen: set[str] = set()
        for event in self._events:
            for name in event["features"]:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        out: Dict[str, np.ndarray] = {}
        for name in names:
            values = [
                event["features"][name]
                for event in self._events
                if name in event["features"]
            ]
            out[name] = np.asarray(values, dtype=float)
        return out

    def predictions(self) -> np.ndarray:
        """Return the predictions seen in the window, skipping missing ones."""
        values = [
            event["prediction"]
            for event in self._events
            if event["prediction"] is not None
        ]
        return np.asarray(values, dtype=float)

    def ready(self, min_samples: int) -> bool:
        """True when the window holds at least ``min_samples`` events."""
        return len(self._events) >= min_samples

    def reset(self) -> None:
        """Drop every event, for example after a new reference profile lands."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)
