"""Reference profile storage, one JSON file per model version."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str) -> str:
    cleaned = _SAFE.sub("-", value.strip()).strip("-.")
    if not cleaned:
        raise ValueError(f"invalid profile name: {value!r}")
    return cleaned


class ProfileStore:
    """Reads and writes reference profiles under a directory root.

    Layout: ``<root>/<model>/<version>.json`` where the file holds
    ``{"model": ..., "version": ..., "features": {name: [floats]}}``.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model: str) -> Path:
        return self.root / _slug(model)

    def save(
        self, model: str, version: str, profile: Dict[str, List[float]]
    ) -> Path:
        """Write a profile atomically and return the file path."""
        if not profile:
            raise ValueError("profile has no features")
        model_dir = self._model_dir(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "version": version,
            "features": {
                name: sorted(float(v) for v in values)
                for name, values in profile.items()
            },
        }
        target = model_dir / f"{_slug(version)}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, target)
        return target

    def load(self, model: str, version: str) -> Dict[str, List[float]]:
        """Load one profile, raising FileNotFoundError when it is missing."""
        path = self._model_dir(model) / f"{_slug(version)}.json"
        if not path.is_file():
            raise FileNotFoundError(f"no profile for {model} {version}")
        data = json.loads(path.read_text(encoding="utf-8"))
        features = data.get("features") or {}
        return {
            str(name): [float(v) for v in values]
            for name, values in features.items()
        }

    def latest(self, model: str) -> Tuple[str, Dict[str, List[float]]]:
        """Return the most recently written version and its profile."""
        model_dir = self._model_dir(model)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"no profiles for model {model}")
        files = sorted(
            model_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not files:
            raise FileNotFoundError(f"no profiles for model {model}")
        newest = files[0]
        data = json.loads(newest.read_text(encoding="utf-8"))
        version = str(data.get("version") or newest.stem)
        features = data.get("features") or {}
        return version, {
            str(name): [float(v) for v in values]
            for name, values in features.items()
        }

    def versions(self, model: str) -> List[str]:
        """List the versions stored for a model, newest first."""
        model_dir = self._model_dir(model)
        if not model_dir.is_dir():
            return []
        files = sorted(
            model_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return [p.stem for p in files]

    def models(self) -> List[str]:
        """List the models that have at least one profile."""
        if not self.root.is_dir():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and any(p.glob("*.json"))
        )
