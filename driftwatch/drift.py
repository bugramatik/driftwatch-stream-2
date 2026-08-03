"""Drift math: population stability index, KS distance and the per-feature verdict."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

EPSILON = 1e-6


class FeatureVerdict(BaseModel):
    """The drift verdict for one feature of one model."""

    feature: str
    psi: float
    ks: float
    drifted: bool


class ModelVerdict(BaseModel):
    """The combined verdict for a model, with the features that drifted."""

    model: str
    version: str
    samples: int
    drifted_features: list[str]
    features: list[FeatureVerdict]

    @property
    def drifted(self) -> bool:
        return bool(self.drifted_features)


def _bin_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Quantile edges of the reference sample, made strictly increasing."""
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        # A constant reference feature: widen the single edge so both sides exist.
        value = float(edges[0]) if edges.size else 0.0
        edges = np.array([value - 0.5, value + 0.5])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index of current against reference, binned on reference quantiles."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("psi needs non-empty reference and current samples")
    if bins < 2:
        raise ValueError("psi needs at least two bins")

    edges = _bin_edges(reference, bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_share = ref_counts / max(ref_counts.sum(), 1)
    cur_share = cur_counts / max(cur_counts.sum(), 1)
    ref_share = np.clip(ref_share, EPSILON, None)
    cur_share = np.clip(cur_share, EPSILON, None)

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance between reference and current."""
    reference = np.sort(np.asarray(reference, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    if reference.size == 0 or current.size == 0:
        raise ValueError("ks_distance needs non-empty reference and current samples")

    grid = np.concatenate([reference, current])
    ref_cdf = np.searchsorted(reference, grid, side="right") / reference.size
    cur_cdf = np.searchsorted(current, grid, side="right") / current.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def evaluate_feature(
    name: str,
    reference: np.ndarray,
    current: np.ndarray,
    psi_threshold: float,
    ks_threshold: float,
) -> FeatureVerdict:
    """Compute both statistics for one feature and flag it when either crosses its threshold."""
    psi_value = psi(reference, current)
    ks_value = ks_distance(reference, current)
    drifted = psi_value >= psi_threshold or ks_value >= ks_threshold
    return FeatureVerdict(
        feature=name,
        psi=round(psi_value, 6),
        ks=round(ks_value, 6),
        drifted=drifted,
    )


def evaluate_model(
    model: str,
    version: str,
    reference: dict[str, list[float]],
    current: dict[str, np.ndarray],
    psi_threshold: float,
    ks_threshold: float,
) -> ModelVerdict:
    """Evaluate every feature the reference profile knows about."""
    verdicts: list[FeatureVerdict] = []
    samples = 0
    for name, ref_values in reference.items():
        cur_values = current.get(name)
        if cur_values is None or cur_values.size == 0:
            continue
        samples = max(samples, int(cur_values.size))
        verdicts.append(
            evaluate_feature(name, np.asarray(ref_values, dtype=float), cur_values, psi_threshold, ks_threshold)
        )
    drifted = [v.feature for v in verdicts if v.drifted]
    return ModelVerdict(
        model=model,
        version=version,
        samples=samples,
        drifted_features=drifted,
        features=verdicts,
    )
