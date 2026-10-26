"""Tests for the drift math and the per-feature verdict."""

import numpy as np
import pytest

from driftwatch.drift import evaluate_feature, ks_distance, psi


def test_psi_identical_is_zero():
    rng = np.random.default_rng(7)
    sample = rng.normal(0.0, 1.0, 5000)
    assert psi(sample, sample.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_shifted_is_large():
    rng = np.random.default_rng(11)
    reference = rng.normal(0.0, 1.0, 5000)
    current = rng.normal(1.5, 1.0, 5000)
    value = psi(reference, current)
    assert value > 0.25
    assert np.isfinite(value)


def test_psi_handles_constant_reference():
    reference = np.zeros(200)
    current = np.zeros(200)
    assert psi(reference, current) == pytest.approx(0.0, abs=1e-9)


def test_ks_detects_shift():
    rng = np.random.default_rng(3)
    reference = rng.normal(0.0, 1.0, 2000)
    same = rng.normal(0.0, 1.0, 2000)
    shifted = rng.normal(2.0, 1.0, 2000)
    assert ks_distance(reference, same) < 0.1
    assert ks_distance(reference, shifted) > 0.5


def test_evaluate_feature_flags_drift():
    rng = np.random.default_rng(5)
    reference = rng.normal(0.0, 1.0, 3000)
    current = rng.normal(1.2, 1.0, 3000)
    verdict = evaluate_feature(
        "amount", reference, current, psi_threshold=0.2, ks_threshold=0.1
    )
    assert verdict.feature == "amount"
    assert verdict.drifted is True
    assert verdict.psi > 0.2
    assert verdict.ks > 0.1


def test_evaluate_feature_stable():
    rng = np.random.default_rng(9)
    reference = rng.normal(0.0, 1.0, 3000)
    current = rng.normal(0.0, 1.0, 3000)
    verdict = evaluate_feature(
        "amount", reference, current, psi_threshold=0.2, ks_threshold=0.1
    )
    assert verdict.drifted is False
