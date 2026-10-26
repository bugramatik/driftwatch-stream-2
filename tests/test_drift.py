"""Tests for the drift math and the per-feature verdict."""

import numpy as np
import pytest

from driftwatch.drift import evaluate_feature, ks_distance, psi


def test_psi_identical_is_zero():
    rng = np.random.default_rng(7)
    sample = rng.normal(0.0, 1.0, 5000)
    assert psi(sample, sample.copy()) == pytest.approx(0.0, abs=1e-9)

