"""Tests for probability calibration module."""

import numpy as np
import pytest

from sports_quant.march_madness._calibration import (
    calibrate_probabilities,
    fit_calibrator,
)


def test_fit_calibrator_isotonic():
    """Test isotonic calibrator fits without error."""
    probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9] * 5)
    labels = np.array([0, 0, 0, 1, 1, 1] * 5)
    calibrator = fit_calibrator(probs, labels, method="isotonic")
    assert calibrator is not None


def test_fit_calibrator_platt():
    """Test Platt scaling calibrator fits without error."""
    probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9] * 5)
    labels = np.array([0, 0, 0, 1, 1, 1] * 5)
    calibrator = fit_calibrator(probs, labels, method="platt")
    assert calibrator is not None


def test_fit_calibrator_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Shape mismatch"):
        fit_calibrator(np.array([0.5, 0.6]), np.array([0, 1, 0]))


def test_fit_calibrator_rejects_too_few_samples():
    with pytest.raises(ValueError, match="Too few samples"):
        fit_calibrator(np.array([0.5] * 5), np.array([0] * 5))


def test_fit_calibrator_rejects_unknown_method():
    probs = np.array([0.5] * 20)
    labels = np.array([0, 1] * 10)
    with pytest.raises(ValueError, match="Unknown calibration method"):
        fit_calibrator(probs, labels, method="unknown")


def test_calibrate_probabilities_clipping():
    """Test that calibrate_probabilities clips to [clip_min, clip_max]."""
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 4)
    labels = np.array([0, 0, 0, 1, 1] * 4)
    calibrator = fit_calibrator(probs, labels, method="isotonic")

    # Test with extreme raw probabilities
    raw = np.array([0.0, 0.01, 0.5, 0.99, 1.0])
    result = calibrate_probabilities(
        calibrator, raw, clip_min=0.025, clip_max=0.975,
    )

    assert result.min() >= 0.025
    assert result.max() <= 0.975
    assert len(result) == len(raw)


def test_calibrate_probabilities_preserves_order():
    """Calibrated probs should maintain monotonic ordering (isotonic)."""
    probs = np.linspace(0.1, 0.9, 30)
    labels = (probs > 0.5).astype(int)
    calibrator = fit_calibrator(probs, labels, method="isotonic")

    raw = np.array([0.2, 0.4, 0.6, 0.8])
    result = calibrate_probabilities(calibrator, raw)

    # Isotonic regression is monotonically increasing
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]
