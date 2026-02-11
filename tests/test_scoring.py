"""Tests for scoring utilities."""

from datetime import datetime

import pytest

from sports_quant.modeling._scoring import compute_season_progress


@pytest.mark.parametrize(
    "date, expected_current_approx",
    [
        # Sep 1 = start of season → 0%
        (datetime(2024, 9, 1), 0.0),
        # Jan 15 = end of season → 100%
        (datetime(2025, 1, 15), 1.0),
        # Mid-October ≈ 30-40%
        (datetime(2024, 10, 15), 0.33),
        # Late November ≈ 60-70%
        (datetime(2024, 11, 25), 0.63),
        # December ≈ 70-80%
        (datetime(2024, 12, 15), 0.78),
    ],
)
def test_compute_season_progress(date, expected_current_approx):
    w_cur, w_last = compute_season_progress(date)

    # Weights should sum to 1
    assert abs(w_cur + w_last - 1.0) < 1e-9

    # Current-season weight should be approximately as expected
    assert abs(w_cur - expected_current_approx) < 0.05, (
        f"Expected ~{expected_current_approx}, got {w_cur}"
    )


def test_season_progress_clamps_to_0_1():
    """Dates outside the season window should clamp to [0, 1]."""
    # Well before Sep 1
    w_cur, w_last = compute_season_progress(datetime(2024, 8, 1))
    assert 0.0 <= w_cur <= 1.0
    assert 0.0 <= w_last <= 1.0

    # Well after Jan 15
    w_cur, w_last = compute_season_progress(datetime(2025, 2, 15))
    assert 0.0 <= w_cur <= 1.0
    assert 0.0 <= w_last <= 1.0


def test_weights_sum_to_one():
    for month in range(1, 13):
        w_cur, w_last = compute_season_progress(datetime(2024, month, 15))
        assert abs(w_cur + w_last - 1.0) < 1e-9
