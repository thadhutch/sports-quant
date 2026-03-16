"""Tests for debiasing with difference features."""

import pandas as pd
import numpy as np
import pytest

from sports_quant.march_madness._debiasing import (
    swap_difference_features,
    swap_team_columns,
    _is_difference_features,
)
from sports_quant.march_madness._features import DIFF_FEATURE_COLUMNS


def test_swap_difference_features_negates():
    """Swapping difference features should negate all values."""
    data = {col: [float(i + 1)] for i, col in enumerate(DIFF_FEATURE_COLUMNS)}
    X = pd.DataFrame(data)

    swapped = swap_difference_features(X)

    for col in DIFF_FEATURE_COLUMNS:
        assert swapped[col].iloc[0] == -X[col].iloc[0]


def test_swap_difference_features_double_swap_identity():
    """Swapping twice should return original values."""
    data = {col: [float(i + 1)] for i, col in enumerate(DIFF_FEATURE_COLUMNS)}
    X = pd.DataFrame(data)

    double_swapped = swap_difference_features(swap_difference_features(X))

    pd.testing.assert_frame_equal(X, double_swapped)


def test_is_difference_features_true():
    data = {col: [1.0] for col in DIFF_FEATURE_COLUMNS}
    X = pd.DataFrame(data)
    assert _is_difference_features(X)


def test_is_difference_features_false():
    X = pd.DataFrame({"Rank": [1], "Rank_Team2": [200]})
    assert not _is_difference_features(X)


def test_swap_team_columns_still_works():
    """Raw column swap preserved for backward compatibility."""
    X = pd.DataFrame({
        "Rank": [5.0],
        "Rank_Team2": [200.0],
        "AdjEM": [25.0],
        "AdjEM_Team2": [5.0],
    })
    swapped = swap_team_columns(X)
    assert swapped["Rank"].iloc[0] == 200.0
    assert swapped["Rank_Team2"].iloc[0] == 5.0
