"""Tests for March Madness debiasing algorithm."""

import numpy as np
import pandas as pd

from sports_quant.march_madness._debiasing import (
    swap_team_columns,
    evaluate_debiased_predictions,
)


def _make_feature_df():
    """Create a small feature DataFrame with Team1/Team2 columns."""
    return pd.DataFrame({
        "Rank": [1, 5],
        "Rank_Team2": [16, 12],
        "AdjEM": [30.0, 20.0],
        "AdjEM_Team2": [5.0, 10.0],
        "AdjO": [110.0, 105.0],
        "AdjO_Team2": [95.0, 100.0],
    })


def test_swap_team_columns_preserves_shape():
    df = _make_feature_df()
    swapped = swap_team_columns(df)
    assert swapped.shape == df.shape


def test_swap_team_columns_symmetry():
    """Swapping twice should return the original data."""
    df = _make_feature_df()
    swapped_once = swap_team_columns(df)
    swapped_twice = swap_team_columns(swapped_once)
    pd.testing.assert_frame_equal(swapped_twice, df)


def test_swap_team_columns_values():
    """Team1 values should become Team2 values and vice versa."""
    df = _make_feature_df()
    swapped = swap_team_columns(df)
    assert swapped["Rank"].iloc[0] == 16
    assert swapped["Rank_Team2"].iloc[0] == 1
    assert swapped["AdjEM"].iloc[0] == 5.0
    assert swapped["AdjEM_Team2"].iloc[0] == 30.0


def test_evaluate_debiased_predictions_metrics_keys():
    probs = np.array([0.8, 0.3, 0.6, 0.2])
    y_true = np.array([1, 0, 1, 0])

    metrics, y_pred = evaluate_debiased_predictions(probs, y_true)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert all(0 <= v <= 1 for v in metrics.values())
    assert len(y_pred) == len(y_true)


def test_evaluate_debiased_predictions_perfect():
    probs = np.array([0.9, 0.1, 0.8, 0.2])
    y_true = np.array([1, 0, 1, 0])

    metrics, _ = evaluate_debiased_predictions(probs, y_true)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
