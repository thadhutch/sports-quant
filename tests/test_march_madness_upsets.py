"""Tests for March Madness upset analysis."""

import numpy as np
import pandas as pd

from sports_quant.march_madness._upsets import analyze_upsets, track_popular_upsets


def _make_team_data():
    """Create sample team data for upset analysis."""
    return pd.DataFrame({
        "Seed1": [1, 2, 8, 3],
        "Seed2": [16, 15, 9, 14],
        "Team1": ["Duke", "Kansas", "UCLA", "Purdue"],
        "Team2": ["FDU", "Vermont", "Michigan", "Grambling"],
        "YEAR": [2023, 2023, 2023, 2023],
        "Team1_Win": [1, 1, 0, 1],
    })


def test_analyze_upsets_no_upsets():
    """All favorites win, no upsets."""
    team_data = _make_team_data()
    actual = np.array([1, 1, 1, 1])  # All Team1 (favorites) win
    predicted = np.array([1, 1, 1, 1])

    results = analyze_upsets(actual, predicted, team_data)
    assert results["total_games"] == 4
    assert results["total_upsets_predicted"] == 0


def test_analyze_upsets_counts_correctly():
    """Verify upset counting when upsets are predicted."""
    team_data = _make_team_data()
    # Actual: Team1 wins games 1,2,4; Team2 wins game 3
    actual = np.array([1, 1, 0, 1])
    # Predicted: predict upset in game 1 (16-seed over 1-seed)
    # Game 3: pred=0 means 9-seed (Team2) beats 8-seed (Team1) = also upset
    predicted = np.array([0, 1, 0, 1])

    results = analyze_upsets(actual, predicted, team_data)
    assert results["total_upsets_predicted"] == 2  # game 1 + game 3
    assert results["total_upsets_actual"] == 1  # game 3: 9-seed beat 8-seed


def test_analyze_upsets_empty():
    team_data = pd.DataFrame({
        "Seed1": [], "Seed2": [], "Team1": [], "Team2": [],
        "YEAR": [], "Team1_Win": [],
    })
    results = analyze_upsets(np.array([]), np.array([]), team_data)
    assert results["total_games"] == 0
    assert results["total_upsets_predicted"] == 0


def test_track_popular_upsets_sorting():
    """Results should be sorted by count descending."""
    team_data = pd.DataFrame({
        "Seed1": [1, 2],
        "Seed2": [16, 15],
        "Team1": ["Duke", "Kansas"],
        "Team2": ["FDU", "Vermont"],
        "YEAR": [2023, 2023],
        "Team1_Win": [1, 1],
    })

    # Model 1: predicts upset in game 1 only
    # Model 2: predicts upset in both games
    all_model_data = [
        {"y_backtest_pred": np.array([0, 1])},
        {"y_backtest_pred": np.array([0, 0])},
    ]

    upsets = track_popular_upsets(all_model_data, team_data)

    # FDU upset was predicted by both models, Vermont by 1
    assert len(upsets) == 2
    assert upsets[0]["count"] >= upsets[1]["count"]
    assert upsets[0]["underdog"] == "FDU"
