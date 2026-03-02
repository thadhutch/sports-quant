"""Tests for March Madness data loading."""

import pandas as pd
import numpy as np
import pytest

from sports_quant.march_madness._data import load_and_prepare, load_prediction_data
from sports_quant.march_madness._features import DROP_COLUMNS, TARGET_COLUMN


@pytest.fixture
def training_csv(tmp_path, monkeypatch):
    """Create a synthetic training CSV and patch config to point to it."""
    # Build a minimal dataset with required columns
    data = {
        "YEAR": [2022, 2022, 2023, 2023],
        "BY YEAR NO": [1, 2, 1, 2],
        "BY ROUND NO": [1, 1, 1, 1],
        "Team1_NO": [1, 2, 1, 2],
        "Team2_NO": [16, 15, 16, 15],
        "Team": ["Duke", "Kansas", "Duke", "Kansas"],
        "Team_Team2": ["FDU", "Vermont", "FDU", "Vermont"],
        "Year": [2022, 2022, 2023, 2023],
        "Year_Team2": [2022, 2022, 2023, 2023],
        "Score1": [80, 75, 85, 70],
        "Score2": [60, 65, 55, 72],
        "ROUND": [1, 1, 1, 1],
        "ROUND.1": [1, 1, 1, 1],
        "CURRENT ROUND.1": [1, 1, 1, 1],
        "W-L": ["30-5", "28-7", "30-5", "28-7"],
        "W-L_Team2": ["15-18", "20-12", "15-18", "20-12"],
        "Team1": ["Duke", "Kansas", "Duke", "Kansas"],
        "Team2": ["FDU", "Vermont", "FDU", "Vermont"],
        "Conference": ["ACC", "B12", "ACC", "B12"],
        "Conference_Team2": ["NEC", "AE", "NEC", "AE"],
        "Seed1": [1, 2, 1, 2],
        "Seed2": [16, 15, 16, 15],
        "CURRENT ROUND": [1, 1, 1, 1],
        "Team1_Win": [1, 1, 1, 0],
        # Actual features (after drop)
        "Rank": [1, 5, 2, 6],
        "Rank_Team2": [300, 200, 310, 210],
        "AdjEM": [30.0, 25.0, 29.0, 24.0],
        "AdjEM_Team2": [5.0, 10.0, 4.0, 9.0],
    }
    csv_path = tmp_path / "training_data.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    import sports_quant.march_madness._config as mm_config
    monkeypatch.setattr(mm_config, "MM_TRAINING_DATA", csv_path)
    return csv_path


@pytest.fixture
def prediction_csv(tmp_path, monkeypatch):
    """Create a synthetic prediction CSV (no Team1_Win column)."""
    data = {
        "YEAR": [2025, 2025],
        "BY YEAR NO": [1, 2],
        "BY ROUND NO": [1, 1],
        "Team1_NO": [1, 2],
        "Team2_NO": [16, 15],
        "Team": ["Duke", "Kansas"],
        "Team_Team2": ["FDU", "Vermont"],
        "Year": [2025, 2025],
        "Year_Team2": [2025, 2025],
        "Score1": [0, 0],
        "Score2": [0, 0],
        "ROUND": [1, 1],
        "ROUND.1": [1, 1],
        "CURRENT ROUND.1": [1, 1],
        "W-L": ["30-5", "28-7"],
        "W-L_Team2": ["15-18", "20-12"],
        "Team1": ["Duke", "Kansas"],
        "Team2": ["FDU", "Vermont"],
        "Conference": ["ACC", "B12"],
        "Conference_Team2": ["NEC", "AE"],
        "Seed1": [1, 2],
        "Seed2": [16, 15],
        "CURRENT ROUND": [1, 1],
        "Rank": [1, 5],
        "Rank_Team2": [300, 200],
        "AdjEM": [30.0, 25.0],
        "AdjEM_Team2": [5.0, 10.0],
    }
    csv_path = tmp_path / "prediction_data.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    import sports_quant.march_madness._config as mm_config
    monkeypatch.setattr(mm_config, "MM_PREDICTION_DATA", csv_path)
    return csv_path


def test_load_and_prepare_returns_prepared_data(training_csv):
    result = load_and_prepare()
    assert hasattr(result, "X")
    assert hasattr(result, "y")
    assert hasattr(result, "years")
    assert hasattr(result, "team_info")
    assert len(result.X) == 4
    assert len(result.y) == 4


def test_load_and_prepare_drops_columns(training_csv):
    result = load_and_prepare()
    for col in DROP_COLUMNS:
        assert col not in result.X.columns


def test_load_and_prepare_target_not_in_features(training_csv):
    result = load_and_prepare()
    assert TARGET_COLUMN not in result.X.columns


def test_load_and_prepare_years(training_csv):
    result = load_and_prepare()
    assert result.years == [2022, 2023]


def test_load_prediction_data(prediction_csv):
    X_pred, team_info = load_prediction_data()
    assert len(X_pred) == 2
    assert "Team1" in team_info.columns
    assert "Team2" in team_info.columns
    assert TARGET_COLUMN not in X_pred.columns
