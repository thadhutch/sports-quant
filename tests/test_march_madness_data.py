"""Tests for March Madness data loading."""

import pandas as pd
import numpy as np
import pytest

from sports_quant.march_madness._data import (
    compute_difference_features,
    load_and_prepare,
    load_prediction_data,
    symmetrize_training_data,
)
from sports_quant.march_madness._features import (
    DIFF_FEATURE_COLUMNS,
    DROP_COLUMNS,
    STAT_PAIRS,
    TARGET_COLUMN,
)


def _make_full_kenpom_row(
    year, team1, seed1, team2, seed2, team1_win,
    adjEM_t1=25.0, adjEM_t2=5.0,
):
    """Build a single training row with all required KenPom columns."""
    return {
        "YEAR": year,
        "BY YEAR NO": 1,
        "BY ROUND NO": 1,
        "Team1_NO": 1,
        "Team2_NO": 16,
        "Team": team1,
        "Team_Team2": team2,
        "Year": year,
        "Year_Team2": year,
        "Score1": 80,
        "Score2": 60,
        "ROUND": 1,
        "ROUND.1": 1,
        "CURRENT ROUND.1": 1,
        "W-L": "30-5",
        "W-L_Team2": "15-18",
        "Team1": team1,
        "Team2": team2,
        "Conference": "ACC",
        "Conference_Team2": "NEC",
        "Seed1": seed1,
        "Seed2": seed2,
        "CURRENT ROUND": 64,
        "Team1_Win": team1_win,
        # Team1 KenPom stats
        "Rank": 5,
        "AdjEM": adjEM_t1,
        "AdjustO": 120.0,
        "AdjustO Rank": 3,
        "AdjustD": 95.0,
        "AdjustD Rank": 10,
        "AdjustT": 68.0,
        "AdjustT Rank": 50,
        "Luck": 0.02,
        "Luck Rank": 100,
        "SOS AdjEM": 10.0,
        "SOS AdjEM Rank": 20,
        "SOS OppO": 108.0,
        "SOS OppO Rank": 25,
        "SOS OppD": 98.0,
        "SOS OppD Rank": 30,
        "NCSOS AdjEM": 5.0,
        "NCSOS AdjEM Rank": 40,
        # Team2 KenPom stats
        "Rank_Team2": 200,
        "AdjEM_Team2": adjEM_t2,
        "AdjustO_Team2": 105.0,
        "AdjustO Rank_Team2": 100,
        "AdjustD_Team2": 100.0,
        "AdjustD Rank_Team2": 80,
        "AdjustT_Team2": 70.0,
        "AdjustT Rank_Team2": 40,
        "Luck_Team2": -0.01,
        "Luck Rank_Team2": 200,
        "SOS AdjEM_Team2": 2.0,
        "SOS AdjEM Rank_Team2": 150,
        "SOS OppO_Team2": 106.0,
        "SOS OppO Rank_Team2": 120,
        "SOS OppD_Team2": 104.0,
        "SOS OppD Rank_Team2": 100,
        "NCSOS AdjEM_Team2": 1.0,
        "NCSOS AdjEM Rank_Team2": 180,
    }


@pytest.fixture
def training_csv_minimal(tmp_path, monkeypatch):
    """Minimal training CSV with only Rank/AdjEM (for raw mode tests)."""
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
def training_csv(tmp_path, monkeypatch):
    """Full training CSV with all KenPom columns (for difference mode)."""
    rows = [
        _make_full_kenpom_row(2022, "Duke", 1, "FDU", 16, 1, 30.0, 5.0),
        _make_full_kenpom_row(2022, "Kansas", 2, "Vermont", 15, 1, 25.0, 10.0),
        _make_full_kenpom_row(2023, "Duke", 1, "FDU", 16, 1, 29.0, 4.0),
        _make_full_kenpom_row(2023, "Kansas", 2, "Vermont", 15, 0, 24.0, 9.0),
    ]
    csv_path = tmp_path / "training_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    import sports_quant.march_madness._config as mm_config
    monkeypatch.setattr(mm_config, "MM_TRAINING_DATA", csv_path)
    return csv_path


@pytest.fixture
def prediction_csv(tmp_path, monkeypatch):
    """Full prediction CSV with all KenPom columns."""
    rows = [
        _make_full_kenpom_row(2025, "Duke", 1, "FDU", 16, 0, 30.0, 5.0),
        _make_full_kenpom_row(2025, "Kansas", 2, "Vermont", 15, 0, 25.0, 10.0),
    ]
    # Remove Team1_Win (prediction data doesn't have outcomes)
    for row in rows:
        del row["Team1_Win"]

    csv_path = tmp_path / "prediction_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    import sports_quant.march_madness._config as mm_config
    monkeypatch.setattr(mm_config, "MM_PREDICTION_DATA", csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Raw mode tests (backward compatibility)
# ---------------------------------------------------------------------------


def test_load_and_prepare_raw_mode(training_csv_minimal):
    result = load_and_prepare(feature_mode="raw")
    assert hasattr(result, "X")
    assert hasattr(result, "y")
    assert hasattr(result, "years")
    assert hasattr(result, "team_info")
    assert len(result.X) == 4
    assert len(result.y) == 4


def test_load_and_prepare_raw_drops_columns(training_csv_minimal):
    result = load_and_prepare(feature_mode="raw")
    for col in DROP_COLUMNS:
        assert col not in result.X.columns


def test_load_and_prepare_raw_target_not_in_features(training_csv_minimal):
    result = load_and_prepare(feature_mode="raw")
    assert TARGET_COLUMN not in result.X.columns


def test_load_and_prepare_years(training_csv_minimal):
    result = load_and_prepare(feature_mode="raw")
    assert result.years == [2022, 2023]


def test_load_prediction_data_raw(prediction_csv):
    X_pred, team_info = load_prediction_data(feature_mode="raw")
    assert len(X_pred) == 2
    assert "Team1" in team_info.columns
    assert "Team2" in team_info.columns
    assert TARGET_COLUMN not in X_pred.columns


# ---------------------------------------------------------------------------
# Difference mode tests
# ---------------------------------------------------------------------------


def test_load_and_prepare_difference_mode(training_csv):
    result = load_and_prepare(feature_mode="difference")
    assert len(result.X) == 4
    assert len(result.y) == 4
    assert set(result.X.columns) == set(DIFF_FEATURE_COLUMNS)
    assert result.X.shape[1] == len(DIFF_FEATURE_COLUMNS)


def test_difference_features_values(training_csv):
    result = load_and_prepare(feature_mode="difference")
    X = result.X

    # Seed diff: Seed1=1, Seed2=16 -> -15
    assert X["seed_diff"].iloc[0] == 1 - 16

    # AdjEM diff: 30.0 - 5.0 = 25.0 (first row)
    assert X["adjEM_diff"].iloc[0] == 25.0


def test_difference_features_no_raw_columns(training_csv):
    result = load_and_prepare(feature_mode="difference")
    for col in result.X.columns:
        assert not col.endswith("_Team2"), f"Raw column {col} leaked through"
    assert "Rank" not in result.X.columns
    assert "AdjEM" not in result.X.columns


def test_load_prediction_data_difference(prediction_csv):
    X_pred, team_info = load_prediction_data(feature_mode="difference")
    assert len(X_pred) == 2
    assert set(X_pred.columns) == set(DIFF_FEATURE_COLUMNS)
    assert "Team1" in team_info.columns


def test_default_feature_mode_is_difference(training_csv):
    result = load_and_prepare()
    assert set(result.X.columns) == set(DIFF_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Compute difference features
# ---------------------------------------------------------------------------


def test_compute_difference_features_columns():
    """Test that compute_difference_features produces correct column set."""
    row = _make_full_kenpom_row(2023, "A", 1, "B", 16, 1, 25.0, 5.0)
    df = pd.DataFrame([row])
    result = compute_difference_features(df)

    assert list(result.columns) == list(DIFF_FEATURE_COLUMNS)
    assert len(result) == 1


def test_compute_difference_features_math():
    """Test that stat diffs are correctly computed."""
    row = _make_full_kenpom_row(2023, "A", 1, "B", 16, 1, 25.0, 5.0)
    df = pd.DataFrame([row])
    result = compute_difference_features(df)

    # adjEM_diff = 25.0 - 5.0 = 20.0
    assert result["adjEM_diff"].iloc[0] == 20.0

    # seed_diff = 1 - 16 = -15
    assert result["seed_diff"].iloc[0] == -15

    # rank_diff = 5 - 200 = -195
    assert result["rank_diff"].iloc[0] == 5 - 200


def test_compute_difference_efficiency_ratio():
    """Test the derived efficiency_ratio_diff feature."""
    row = _make_full_kenpom_row(2023, "A", 1, "B", 16, 1)
    df = pd.DataFrame([row])
    result = compute_difference_features(df)

    # efficiency_ratio_diff = (120/95) - (105/100)
    expected = (120.0 / 95.0) - (105.0 / 100.0)
    assert abs(result["efficiency_ratio_diff"].iloc[0] - expected) < 1e-10


def test_compute_difference_interaction():
    """Test the seed_x_adjEM_interaction derived feature."""
    row = _make_full_kenpom_row(2023, "A", 1, "B", 16, 1, 25.0, 5.0)
    df = pd.DataFrame([row])
    result = compute_difference_features(df)

    seed_diff = 1 - 16  # -15
    adjEM_diff = 25.0 - 5.0  # 20.0
    expected = seed_diff * adjEM_diff  # -300.0
    assert result["seed_x_adjEM_interaction"].iloc[0] == expected


# ---------------------------------------------------------------------------
# Symmetrization tests
# ---------------------------------------------------------------------------


def test_symmetrize_doubles_rows(training_csv):
    result = load_and_prepare(feature_mode="difference")
    X_sym, y_sym = symmetrize_training_data(result.X, result.y)

    assert len(X_sym) == 2 * len(result.X)
    assert len(y_sym) == 2 * len(result.y)


def test_symmetrize_negates_first_order_features(training_csv):
    result = load_and_prepare(feature_mode="difference")
    X_sym, y_sym = symmetrize_training_data(result.X, result.y)

    n = len(result.X)
    original = X_sym.iloc[:n].reset_index(drop=True)
    flipped = X_sym.iloc[n:].reset_index(drop=True)

    # First-order diffs are negated
    first_order = [c for c in DIFF_FEATURE_COLUMNS if c != "seed_x_adjEM_interaction"]
    for col in first_order:
        pd.testing.assert_series_equal(
            original[col], -flipped[col], check_names=False,
        )

    # Product feature is recomputed (stays same since (-a)*(-b) = a*b)
    pd.testing.assert_series_equal(
        original["seed_x_adjEM_interaction"],
        flipped["seed_x_adjEM_interaction"],
        check_names=False,
    )


def test_symmetrize_flips_labels(training_csv):
    result = load_and_prepare(feature_mode="difference")
    X_sym, y_sym = symmetrize_training_data(result.X, result.y)

    n = len(result.y)
    original_labels = y_sym.iloc[:n].reset_index(drop=True)
    flipped_labels = y_sym.iloc[n:].reset_index(drop=True)

    pd.testing.assert_series_equal(original_labels, 1 - flipped_labels)


def test_symmetrize_rejects_raw_features(training_csv_minimal):
    result = load_and_prepare(feature_mode="raw")
    with pytest.raises(ValueError, match="difference features"):
        symmetrize_training_data(result.X, result.y)
