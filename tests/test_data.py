"""Tests for data loading and preparation."""

import numpy as np
import pandas as pd
import pytest

from sports_quant.modeling._data import DATE_COLUMN, TARGET_COLUMN, load_and_prepare
from sports_quant.modeling._features import ALL_FEATURES


@pytest.fixture()
def synthetic_ranked_csv(tmp_path, monkeypatch):
    """Create a minimal synthetic ranked CSV and point config at it."""
    import sports_quant._config as config

    rng = np.random.RandomState(42)
    n = 300
    dates = pd.date_range("2020-09-10", periods=n, freq="7D")
    seasons = np.where(dates.month >= 9, dates.year, dates.year - 1)

    data = {DATE_COLUMN: dates, "season": seasons}
    for feat in ALL_FEATURES:
        data[feat] = rng.rand(n)
    data[TARGET_COLUMN] = rng.choice([0, 1, 2], size=n)
    data["home_gp"] = np.where(np.arange(n) % 18 == 0, 0, rng.randint(1, 17, n))
    data["away_gp"] = np.where(np.arange(n) % 20 == 0, 0, rng.randint(1, 17, n))

    csv_path = tmp_path / "ranked.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    monkeypatch.setattr(config, "OVERUNDER_RANKED", csv_path)
    return csv_path


def test_load_and_prepare_returns_prepared_data(synthetic_ranked_csv):
    result = load_and_prepare(min_training_seasons=2)

    assert result.df is not None
    assert len(result.X.columns) == len(ALL_FEATURES)
    assert result.y.dtype in (np.int64, np.int32, int)
    assert len(result.seasons) >= 3
    assert len(result.test_dates) > 0


def test_load_and_prepare_filters_week1(synthetic_ranked_csv):
    result = load_and_prepare(min_training_seasons=2)

    # All rows should have gp > 0
    assert (result.df["home_gp"] > 0).all()
    assert (result.df["away_gp"] > 0).all()


def test_load_and_prepare_insufficient_seasons(tmp_path, monkeypatch):
    """Should raise ValueError if fewer seasons than required."""
    import sports_quant._config as config

    rng = np.random.RandomState(0)
    n = 50
    dates = pd.date_range("2023-09-10", periods=n, freq="7D")

    data = {DATE_COLUMN: dates, "season": 2023}
    for feat in ALL_FEATURES:
        data[feat] = rng.rand(n)
    data[TARGET_COLUMN] = rng.choice([0, 1, 2], size=n)
    data["home_gp"] = 5
    data["away_gp"] = 5

    csv_path = tmp_path / "ranked_one_season.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    monkeypatch.setattr(config, "OVERUNDER_RANKED", csv_path)

    with pytest.raises(ValueError, match="Need at least"):
        load_and_prepare(min_training_seasons=2)
