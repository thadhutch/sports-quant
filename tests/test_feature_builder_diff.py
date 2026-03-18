"""Tests for difference feature building in FeatureLookup."""

import pandas as pd
import pytest

from sports_quant.march_madness._feature_builder import FeatureLookup, STAT_COLUMNS
from sports_quant.march_madness._features import DIFF_FEATURE_COLUMNS


@pytest.fixture
def kenpom_df():
    """Minimal KenPom DataFrame with two teams."""
    rows = []
    for team, year, rank, adjEM, adjO, adjD in [
        ("Duke", 2023, 3, 28.0, 120.0, 92.0),
        ("FDU", 2023, 250, -5.0, 100.0, 105.0),
    ]:
        row = {
            "Team": team,
            "Year": year,
            "Rank": rank,
            "AdjEM": adjEM,
            "AdjustO": adjO,
            "AdjustO Rank": rank + 1,
            "AdjustD": adjD,
            "AdjustD Rank": rank + 2,
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
        }
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def lookup(kenpom_df):
    return FeatureLookup(kenpom_df)


def test_build_difference_features_columns(lookup):
    duke = lookup.get_team("Duke", 2023, 1)
    fdu = lookup.get_team("FDU", 2023, 16)

    result = lookup.build_difference_features(duke, fdu)
    assert list(result.columns) == list(DIFF_FEATURE_COLUMNS)
    assert len(result) == 1


def test_build_difference_features_adjEM_diff(lookup):
    duke = lookup.get_team("Duke", 2023, 1)
    fdu = lookup.get_team("FDU", 2023, 16)

    result = lookup.build_difference_features(duke, fdu)
    assert result["adjEM_diff"].iloc[0] == 28.0 - (-5.0)  # 33.0


def test_build_difference_features_seed_diff(lookup):
    duke = lookup.get_team("Duke", 2023, 1)
    fdu = lookup.get_team("FDU", 2023, 16)

    result = lookup.build_difference_features(duke, fdu)
    assert result["seed_diff"].iloc[0] == 1 - 16  # -15


def test_build_difference_features_efficiency_ratio(lookup):
    duke = lookup.get_team("Duke", 2023, 1)
    fdu = lookup.get_team("FDU", 2023, 16)

    result = lookup.build_difference_features(duke, fdu)
    expected = (120.0 / 92.0) - (100.0 / 105.0)
    assert abs(result["efficiency_ratio_diff"].iloc[0] - expected) < 1e-10


def test_build_difference_features_interaction(lookup):
    duke = lookup.get_team("Duke", 2023, 1)
    fdu = lookup.get_team("FDU", 2023, 16)

    result = lookup.build_difference_features(duke, fdu)
    seed_diff = 1 - 16
    adjEM_diff = 28.0 - (-5.0)
    expected = seed_diff * adjEM_diff
    assert result["seed_x_adjEM_interaction"].iloc[0] == expected


def test_build_difference_features_symmetry(lookup):
    """First-order diffs: diff(A,B) = -diff(B,A).
    Product features: product(A,B) = product(B,A) (even under swap).
    """
    duke = lookup.get_team("Duke", 2023, 1)
    fdu = lookup.get_team("FDU", 2023, 16)

    ab = lookup.build_difference_features(duke, fdu)
    ba = lookup.build_difference_features(fdu, duke)

    for col in DIFF_FEATURE_COLUMNS:
        if col == "seed_x_adjEM_interaction":
            # Product of two diffs is even: (-a)*(-b) = a*b
            assert abs(ab[col].iloc[0] - ba[col].iloc[0]) < 1e-10, (
                f"{col}: {ab[col].iloc[0]} != {ba[col].iloc[0]}"
            )
        else:
            # First-order diffs are odd: diff(A,B) = -diff(B,A)
            assert abs(ab[col].iloc[0] + ba[col].iloc[0]) < 1e-10, (
                f"{col}: {ab[col].iloc[0]} != -{ba[col].iloc[0]}"
            )


def test_build_matchup_features_still_works(lookup):
    """Ensure raw 36-column method is preserved."""
    duke = lookup.get_team("Duke", 2023, 1)
    fdu = lookup.get_team("FDU", 2023, 16)

    result = lookup.build_matchup_features(duke, fdu)
    assert result.shape[1] == 36
