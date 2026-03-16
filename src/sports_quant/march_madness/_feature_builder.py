"""Build feature vectors for arbitrary team pairings from KenPom data.

Provides a lookup that constructs feature vectors (either 36-column raw
or 21-column difference) that trained LightGBM models expect, for any
two teams in a given year.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from sports_quant.march_madness._features import (
    DIFF_FEATURE_COLUMNS,
    STAT_PAIRS,
    standardize_team_name,
)

logger = logging.getLogger(__name__)

# The 18 KenPom stat columns that survive DROP_COLUMNS filtering.
# Order must match training_data.csv after non-feature columns are removed.
STAT_COLUMNS: tuple[str, ...] = (
    "Rank",
    "AdjEM",
    "AdjustO",
    "AdjustO Rank",
    "AdjustD",
    "AdjustD Rank",
    "AdjustT",
    "AdjustT Rank",
    "Luck",
    "Luck Rank",
    "SOS AdjEM",
    "SOS AdjEM Rank",
    "SOS OppO",
    "SOS OppO Rank",
    "SOS OppD",
    "SOS OppD Rank",
    "NCSOS AdjEM",
    "NCSOS AdjEM Rank",
)

# Full 36-column feature order: Team1 stats (bare) then Team2 stats (_Team2).
FEATURE_COLUMNS: tuple[str, ...] = (
    *STAT_COLUMNS,
    *(f"{col}_Team2" for col in STAT_COLUMNS),
)


@dataclass(frozen=True)
class TeamStats:
    """Pre-tournament KenPom statistics for a single team."""

    team: str
    year: int
    seed: int
    features: dict[str, float]


class FeatureLookup:
    """Lookup for constructing matchup feature vectors from KenPom data.

    Indexes KenPom statistics by (team_name, year) and builds feature
    DataFrames (raw 36-column or difference 21-column) that trained
    models expect.
    """

    def __init__(self, kenpom_df: pd.DataFrame) -> None:
        self._index: dict[tuple[str, int], dict[str, float]] = {}

        for _, row in kenpom_df.iterrows():
            name = standardize_team_name(str(row["Team"]).strip())
            year = int(row["Year"])
            features = {col: float(row[col]) for col in STAT_COLUMNS}
            self._index[(name, year)] = features

        logger.info(
            "FeatureLookup built with %d team-year entries", len(self._index),
        )

    @property
    def team_years(self) -> frozenset[tuple[str, int]]:
        """All (team, year) keys in the lookup."""
        return frozenset(self._index.keys())

    def get_team(self, team: str, year: int, seed: int) -> TeamStats:
        """Look up a team's stats by name and year.

        Args:
            team: Team name (will be standardized).
            year: Tournament year.
            seed: Tournament seed (1-16).

        Returns:
            TeamStats with the team's KenPom features.

        Raises:
            KeyError: If team/year combination is not found.
        """
        name = standardize_team_name(team.strip())
        features = self._index.get((name, year))
        if features is None:
            raise KeyError(
                f"No KenPom data for team={name!r}, year={year}. "
                f"Check team name standardization."
            )
        return TeamStats(
            team=name, year=year, seed=seed, features=dict(features),
        )

    def build_matchup_features(
        self,
        team1: TeamStats,
        team2: TeamStats,
    ) -> pd.DataFrame:
        """Build a single-row raw feature DataFrame for a matchup.

        Column layout matches the exact 36-column order that trained
        LightGBM models expect after DROP_COLUMNS are removed.

        Args:
            team1: First team (features become bare column names).
            team2: Second team (features become _Team2 suffixed columns).

        Returns:
            Single-row DataFrame with 36 float64 feature columns.
        """
        row: dict[str, float] = {}
        for col in STAT_COLUMNS:
            row[col] = team1.features[col]
        for col in STAT_COLUMNS:
            row[f"{col}_Team2"] = team2.features[col]

        return pd.DataFrame([row], columns=list(FEATURE_COLUMNS))

    def build_difference_features(
        self,
        team1: TeamStats,
        team2: TeamStats,
    ) -> pd.DataFrame:
        """Build a single-row difference feature DataFrame for a matchup.

        Computes 21 pairwise difference features: 18 stat diffs +
        seed_diff + 2 derived features.

        Args:
            team1: First team.
            team2: Second team.

        Returns:
            Single-row DataFrame with 21 float64 difference feature columns.
        """
        row: dict[str, float] = {}

        # 18 stat differences (using the base stat column names)
        for team1_col, _, diff_name in STAT_PAIRS:
            # STAT_PAIRS team1_col matches STAT_COLUMNS names
            base_col = team1_col
            row[diff_name] = team1.features[base_col] - team2.features[base_col]

        # Seed difference
        row["seed_diff"] = float(team1.seed - team2.seed)

        # Derived: efficiency ratio diff
        ratio_t1 = team1.features["AdjustO"] / team1.features["AdjustD"]
        ratio_t2 = team2.features["AdjustO"] / team2.features["AdjustD"]
        row["efficiency_ratio_diff"] = ratio_t1 - ratio_t2

        # Derived: seed x adjEM interaction
        row["seed_x_adjEM_interaction"] = row["seed_diff"] * row["adjEM_diff"]

        return pd.DataFrame([row], columns=list(DIFF_FEATURE_COLUMNS))
