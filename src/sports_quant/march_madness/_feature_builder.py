"""Build feature vectors for arbitrary team pairings from KenPom data.

Provides a lookup that constructs the exact 36-column feature vectors
that trained LightGBM models expect, for any two teams in a given year.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from sports_quant.march_madness._features import standardize_team_name

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

    Indexes KenPom statistics by (team_name, year) and builds the exact
    36-column feature DataFrames that trained models expect.
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
        return TeamStats(team=name, year=year, seed=seed, features=features)

    def build_matchup_features(
        self,
        team1: TeamStats,
        team2: TeamStats,
    ) -> pd.DataFrame:
        """Build a single-row feature DataFrame for a matchup.

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
