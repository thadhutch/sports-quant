"""Build feature vectors for arbitrary team pairings from KenPom and Barttorvik data.

Provides a lookup that constructs feature vectors (either 36-column raw,
21-column KenPom difference, or 34-column combined difference) that
trained LightGBM models expect, for any two teams in a given year.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from sports_quant.march_madness._features import (
    BARTTORVIK_STAT_COLUMNS,
    BARTTORVIK_STAT_PAIRS,
    COMBINED_DIFF_FEATURE_COLUMNS,
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
    """Pre-tournament statistics for a single team.

    Holds KenPom features and optionally Barttorvik features.
    """

    team: str
    year: int
    seed: int
    features: dict[str, float]
    barttorvik_features: dict[str, float] | None = None


class FeatureLookup:
    """Lookup for constructing matchup feature vectors from KenPom and Barttorvik data.

    Indexes statistics by (team_name, year) and builds feature
    DataFrames (raw 36-column, difference 21-column, or combined
    34-column) that trained models expect.
    """

    def __init__(
        self,
        kenpom_df: pd.DataFrame,
        barttorvik_df: pd.DataFrame | None = None,
    ) -> None:
        self._kenpom_index: dict[tuple[str, int], dict[str, float]] = {}
        self._bart_index: dict[tuple[str, int], dict[str, float]] = {}

        for _, row in kenpom_df.iterrows():
            name = standardize_team_name(str(row["Team"]).strip())
            year = int(row["Year"])
            features = {col: float(row[col]) for col in STAT_COLUMNS}
            self._kenpom_index[(name, year)] = features

        logger.info(
            "FeatureLookup: %d KenPom team-year entries",
            len(self._kenpom_index),
        )

        if barttorvik_df is not None:
            for _, row in barttorvik_df.iterrows():
                name = standardize_team_name(str(row["Team"]).strip())
                year = int(row["Year"])
                features = {}
                for col in BARTTORVIK_STAT_COLUMNS:
                    val = row.get(col)
                    features[col] = float(val) if pd.notna(val) else 0.0
                self._bart_index[(name, year)] = features

            logger.info(
                "FeatureLookup: %d Barttorvik team-year entries",
                len(self._bart_index),
            )

    @property
    def has_barttorvik(self) -> bool:
        """Whether Barttorvik data is loaded."""
        return len(self._bart_index) > 0

    @property
    def team_years(self) -> frozenset[tuple[str, int]]:
        """All (team, year) keys in the KenPom lookup."""
        return frozenset(self._kenpom_index.keys())

    def get_team(self, team: str, year: int, seed: int) -> TeamStats:
        """Look up a team's stats by name and year.

        Args:
            team: Team name (will be standardized).
            year: Tournament year.
            seed: Tournament seed (1-16).

        Returns:
            TeamStats with KenPom and optionally Barttorvik features.

        Raises:
            KeyError: If team/year combination is not found in KenPom data.
        """
        name = standardize_team_name(team.strip())
        kenpom = self._kenpom_index.get((name, year))
        if kenpom is None:
            raise KeyError(
                f"No KenPom data for team={name!r}, year={year}. "
                f"Check team name standardization."
            )

        bart = self._bart_index.get((name, year))
        if self.has_barttorvik and bart is None:
            logger.warning(
                "No Barttorvik data for team=%r, year=%d", name, year,
            )

        return TeamStats(
            team=name,
            year=year,
            seed=seed,
            features=dict(kenpom),
            barttorvik_features=dict(bart) if bart else None,
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
        """Build a single-row KenPom difference feature DataFrame.

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

    def build_combined_difference_features(
        self,
        team1: TeamStats,
        team2: TeamStats,
    ) -> pd.DataFrame:
        """Build a single-row combined KenPom + Barttorvik difference DataFrame.

        Computes 34 features: 21 KenPom diffs + 13 Barttorvik diffs.

        Args:
            team1: First team (must have barttorvik_features).
            team2: Second team (must have barttorvik_features).

        Returns:
            Single-row DataFrame with 34 float64 columns.

        Raises:
            ValueError: If either team is missing Barttorvik features.
        """
        kenpom_df = self.build_difference_features(team1, team2)

        if team1.barttorvik_features is None or team2.barttorvik_features is None:
            raise ValueError(
                "Combined features require Barttorvik data for both teams. "
                f"Team1 ({team1.team}): {'has' if team1.barttorvik_features else 'missing'}, "
                f"Team2 ({team2.team}): {'has' if team2.barttorvik_features else 'missing'}"
            )

        bart_row: dict[str, float] = {}
        t1_bart = team1.barttorvik_features
        t2_bart = team2.barttorvik_features

        # 12 Barttorvik stat differences
        for team1_col, _, diff_name in BARTTORVIK_STAT_PAIRS:
            bart_row[diff_name] = t1_bart[team1_col] - t2_bart[team1_col]

        # Derived: Barttorvik efficiency ratio diff
        ratio_t1 = t1_bart["Bart_AdjOE"] / t1_bart["Bart_AdjDE"]
        ratio_t2 = t2_bart["Bart_AdjOE"] / t2_bart["Bart_AdjDE"]
        bart_row["bart_efficiency_ratio_diff"] = ratio_t1 - ratio_t2

        bart_df = pd.DataFrame([bart_row])

        return pd.concat([kenpom_df, bart_df], axis=1)
