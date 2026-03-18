"""Feature definitions and team name mappings for March Madness."""

# Columns to drop before training (metadata / non-feature columns)
DROP_COLUMNS: list[str] = [
    "BY YEAR NO",
    "BY ROUND NO",
    "Team1_NO",
    "Team2_NO",
    "Team",
    "Team_Team2",
    "Year",
    "Year_Team2",
    "Score1",
    "Score2",
    "ROUND",
    "ROUND.1",
    "CURRENT ROUND.1",
    "W-L",
    "W-L_Team2",
    "Team1",
    "Team2",
    "Conference",
    "Conference_Team2",
    "Seed1",
    "Seed2",
    "YEAR",
    "CURRENT ROUND",
    # Barttorvik metadata columns
    "Bart_Team",
    "Bart_Team_Team2",
    "Bart_Year",
    "Bart_Year_Team2",
]

# Columns preserved for team identification before dropping
TEAM_INFO_COLUMNS: list[str] = [
    "YEAR",
    "Team1",
    "Seed1",
    "Team2",
    "Seed2",
    "Team1_Win",
]

TARGET_COLUMN = "Team1_Win"
YEAR_COLUMN = "YEAR"

# ---------------------------------------------------------------------------
# Barttorvik column definitions
# ---------------------------------------------------------------------------

# Raw column names as they appear in the Barttorvik CSV download.
# Order matters — must align with BARTTORVIK_COLUMNS below.
BARTTORVIK_RAW_COLUMNS: tuple[str, ...] = (
    "team",
    "rank",
    "adjoe",
    "adjde",
    "barthag",
    "adjt",
    "sos",
    "ncsos",
    "elite SOS",
    "WAB",
    "Qual O",
    "Qual D",
    "Qual Barthag",
)

# Standardized column names used internally after download.
BARTTORVIK_COLUMNS: tuple[str, ...] = (
    "Team",
    "Bart_Rank",
    "Bart_AdjOE",
    "Bart_AdjDE",
    "Bart_Barthag",
    "Bart_AdjT",
    "Bart_SOS",
    "Bart_NCSOS",
    "Bart_EliteSOS",
    "Bart_WAB",
    "Bart_QualO",
    "Bart_QualD",
    "Bart_QualBarthag",
)

# Stat columns used for feature engineering (excludes Team and Year).
BARTTORVIK_STAT_COLUMNS: tuple[str, ...] = (
    "Bart_Rank",
    "Bart_AdjOE",
    "Bart_AdjDE",
    "Bart_Barthag",
    "Bart_AdjT",
    "Bart_SOS",
    "Bart_NCSOS",
    "Bart_EliteSOS",
    "Bart_WAB",
    "Bart_QualO",
    "Bart_QualD",
    "Bart_QualBarthag",
)

# KenPom column names after scraping
KENPOM_COLUMNS: list[str] = [
    "Rank",
    "Team",
    "Conference",
    "W-L",
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
    "Year",
]

# ---------------------------------------------------------------------------
# Difference feature definitions
# ---------------------------------------------------------------------------

# Mapping of (team1_col, team2_col, diff_name) for pairwise difference features.
# Column names match the merged training_data.csv (Team2 cols use _Team2 suffix).
STAT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("Rank", "Rank_Team2", "rank_diff"),
    ("AdjEM", "AdjEM_Team2", "adjEM_diff"),
    ("AdjustO", "AdjustO_Team2", "adjO_diff"),
    ("AdjustO Rank", "AdjustO Rank_Team2", "adjO_rank_diff"),
    ("AdjustD", "AdjustD_Team2", "adjD_diff"),
    ("AdjustD Rank", "AdjustD Rank_Team2", "adjD_rank_diff"),
    ("AdjustT", "AdjustT_Team2", "adjT_diff"),
    ("AdjustT Rank", "AdjustT Rank_Team2", "adjT_rank_diff"),
    ("Luck", "Luck_Team2", "luck_diff"),
    ("Luck Rank", "Luck Rank_Team2", "luck_rank_diff"),
    ("SOS AdjEM", "SOS AdjEM_Team2", "sos_adjEM_diff"),
    ("SOS AdjEM Rank", "SOS AdjEM Rank_Team2", "sos_adjEM_rank_diff"),
    ("SOS OppO", "SOS OppO_Team2", "sos_oppO_diff"),
    ("SOS OppO Rank", "SOS OppO Rank_Team2", "sos_oppO_rank_diff"),
    ("SOS OppD", "SOS OppD_Team2", "sos_oppD_diff"),
    ("SOS OppD Rank", "SOS OppD Rank_Team2", "sos_oppD_rank_diff"),
    ("NCSOS AdjEM", "NCSOS AdjEM_Team2", "ncsos_adjEM_diff"),
    ("NCSOS AdjEM Rank", "NCSOS AdjEM Rank_Team2", "ncsos_adjEM_rank_diff"),
)

SEED_DIFF_COLUMN = "seed_diff"

# Derived features computed from differences
DERIVED_FEATURES: tuple[str, ...] = (
    "efficiency_ratio_diff",
    "seed_x_adjEM_interaction",
)

# Complete ordered list of KenPom-only difference feature columns (21 total)
DIFF_FEATURE_COLUMNS: tuple[str, ...] = (
    *(diff_name for _, _, diff_name in STAT_PAIRS),
    SEED_DIFF_COLUMN,
    *DERIVED_FEATURES,
)

# ---------------------------------------------------------------------------
# Barttorvik difference feature definitions
# ---------------------------------------------------------------------------

# Pairwise difference features for Barttorvik stats.
# Maps (team1_col, team2_col, diff_name) — same pattern as STAT_PAIRS.
BARTTORVIK_STAT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("Bart_Rank", "Bart_Rank_Team2", "bart_rank_diff"),
    ("Bart_AdjOE", "Bart_AdjOE_Team2", "bart_adjOE_diff"),
    ("Bart_AdjDE", "Bart_AdjDE_Team2", "bart_adjDE_diff"),
    ("Bart_Barthag", "Bart_Barthag_Team2", "bart_barthag_diff"),
    ("Bart_AdjT", "Bart_AdjT_Team2", "bart_adjT_diff"),
    ("Bart_SOS", "Bart_SOS_Team2", "bart_sos_diff"),
    ("Bart_NCSOS", "Bart_NCSOS_Team2", "bart_ncsos_diff"),
    ("Bart_EliteSOS", "Bart_EliteSOS_Team2", "bart_elite_sos_diff"),
    ("Bart_WAB", "Bart_WAB_Team2", "bart_wab_diff"),
    ("Bart_QualO", "Bart_QualO_Team2", "bart_qualO_diff"),
    ("Bart_QualD", "Bart_QualD_Team2", "bart_qualD_diff"),
    ("Bart_QualBarthag", "Bart_QualBarthag_Team2", "bart_qual_barthag_diff"),
)

# Barttorvik-derived features
BARTTORVIK_DERIVED_FEATURES: tuple[str, ...] = (
    "bart_efficiency_ratio_diff",
)

# Barttorvik-only difference feature columns (13 total)
BARTTORVIK_DIFF_FEATURE_COLUMNS: tuple[str, ...] = (
    *(diff_name for _, _, diff_name in BARTTORVIK_STAT_PAIRS),
    *BARTTORVIK_DERIVED_FEATURES,
)

# ---------------------------------------------------------------------------
# Matchup-specific interaction features
# ---------------------------------------------------------------------------

# Historical seed-pairing upset rates (lower seed winning), 2003-2024.
# Keys are (higher_seed, lower_seed) — e.g. (1, 16) means 1-seed vs 16-seed.
# Values are the historical probability that the LOWER seed wins (the upset).
SEED_MATCHUP_PRIORS: dict[tuple[int, int], float] = {
    (1, 16): 0.01,
    (2, 15): 0.06,
    (3, 14): 0.15,
    (4, 13): 0.20,
    (5, 12): 0.35,
    (6, 11): 0.37,
    (7, 10): 0.39,
    (8, 9): 0.48,
}


def seed_upset_prior(seed1: int, seed2: int) -> float:
    """Return a centered upset prior from Team1's perspective.

    Positive means Team1 is historically favored, negative means
    Team1 is the historical underdog. Zero when no prior exists.

    The value is ``team1_win_prob - 0.5``, so it is odd under swap:
    ``seed_upset_prior(a, b) == -seed_upset_prior(b, a)``.
    """
    if seed1 == seed2:
        return 0.0

    higher_seed = min(seed1, seed2)  # lower number = higher seed
    lower_seed = max(seed1, seed2)
    upset_rate = SEED_MATCHUP_PRIORS.get((higher_seed, lower_seed), 0.50)

    # P(Team1 wins) from Team1's perspective
    if seed1 <= seed2:
        # Team1 is the higher seed (favorite)
        team1_win_prob = 1.0 - upset_rate
    else:
        # Team1 is the lower seed (underdog)
        team1_win_prob = upset_rate

    return team1_win_prob - 0.5


# Matchup interaction feature names (11 total)
MATCHUP_FEATURE_COLUMNS: tuple[str, ...] = (
    # Group 1: Offensive vs defensive style interactions
    "offense_vs_defense_mismatch",
    "bart_offense_vs_defense_mismatch",
    "offense_defense_product",
    "bart_offense_defense_product",
    # Group 2: Tempo mismatch
    "tempo_mismatch_magnitude",
    "tempo_x_quality_interaction",
    "tempo_x_seed_interaction",
    # Group 3: Historical seed priors
    "seed_upset_prior_centered",
    "seed_x_quality_gap",
    # Group 4: Quality consistency
    "quality_source_agreement",
    "sos_quality_interaction",
)

# Features with even symmetry (invariant under team swap).
# These must NOT be negated during symmetrization — they are either
# absolute values or products of two odd terms.
MATCHUP_EVEN_FEATURES: frozenset[str] = frozenset({
    "offense_defense_product",
    "bart_offense_defense_product",
    "tempo_mismatch_magnitude",
    "tempo_x_quality_interaction",
    "tempo_x_seed_interaction",
    "seed_x_quality_gap",
    "quality_source_agreement",
    "sos_quality_interaction",
})

# Combined KenPom + Barttorvik + matchup features (45 total)
COMBINED_DIFF_FEATURE_COLUMNS: tuple[str, ...] = (
    *DIFF_FEATURE_COLUMNS,
    *BARTTORVIK_DIFF_FEATURE_COLUMNS,
    *MATCHUP_FEATURE_COLUMNS,
)

# Team name normalization mapping.
# Maps variant names (from KenPom historical data, matchups, etc.) to a
# single canonical form used throughout the pipeline.
TEAM_NAME_MAPPING: dict[str, str] = {
    # Existing mappings
    "N.C. State": "North Carolina St.",
    "Louisiana Lafayette": "Louisiana",
    "College of Charleston": "Charleston",
    # Historical KenPom (2011-2016) uses full "State" instead of "St."
    "Michigan State": "Michigan St.",
    "Ohio State": "Ohio St.",
    "Wichita State": "Wichita St.",
    "Iowa State": "Iowa St.",
    "Kansas State": "Kansas St.",
    "Florida State": "Florida St.",
    "San Diego State": "San Diego St.",
    "Colorado State": "Colorado St.",
    "North Carolina State": "North Carolina St.",
    "Murray State": "Murray St.",
    "Oregon State": "Oregon St.",
    "Oklahoma State": "Oklahoma St.",
    "Arizona State": "Arizona St.",
    "Weber State": "Weber St.",
    "Norfolk State": "Norfolk St.",
    "Fresno State": "Fresno St.",
    "Georgia State": "Georgia St.",
    "Long Beach State": "Long Beach St.",
    "New Mexico State": "New Mexico St.",
    "North Dakota State": "North Dakota St.",
    "South Dakota State": "South Dakota St.",
    "Northwestern State": "Northwestern St.",
    "Cal State Bakersfield": "Cal St. Bakersfield",
    # Name variant differences
    "Ole Miss": "Mississippi",
    "Miami (FL)": "Miami FL",
    "Loyola (MD)": "Loyola MD",
    "UConn": "Connecticut",
    "Southern Methodist": "SMU",
    "Long Island": "LIU Brooklyn",
    "St. Peter's": "Saint Peter's",
    "Louisiana-Lafayette": "Louisiana Lafayette",
    "Texas San Antonio": "UTSA",
    "California-Irvine": "UC Irvine",
    "NC Central": "North Carolina Central",
    # 2011 christoukmaji-specific variants
    "Texas A&M;": "Texas A&M",
    "Brigham Young": "BYU",
    "Virginia Commonwealth": "VCU",
    "Nevada Las Vegas": "UNLV",
    # Barttorvik uses "St." suffix where matchup data does not
    "McNeese St.": "McNeese",
    # Barttorvik/matchup name variants
    "SIU Edwardsville": "SIUE",
    "LIU": "LIU Brooklyn",
    "Detroit Mercy": "Detroit",
    "Arkansas Little Rock": "Little Rock",
}


def standardize_team_name(name: str) -> str:
    """Apply team name normalization mapping."""
    return TEAM_NAME_MAPPING.get(name, name)
