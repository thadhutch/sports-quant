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

# Complete ordered list of difference feature columns (21 total)
DIFF_FEATURE_COLUMNS: tuple[str, ...] = (
    *(diff_name for _, _, diff_name in STAT_PAIRS),
    SEED_DIFF_COLUMN,
    *DERIVED_FEATURES,
)

# Team name normalization mapping
TEAM_NAME_MAPPING: dict[str, str] = {
    "N.C. State": "North Carolina St.",
    "Louisiana Lafayette": "Louisiana",
    "College of Charleston": "Charleston",
    "Mount St. Mary's MAAC": "Mount St. Mary's",
}


def standardize_team_name(name: str) -> str:
    """Apply team name normalization mapping."""
    return TEAM_NAME_MAPPING.get(name, name)
