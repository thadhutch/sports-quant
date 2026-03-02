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

# Team name normalization mapping
TEAM_NAME_MAPPING: dict[str, str] = {
    "N.C. State": "North Carolina St.",
    "Louisiana Lafayette": "Louisiana",
    "College of Charleston": "Charleston",
}


def standardize_team_name(name: str) -> str:
    """Apply team name normalization mapping."""
    return TEAM_NAME_MAPPING.get(name, name)
