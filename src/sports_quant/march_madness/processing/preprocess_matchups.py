"""Preprocess raw matchup data by pairing alternating rows into Team1 vs Team2."""

import logging

import pandas as pd

from sports_quant.march_madness import _config as mm_config

logger = logging.getLogger(__name__)


def preprocess_matchups(
    input_path: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Pair raw matchup rows into structured Team1 vs Team2 matchups.

    The raw matchups CSV has alternating rows: Team1, Team2, Team1, Team2...
    This function pairs them and adds a Team1_Win label.

    Args:
        input_path: Path to raw matchups.csv. Defaults to mm_config.
        output_path: Path for restructured output. Defaults to mm_config.

    Returns:
        DataFrame with paired matchups and Team1_Win label.
    """
    input_path = input_path or mm_config.MM_MATCHUPS_RAW
    output_path = output_path or mm_config.MM_MATCHUPS_RESTRUCTURED

    data = pd.read_csv(input_path)
    logger.info("Loaded %d raw matchup rows from %s", len(data), input_path)

    # Split into team1 (even rows) and team2 (odd rows)
    team1s = data.iloc[::2].reset_index(drop=True)
    team2s = data.iloc[1::2].reset_index(drop=True)

    # Rename columns to distinguish teams
    team1s.rename(
        columns={
            "TEAM NO": "Team1_NO",
            "TEAM": "Team1",
            "SEED": "Seed1",
            "SCORE": "Score1",
        },
        inplace=True,
    )
    team2s.rename(
        columns={
            "TEAM NO": "Team2_NO",
            "TEAM": "Team2",
            "SEED": "Seed2",
            "SCORE": "Score2",
        },
        inplace=True,
    )

    # Drop duplicate columns from team2s
    team2s.drop(["YEAR", "BY YEAR NO", "BY ROUND NO"], axis=1, inplace=True)

    # Merge on index
    matchups = pd.concat([team1s, team2s], axis=1)
    matchups["Team1_Win"] = (matchups["Score1"] > matchups["Score2"]).astype(int)

    # Save
    mm_config.MM_MATCHUPS_DIR.mkdir(parents=True, exist_ok=True)
    matchups.to_csv(output_path, index=False)
    logger.info("Restructured %d matchups saved to %s", len(matchups), output_path)

    return matchups


if __name__ == "__main__":
    preprocess_matchups()
