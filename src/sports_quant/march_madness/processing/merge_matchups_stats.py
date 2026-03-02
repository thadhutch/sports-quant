"""Merge restructured matchups with KenPom statistics."""

import logging

import pandas as pd

from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._features import standardize_team_name

logger = logging.getLogger(__name__)


def merge_matchups_stats(
    kenpom_path: str | None = None,
    matchups_path: str | None = None,
    output_path: str | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Merge matchup data with KenPom statistics for both teams.

    Joins on team name and year, producing a dataset with features from both
    Team1 and Team2 KenPom ratings.

    Args:
        kenpom_path: Path to KenPom CSV. Defaults to mm_config.MM_KENPOM_RAW.
        matchups_path: Path to restructured matchups. Defaults to mm_config.
        output_path: Output path. Defaults to mm_config.MM_TRAINING_DATA.
        years: If provided, filter to only these years in the output.

    Returns:
        Merged DataFrame.
    """
    kenpom_path = kenpom_path or mm_config.MM_KENPOM_RAW
    matchups_path = matchups_path or mm_config.MM_MATCHUPS_RESTRUCTURED
    output_path = output_path or mm_config.MM_TRAINING_DATA

    kenpom_data = pd.read_csv(kenpom_path)
    matchups = pd.read_csv(matchups_path)
    logger.info(
        "Loaded %d KenPom rows and %d matchups",
        len(kenpom_data), len(matchups),
    )

    # Standardize team names
    kenpom_data["Team"] = kenpom_data["Team"].apply(standardize_team_name).str.strip()
    matchups["Team1"] = matchups["Team1"].apply(standardize_team_name).str.strip()
    matchups["Team2"] = matchups["Team2"].apply(standardize_team_name).str.strip()

    # Merge KenPom stats for Team1
    matchups = matchups.merge(
        kenpom_data,
        left_on=["Team1", "YEAR"],
        right_on=["Team", "Year"],
        how="left",
        suffixes=("", "_Team1"),
    )

    # Merge KenPom stats for Team2
    matchups = matchups.merge(
        kenpom_data,
        left_on=["Team2", "YEAR"],
        right_on=["Team", "Year"],
        how="left",
        suffixes=("", "_Team2"),
    )

    # Filter to specific years if requested
    if years:
        matchups = matchups[matchups["YEAR"].isin(years)]
        logger.info("Filtered to years %s: %d rows", years, len(matchups))

    # Save output
    mm_config.MM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    matchups.to_csv(output_path, index=False)
    logger.info("Merged data saved to %s (%d rows)", output_path, len(matchups))

    return matchups


if __name__ == "__main__":
    merge_matchups_stats()
