"""Merge restructured matchups with KenPom and Barttorvik statistics."""

import logging
from pathlib import Path

import pandas as pd

from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._features import standardize_team_name

logger = logging.getLogger(__name__)


def _merge_source_for_both_teams(
    matchups: pd.DataFrame,
    source_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """Merge a stats source for both Team1 and Team2.

    Args:
        matchups: Matchup DataFrame with Team1/Team2 and YEAR columns.
        source_df: Stats DataFrame with Team and Year columns.
        source_name: Name for logging (e.g. "KenPom", "Barttorvik").

    Returns:
        New DataFrame with source stats merged for both teams.
    """
    source_df = source_df.copy()
    source_df["Team"] = (
        source_df["Team"].apply(standardize_team_name).str.strip()
    )

    # Prefix source columns to avoid collision (except Team/Year used for join)
    team_col = f"{source_name}_Team" if source_name != "KenPom" else "Team"
    year_col = f"{source_name}_Year" if source_name != "KenPom" else "Year"

    # Rename Team/Year to source-prefixed names for non-KenPom sources
    if source_name != "KenPom":
        source_df = source_df.rename(
            columns={"Team": team_col, "Year": year_col},
        )

    # Merge for Team1
    result = matchups.merge(
        source_df,
        left_on=["Team1", "YEAR"],
        right_on=[team_col, year_col],
        how="left",
        suffixes=("", "_Team1"),
    )

    # Merge for Team2
    result = result.merge(
        source_df,
        left_on=["Team2", "YEAR"],
        right_on=[team_col, year_col],
        how="left",
        suffixes=("", "_Team2"),
    )

    # Check for unmatched teams
    stat_cols = [c for c in source_df.columns if c not in (team_col, year_col)]
    if stat_cols:
        first_stat = stat_cols[0]
        missing_t1 = result[first_stat].isna().sum()
        missing_t2 = result[f"{first_stat}_Team2"].isna().sum() if f"{first_stat}_Team2" in result.columns else 0
        if missing_t1 > 0 or missing_t2 > 0:
            logger.warning(
                "%s merge: %d Team1 and %d Team2 rows unmatched",
                source_name, missing_t1, missing_t2,
            )

    logger.info(
        "Merged %s stats: %d columns added per team",
        source_name, len(stat_cols),
    )
    return result


def merge_matchups_stats(
    kenpom_path: str | None = None,
    matchups_path: str | None = None,
    barttorvik_path: str | None = None,
    output_path: str | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Merge matchup data with KenPom and optionally Barttorvik statistics.

    Joins on team name and year, producing a dataset with features from both
    Team1 and Team2 ratings from each source.

    Args:
        kenpom_path: Path to KenPom CSV. Defaults to mm_config.MM_KENPOM_RAW.
        matchups_path: Path to restructured matchups. Defaults to mm_config.
        barttorvik_path: Path to Barttorvik CSV. None to skip. Defaults to
            mm_config.MM_BARTTORVIK_DATA if the file exists.
        output_path: Output path. Defaults to mm_config.MM_TRAINING_DATA.
        years: If provided, filter to only these years in the output.

    Returns:
        Merged DataFrame.
    """
    kenpom_path = kenpom_path or mm_config.MM_KENPOM_RAW
    matchups_path = matchups_path or mm_config.MM_MATCHUPS_RESTRUCTURED
    output_path = output_path or mm_config.MM_TRAINING_DATA

    # Auto-detect Barttorvik data if not explicitly specified
    if barttorvik_path is None:
        bart_file = Path(mm_config.MM_BARTTORVIK_DATA)
        if bart_file.exists():
            barttorvik_path = str(bart_file)

    kenpom_data = pd.read_csv(kenpom_path)
    matchups = pd.read_csv(matchups_path)
    logger.info(
        "Loaded %d KenPom rows and %d matchups",
        len(kenpom_data), len(matchups),
    )

    # Standardize matchup team names
    matchups["Team1"] = matchups["Team1"].apply(standardize_team_name).str.strip()
    matchups["Team2"] = matchups["Team2"].apply(standardize_team_name).str.strip()

    # Merge KenPom stats
    merged = _merge_source_for_both_teams(matchups, kenpom_data, "KenPom")

    # Merge Barttorvik stats if available
    if barttorvik_path:
        barttorvik_data = pd.read_csv(barttorvik_path)
        logger.info("Loaded %d Barttorvik rows", len(barttorvik_data))
        merged = _merge_source_for_both_teams(merged, barttorvik_data, "Bart")
    else:
        logger.info("No Barttorvik data — using KenPom features only")

    # Filter to specific years if requested
    if years:
        merged = merged[merged["YEAR"].isin(years)]
        logger.info("Filtered to years %s: %d rows", years, len(merged))

    # Save output
    mm_config.MM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info("Merged data saved to %s (%d rows)", output_path, len(merged))

    return merged


if __name__ == "__main__":
    merge_matchups_stats()
