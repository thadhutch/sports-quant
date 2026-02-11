"""Merge PFF and PFR datasets on date and team columns."""

import logging

import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def merge_datasets():
    # Load the first CSV with game details
    df1 = pd.read_csv(config.PFR_FINAL_FILE)

    # Load the second CSV with game statistics
    df2 = pd.read_csv(config.PFF_NORMALIZED_FILE)

    # Merge the dataframes on 'Formatted Date', 'away_team', and 'home_team'
    merged_df = pd.merge(
        df1,
        df2,
        left_on=['Formatted Date', 'away_team', 'home_team'],
        right_on=['date', 'team_0', 'team_1'],
        how='inner'  # Change to 'outer' if you want to include non-matching rows as well
    )

    # Drop any columns that are redundant after merging, if necessary
    merged_df.drop(columns=['team_0', 'team_1', 'Title', 'game-string', 'date'], inplace=True)

    logger.info("Merged %d rows", len(merged_df))

    # Save the merged dataframe to a CSV file if needed
    merged_df.to_csv(config.MERGED_FILE, index=False)
    logger.info("Merged data saved to %s", config.MERGED_FILE)


if __name__ == "__main__":
    merge_datasets()
