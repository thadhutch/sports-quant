"""Extract away and home team names from PFR game titles."""

import logging

import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def extract_teams(title: str) -> tuple:
    """Parse a PFR title to extract away and home team names."""
    # Split the title by ' at ' to separate away and home teams
    teams = title.split(' at ')
    away_team = teams[0].strip().replace('"', '')  # Remove leading/trailing spaces and quotes
    home_team = teams[1].split(' - ')[0].strip()  # Extract home team before the dash and date
    return away_team, home_team


def extract_pfr_teams():
    # Load the CSV file into a DataFrame
    df = pd.read_csv(config.PFR_NORMALIZED_FILE)

    # Apply the function to extract teams and create new columns
    df[['away_team', 'home_team']] = df['Title'].apply(lambda x: pd.Series(extract_teams(x)))

    logger.info("Extracted teams for %d games", len(df))

    # Save the updated DataFrame if needed
    df.to_csv(config.PFR_FINAL_FILE, index=False)
    logger.info("Saved to %s", config.PFR_FINAL_FILE)


if __name__ == "__main__":
    extract_pfr_teams()
