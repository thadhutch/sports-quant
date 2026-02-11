"""Extract dates and seasons from PFF game strings."""

import logging

import pandas as pd
from dateutil.parser import parse

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def extract_date_and_season(game_str: str) -> tuple:
    """Parse a game string to extract the formatted date and season year."""
    parts = game_str.split('-')
    if len(parts) > 2:
        date_str = parts[-1].strip()
        try:
            date = parse(date_str)
            # Extract the original year for the season before any adjustments
            season_year = date.year
            # Adjust the year if the month is January or February
            if date.month in [1, 2]:
                date = date.replace(year=date.year + 1)
            formatted_date = date.strftime('%m/%d/%Y')
            return formatted_date, season_year
        except ValueError:
            return None, None
    return None, None


def extract_dates():
    # Load the CSV data into a pandas DataFrame
    data = pd.read_csv(config.PFF_RAW_FILE)

    # Rename 'Unnamed: 0' to 'game-string' and 'game' to 'index'
    data.rename(columns={'Unnamed: 0': 'game-string', 'game': 'index'}, inplace=True)

    # Apply the function to create date and season columns
    data[['date', 'season']] = data['game-string'].apply(lambda x: pd.Series(extract_date_and_season(x)))

    # Save the modified DataFrame to a new CSV file
    data.to_csv(config.PFF_DATES_FILE, index=False)

    logger.info("Date and season columns added and saved to %s", config.PFF_DATES_FILE)


if __name__ == "__main__":
    extract_dates()
