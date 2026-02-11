"""Normalize Pro Football Reference game titles to formatted dates."""

import logging

import pandas as pd
from dateutil.parser import parse

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def extract_date(title: str) -> str:
    """Extract and format the date from a PFR game title."""
    # Find the date portion in the title (after the second hyphen)
    date_str = title.split('-')[-1].strip()
    # Parse the date and format it as 'MM/DD/YYYY'
    parsed_date = parse(date_str).strftime('%m/%d/%Y')
    return parsed_date


def normalize_pfr_dates():
    # Load the CSV file into a DataFrame
    df = pd.read_csv(config.PFR_GAME_DATA_FILE)

    # Apply the function to extract dates from the Title column
    df['Formatted Date'] = df['Title'].apply(extract_date)

    logger.info("Extracted dates for %d games", len(df))

    df.to_csv(config.PFR_NORMALIZED_FILE, index=False)
    logger.info("Saved to %s", config.PFR_NORMALIZED_FILE)


if __name__ == "__main__":
    normalize_pfr_dates()
