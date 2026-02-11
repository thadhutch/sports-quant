"""Extract Over/Under betting information from the merged dataset."""

import logging

import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def set_total(over_under: str) -> int | None:
    """Determine the total value (1=over, 0=under) from the Over/Under string."""
    if '(under)' in over_under:
        return 0
    elif '(over)' in over_under:
        return 1
    return None  # In case the value doesn't match either condition


def extract_ou_line(over_under: str) -> float | None:
    """Extract the numeric betting line from the Over/Under string."""
    # Split the string and try to convert the first part to a float
    try:
        return float(over_under.split()[0])
    except ValueError:
        return None  # Return None if conversion fails


def process_over_under():
    # Load the dataset
    df = pd.read_csv(config.MERGED_FILE)

    # Apply the functions to create the 'total' and 'ou_line' columns
    df['total'] = df['Over/Under'].apply(set_total)
    # Replace NaN values in the 'total' column with 2
    df['total'].fillna(2, inplace=True)

    # Count the number of 2 values in the 'total' column
    count_twos = (df['total'] == 2).sum()
    logger.info("Number of pushes in the 'total' column: %d", count_twos)

    df['ou_line'] = df['Over/Under'].apply(extract_ou_line)

    df.drop(columns=['Over/Under'], inplace=True)  # Drop the original column

    # Save the updated DataFrame back to the CSV file if needed
    df.to_csv(config.OVERUNDER_RAW, index=False)

    logger.info("Over/Under columns created and saved to %s", config.OVERUNDER_RAW)


if __name__ == "__main__":
    process_over_under()
