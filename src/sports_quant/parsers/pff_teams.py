"""Normalize PFF team abbreviations to full names."""

import logging

import pandas as pd

from sports_quant.teams import encoded_teams
from sports_quant import _config as config

logger = logging.getLogger(__name__)

# Create a reverse mapping of encoded_teams dictionary
reverse_encoded_teams = {v: k for k, v in encoded_teams.items()}


def map_teams(game_string: str) -> tuple:
    """Map team abbreviations from a game string to full team names."""
    # Split the game string to get team abbreviations
    teams = game_string.split('-')[:2]
    # Map abbreviations to full team names
    team_0 = reverse_encoded_teams.get(teams[0], teams[0])  # Default to abbreviation if not found
    team_1 = reverse_encoded_teams.get(teams[1], teams[1])  # Default to abbreviation if not found
    return team_0, team_1


def normalize_pff_teams():
    # Load the dataset
    df = pd.read_csv(config.PFF_DATES_FILE)

    # Apply the function and create team_0 and team_1 columns
    df[['team_0', 'team_1']] = df['game-string'].apply(lambda x: pd.Series(map_teams(x)))

    logger.info("Normalized %d rows", len(df))

    # Save the updated DataFrame if needed
    df.to_csv(config.PFF_NORMALIZED_FILE, index=False)
    logger.info("Saved to %s", config.PFF_NORMALIZED_FILE)


if __name__ == "__main__":
    normalize_pff_teams()
