"""Calculate games played for each team before each game."""

import logging

import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def add_games_played():
    # Read the CSV file
    df = pd.read_csv(config.OVERUNDER_AVERAGES)

    # Convert 'Formatted Date' to datetime for proper sorting
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'])

    # Sort the DataFrame by 'season' and 'Formatted Date' to ensure chronological order
    df = df.sort_values(by=['season', 'Formatted Date']).reset_index(drop=True)

    # Initialize dictionaries to keep track of games played per team per season
    # Structure: {season: {team: games_played}}
    games_played = {}

    # Lists to store the games played for each row
    home_gp_list = []
    away_gp_list = []

    # Iterate through each row to calculate 'home_gp' and 'away_gp'
    for index, row in df.iterrows():
        season = row['season']
        home_team = row['home_team']
        away_team = row['away_team']

        # Initialize season in the dictionary if not present
        if season not in games_played:
            games_played[season] = {}

        # Initialize teams in the season if not present
        if home_team not in games_played[season]:
            games_played[season][home_team] = 0
        if away_team not in games_played[season]:
            games_played[season][away_team] = 0

        # Append current games played before this game
        home_gp_list.append(games_played[season][home_team])
        away_gp_list.append(games_played[season][away_team])

        # Increment the games played for both teams
        games_played[season][home_team] += 1
        games_played[season][away_team] += 1

    # Add the new columns to the DataFrame
    df['home_gp'] = home_gp_list
    df['away_gp'] = away_gp_list

    # Reorder the columns to place the new columns next to related columns
    cols = df.columns.tolist()
    home_team_idx = cols.index('home_team')
    away_team_idx = cols.index('away_team')
    # Insert 'home_gp' after 'home_team' and 'away_gp' after 'away_team'
    cols.insert(home_team_idx + 1, cols.pop(cols.index('home_gp')))
    cols.insert(away_team_idx + 2, cols.pop(cols.index('away_gp')))
    df = df[cols]

    # Save the modified DataFrame to a new CSV file
    df.to_csv(config.OVERUNDER_GP, index=False)

    logger.info("Games played columns added and saved to %s", config.OVERUNDER_GP)


if __name__ == "__main__":
    add_games_played()
