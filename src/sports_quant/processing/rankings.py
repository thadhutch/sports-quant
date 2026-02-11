"""Convert numerical PFF features into rankings."""

import logging

import numpy as np
import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def compute_rankings():
    # Read the CSV file
    df = pd.read_csv(config.OVERUNDER_GP)

    # Convert 'Formatted Date' to datetime for proper sorting
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'])

    # Sort the DataFrame by 'Formatted Date' to process the games chronologically
    df = df.sort_values('Formatted Date').reset_index(drop=True)

    # Get the list of numerical columns excluding 'season', 'total', 'ou_line'
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['season', 'total', 'ou_line']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

    # Identify features associated with teams (columns starting with 'home-' or 'away-')
    feature_names = set()
    for col in numerical_cols:
        if col.startswith('home-'):
            feature_name = col[len('home-'):]
            feature_names.add(feature_name)
        elif col.startswith('away-'):
            feature_name = col[len('away-'):]
            feature_names.add(feature_name)

    # Initialize dictionaries to keep track of the latest feature values for each team
    feature_values = {}  # feature_values[feature_name][team] = latest value
    for feature_name in feature_names:
        feature_values[feature_name] = {}

        # Create new columns in the DataFrame for ranks
        df[f'home-{feature_name}-rank'] = np.nan
        df[f'away-{feature_name}-rank'] = np.nan

    # Process the DataFrame per date
    dates = df['Formatted Date'].unique()
    for date in dates:
        # Get the subset of the DataFrame for the current date
        df_date = df[df['Formatted Date'] == date]

        # **First, update the latest feature values for each team with the values from the games on that date**
        for idx in df_date.index:
            row = df.loc[idx]
            home_team = row['home_team']
            away_team = row['away_team']

            for feature_name in feature_names:
                home_feature_col = f'home-{feature_name}'
                away_feature_col = f'away-{feature_name}'

                home_feature_value = row[home_feature_col]
                away_feature_value = row[away_feature_col]

                if not np.isnan(home_feature_value):
                    feature_values[feature_name][home_team] = home_feature_value
                if not np.isnan(away_feature_value):
                    feature_values[feature_name][away_team] = away_feature_value

        # **Now, compute the rankings using the updated feature values**
        for feature_name in feature_names:
            # Get the latest feature values for all teams from feature_values
            team_feature = feature_values[feature_name]  # {team: feature_value}

            # Create a DataFrame of teams and their feature values
            teams = list(team_feature.keys())
            feature_values_list = list(team_feature.values())
            teams_feature_df = pd.DataFrame({'team': teams, 'feature_value': feature_values_list})

            # Drop teams with missing or zero feature values
            teams_feature_df = teams_feature_df.dropna(subset=['feature_value'])
            teams_feature_df = teams_feature_df[teams_feature_df['feature_value'] != 0]

            if teams_feature_df.empty:
                # No teams have valid feature values yet; skip ranking
                continue

            # Compute rankings (1 = highest value)
            teams_feature_df['rank'] = teams_feature_df['feature_value'].rank(ascending=False, method='min')

            # Create a mapping from team to rank
            team_to_rank = dict(zip(teams_feature_df['team'], teams_feature_df['rank']))

            # For each game on this date, assign ranks to home and away teams
            indices = df_date.index
            for idx in indices:
                row = df.loc[idx]
                home_team = row['home_team']
                away_team = row['away_team']

                # Get the rank of the home team and away team
                home_rank = team_to_rank.get(home_team, np.nan)
                away_rank = team_to_rank.get(away_team, np.nan)

                # Update the DataFrame with the ranks
                df.at[idx, f'home-{feature_name}-rank'] = home_rank
                df.at[idx, f'away-{feature_name}-rank'] = away_rank

    # Save the modified DataFrame to a new CSV file
    df.to_csv(config.OVERUNDER_RANKED, index=False)

    logger.info("Rankings saved to %s", config.OVERUNDER_RANKED)


if __name__ == "__main__":
    compute_rankings()
