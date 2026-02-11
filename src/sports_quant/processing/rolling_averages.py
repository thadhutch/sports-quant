"""Calculate rolling averages for PFF statistics."""

import logging

import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)

# List of stat columns
stat_columns = [
    "off",
    "pass",
    "pblk",
    "recv",
    "run",
    "rblk",
    "def",
    "rdef",
    "tack",
    "prsh",
    "cov",
]


def initialize_team_stats() -> dict:
    """Initialize zeroed stat counters for a team."""
    return {stat: 0.0 for stat in stat_columns}


def calculate_avg_stats(cumulative_stats: dict, game_count: int) -> dict:
    """Calculate average stats from cumulative totals."""
    if game_count > 0:
        return {stat: cumulative_stats[stat] / game_count for stat in cumulative_stats}
    else:
        return {stat: 0.0 for stat in cumulative_stats}


def compute_rolling_averages():
    # Read the CSV file into a DataFrame
    df = pd.read_csv(config.OVERUNDER_RAW)

    # Convert the 'Formatted Date' to a datetime object to sort the games chronologically
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], format="%m/%d/%Y")

    # Sort the DataFrame by date
    df = df.sort_values("Formatted Date")

    # Initialize team_stats as an empty dictionary
    team_stats = {}

    # Iterate over each game
    for index, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        game_season = row["season"]

        # Process home team
        if home_team not in team_stats:
            # Initialize team stats
            team_stats[home_team] = {
                "stats": initialize_team_stats(),
                "count": 0,
                "season": game_season,
            }
        else:
            # Check if season has changed
            if team_stats[home_team]["season"] != game_season:
                # Reset stats and count
                team_stats[home_team]["stats"] = initialize_team_stats()
                team_stats[home_team]["count"] = 0
                team_stats[home_team]["season"] = game_season

        # Process away team
        if away_team not in team_stats:
            team_stats[away_team] = {
                "stats": initialize_team_stats(),
                "count": 0,
                "season": game_season,
            }
        else:
            if team_stats[away_team]["season"] != game_season:
                team_stats[away_team]["stats"] = initialize_team_stats()
                team_stats[away_team]["count"] = 0
                team_stats[away_team]["season"] = game_season

        # Calculate average stats before the current game
        avg_home_stats = calculate_avg_stats(
            team_stats[home_team]["stats"], team_stats[home_team]["count"]
        )
        avg_away_stats = calculate_avg_stats(
            team_stats[away_team]["stats"], team_stats[away_team]["count"]
        )

        # Store average stats in DataFrame (create new columns)
        for stat in stat_columns:
            df.at[index, f"home-{stat}-avg"] = avg_home_stats[stat]
            df.at[index, f"away-{stat}-avg"] = avg_away_stats[stat]

        # Update cumulative stats with current game's stats
        for stat in stat_columns:
            # For home team
            team_stats[home_team]["stats"][stat] += row[f"home-{stat}"]
            # For away team
            team_stats[away_team]["stats"][stat] += row[f"away-{stat}"]

        # Increment game counts
        team_stats[home_team]["count"] += 1
        team_stats[away_team]["count"] += 1

    df.drop(
        columns=[
            "home-off",
            "home-pass",
            "home-pblk",
            "home-recv",
            "home-run",
            "home-rblk",
            "away-off",
            "away-pass",
            "away-pblk",
            "away-recv",
            "away-run",
            "away-rblk",
            "home-def",
            "home-rdef",
            "home-tack",
            "home-prsh",
            "home-cov",
            "away-def",
            "away-rdef",
            "away-tack",
            "away-prsh",
            "away-cov",
        ],
        inplace=True,
    )
    # Save the modified DataFrame to a new CSV if needed
    df.to_csv(config.OVERUNDER_AVERAGES, index=False)
    logger.info("Rolling averages saved to %s", config.OVERUNDER_AVERAGES)


if __name__ == "__main__":
    compute_rolling_averages()
