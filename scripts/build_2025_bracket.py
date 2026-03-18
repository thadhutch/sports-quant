"""Build 2025 NCAA tournament bracket data and merge into training data.

This script:
1. Populates 2025 matchup results (all 63 games) into restructured_matchups.csv
2. Updates kenpom_data.csv to include 2025 KenPom data
3. Re-runs the merge to produce updated training_data.csv
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/march-madness")
OLD_REPO = Path.home() / "Documents/GitHub/march-madness-2025/data"


def build_2025_matchups() -> pd.DataFrame:
    """Build correct 2025 tournament matchup data from actual results."""
    # All 63 actual 2025 NCAA tournament games
    # Format: (Seed1, Team1, Score1, Seed2, Team2, Score2)
    games = [
        # === FIRST ROUND (32 games) ===
        # East Region
        (1, "Duke", 93, 16, "Mount St. Mary's", 49),
        (8, "Mississippi St.", 72, 9, "Baylor", 75),
        (5, "Oregon", 81, 12, "Liberty", 52),
        (4, "Arizona", 93, 13, "Akron", 65),
        (6, "BYU", 80, 11, "VCU", 71),
        (3, "Wisconsin", 85, 14, "Montana", 66),
        (7, "Saint Mary's", 59, 10, "Vanderbilt", 56),
        (2, "Alabama", 90, 15, "Robert Morris", 81),
        # Midwest Region
        (1, "Houston", 78, 16, "SIUE", 40),
        (8, "Gonzaga", 89, 9, "Georgia", 68),
        (5, "Clemson", 67, 12, "McNeese", 69),
        (4, "Purdue", 75, 13, "High Point", 63),
        (6, "Illinois", 86, 11, "Xavier", 73),
        (3, "Kentucky", 76, 14, "Troy", 57),
        (7, "UCLA", 72, 10, "Utah St.", 47),
        (2, "Tennessee", 77, 15, "Wofford", 62),
        # South Region
        (1, "Auburn", 83, 16, "Alabama St.", 63),
        (8, "Louisville", 75, 9, "Creighton", 89),
        (5, "Michigan", 68, 12, "UC San Diego", 65),
        (4, "Texas A&M", 80, 13, "Yale", 71),
        (6, "Mississippi", 71, 11, "North Carolina", 64),
        (3, "Iowa St.", 82, 14, "Lipscomb", 55),
        (7, "Marquette", 66, 10, "New Mexico", 75),
        (2, "Michigan St.", 87, 15, "Bryant", 62),
        # West Region
        (1, "Florida", 95, 16, "Norfolk St.", 69),
        (8, "Connecticut", 67, 9, "Oklahoma", 59),
        (5, "Memphis", 70, 12, "Colorado St.", 78),
        (4, "Maryland", 81, 13, "Grand Canyon", 49),
        (6, "Missouri", 57, 11, "Drake", 67),
        (3, "Texas Tech", 82, 14, "UNC Wilmington", 72),
        (7, "Kansas", 72, 10, "Arkansas", 79),
        (2, "St. John's", 83, 15, "Nebraska Omaha", 53),
        # === SECOND ROUND (16 games) ===
        # East Region
        (1, "Duke", 89, 9, "Baylor", 66),
        (4, "Arizona", 87, 5, "Oregon", 83),
        (6, "BYU", 91, 3, "Wisconsin", 89),
        (2, "Alabama", 80, 7, "Saint Mary's", 66),
        # Midwest Region
        (1, "Houston", 81, 8, "Gonzaga", 76),
        (12, "McNeese", 62, 4, "Purdue", 76),
        (3, "Kentucky", 84, 6, "Illinois", 75),
        (2, "Tennessee", 67, 7, "UCLA", 58),
        # South Region
        (1, "Auburn", 82, 9, "Creighton", 70),
        (5, "Michigan", 91, 4, "Texas A&M", 79),
        (6, "Mississippi", 91, 3, "Iowa St.", 78),
        (10, "New Mexico", 63, 2, "Michigan St.", 71),
        # West Region
        (1, "Florida", 77, 8, "Connecticut", 75),
        (12, "Colorado St.", 71, 4, "Maryland", 72),
        (11, "Drake", 64, 3, "Texas Tech", 77),
        (10, "Arkansas", 75, 2, "St. John's", 66),
        # === SWEET 16 (8 games) ===
        # East
        (1, "Duke", 100, 4, "Arizona", 93),
        (2, "Alabama", 113, 6, "BYU", 88),
        # Midwest
        (1, "Houston", 62, 4, "Purdue", 60),
        (3, "Kentucky", 65, 2, "Tennessee", 78),
        # South
        (1, "Auburn", 78, 5, "Michigan", 65),
        (6, "Mississippi", 70, 2, "Michigan St.", 73),
        # West
        (1, "Florida", 87, 4, "Maryland", 71),
        (3, "Texas Tech", 85, 10, "Arkansas", 80),
        # === ELITE EIGHT (4 games) ===
        (1, "Duke", 85, 2, "Alabama", 65),
        (1, "Houston", 69, 2, "Tennessee", 50),
        (1, "Auburn", 70, 2, "Michigan St.", 64),
        (1, "Florida", 84, 3, "Texas Tech", 79),
        # === FINAL FOUR (2 games) ===
        (1, "Florida", 79, 1, "Auburn", 73),
        (1, "Houston", 70, 1, "Duke", 67),
        # === CHAMPIONSHIP ===
        (1, "Florida", 65, 1, "Houston", 63),
    ]

    assert len(games) == 63, f"Expected 63 games, got {len(games)}"

    # Build DataFrame matching restructured_matchups.csv format
    rows = []
    for i, (s1, t1, sc1, s2, t2, sc2) in enumerate(games):
        # Determine round based on game index
        if i < 32:
            rnd = 64  # Round of 64
        elif i < 48:
            rnd = 32  # Round of 32
        elif i < 56:
            rnd = 16  # Sweet 16
        elif i < 60:
            rnd = 8   # Elite 8
        elif i < 62:
            rnd = 4   # Final Four
        else:
            rnd = 2   # Championship

        rows.append({
            "YEAR": 2025,
            "BY YEAR NO": 2036,
            "BY ROUND NO": i,
            "Team1_NO": 0,
            "Team1": t1,
            "Seed1": s1,
            "ROUND": rnd,
            "CURRENT ROUND": rnd,
            "Score1": sc1,
            "Team2_NO": 0,
            "Team2": t2,
            "Seed2": s2,
            "ROUND.1": rnd,
            "CURRENT ROUND.1": rnd,
            "Score2": sc2,
            "Team1_Win": int(sc1 > sc2),
        })

    return pd.DataFrame(rows)


def update_kenpom():
    """Copy 2025 KenPom data from old repo into current kenpom_data.csv."""
    kenpom_path = DATA_DIR / "kenpom" / "kenpom_data.csv"
    old_kenpom_path = OLD_REPO / "kenpom.csv"

    current = pd.read_csv(kenpom_path)
    old = pd.read_csv(old_kenpom_path)

    # Remove any existing 2025 rows
    current = current[current["Year"] != 2025]

    # Get 2025 rows from old repo
    new_2025 = old[old["Year"] == 2025]
    print(f"Adding {len(new_2025)} KenPom 2025 rows")

    updated = pd.concat([current, new_2025], ignore_index=True)
    updated.to_csv(kenpom_path, index=False)
    print(f"Updated kenpom_data.csv: {len(updated)} total rows")


def update_matchups():
    """Replace 2025 rows in restructured_matchups.csv with actual results."""
    matchups_path = DATA_DIR / "matchups" / "restructured_matchups.csv"
    matchups = pd.read_csv(matchups_path)

    # Remove existing 2025 rows
    old_count = len(matchups[matchups["YEAR"] == 2025])
    matchups = matchups[matchups["YEAR"] != 2025]
    print(f"Removed {old_count} old 2025 rows")

    # Add correct 2025 data
    new_2025 = build_2025_matchups()
    updated = pd.concat([new_2025, matchups], ignore_index=True)
    updated.to_csv(matchups_path, index=False)
    print(f"Added {len(new_2025)} corrected 2025 rows")
    print(f"Total matchups: {len(updated)}")


def rebuild_training_data():
    """Re-run merge to rebuild training_data.csv."""
    from sports_quant.march_madness.processing.merge_matchups_stats import (
        merge_matchups_stats,
    )

    result = merge_matchups_stats()
    print(f"Training data rebuilt: {len(result)} rows")
    years = sorted(result["YEAR"].unique())
    print(f"Years: {years}")


if __name__ == "__main__":
    print("=== Step 1: Update KenPom data ===")
    update_kenpom()

    print("\n=== Step 2: Fix 2025 matchups ===")
    update_matchups()

    print("\n=== Step 3: Rebuild training data ===")
    rebuild_training_data()
