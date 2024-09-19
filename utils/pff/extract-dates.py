import pandas as pd
from dateutil.parser import parse

# Load the CSV data into a pandas DataFrame
data = pd.read_csv("data/pff/dates_team_data.csv")

# Rename 'Unnamed: 0' to 'game-string' and 'game' to 'index'
data.rename(columns={'Unnamed: 0': 'game-string', 'game': 'index'}, inplace=True)

# Extract the date and season by splitting the 'game-string' column after the second hyphen
def extract_date_and_season(game_str):
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

# Apply the function to create date and season columns
data[['date', 'season']] = data['game-string'].apply(lambda x: pd.Series(extract_date_and_season(x)))

# Save the modified DataFrame to a new CSV file
data.to_csv("data/pff/dates_team_data.csv", index=False)

print("Date and season columns added, 'game-string' and 'index' columns renamed, and CSV saved successfully.")
