import pandas as pd
from dateutil.parser import parse

# Load the CSV data into a pandas DataFrame
data = pd.read_csv("data/pff/dates_team_data.csv")

# Rename 'Unnamed: 0' to 'game-string' and 'game' to 'index'
data.rename(columns={'Unnamed: 0': 'game-string', 'game': 'index'}, inplace=True)

# Extract the date by splitting the 'game-string' column after the second hyphen
def extract_date(game_str):
    parts = game_str.split('-')
    if len(parts) > 2:
        date_str = parts[-1].strip()
        try:
            date = parse(date_str)
            # Adjust the year if the month is January or February
            if date.month in [1, 2]:
                date = date.replace(year=date.year + 1)
            return date.strftime('%m/%d/%Y')
        except ValueError:
            return None
    return None

data['date'] = data['game-string'].apply(extract_date)

# Save the modified DataFrame to a new CSV file
data.to_csv("data/pff/dates_team_data.csv", index=False)

print("Date column added, 'game-string' and 'index' columns renamed, and CSV saved successfully.")
