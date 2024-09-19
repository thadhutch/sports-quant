import pandas as pd

# Load the CSV data into a pandas DataFrame
data = pd.read_csv("data/team_data.csv")

# Rename 'Unnamed: 0' to 'game-string' and 'game' to 'index'
data.rename(columns={'Unnamed: 0': 'game-string', 'game': 'index'}, inplace=True)

# Extract the date by splitting the 'game-string' column after the second hyphen
def extract_date(game_str):
    parts = game_str.split('-')
    return parts[-1] if len(parts) > 2 else None

data['date'] = data['game-string'].apply(extract_date)

# Save the modified DataFrame to a new CSV file
data.to_csv("data/normalized_team_data.csv", index=False)

print("Date column added, 'game-string' and 'index' columns renamed, and CSV saved successfully.")
