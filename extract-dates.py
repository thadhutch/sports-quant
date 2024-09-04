import pandas as pd
import re

# Load the CSV data into a pandas DataFrame
data = pd.read_csv("data/team_data.csv")

# Extract the index (which contains the dates) and reset the index
data.reset_index(inplace=True)

# Rename the index column to 'game' or something descriptive
data.rename(columns={'index': 'game'}, inplace=True)

# Convert the 'game' column to string
data['game'] = data['game'].astype(str)

# Extract the date using a regular expression with error handling
def extract_date(game_str):
    match = re.search(r'\d{2}/\d{2}/\d{4}', game_str)
    return match.group() if match else None

data['date'] = data['game'].apply(extract_date)

# Optionally, you can drop the 'game' column if you only want the date
# data.drop(columns=['game'], inplace=True)

# Save the modified DataFrame to a new CSV file
data.to_csv("data/normalized_team_data.csv", index=False)

print("Date column added and CSV saved successfully.")
