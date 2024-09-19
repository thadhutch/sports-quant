import pandas as pd

# Read the modified CSV file into a DataFrame
df = pd.read_csv('data/over-under/v1-dataset-modified.csv')

# Filter rows where 'Houston Texans' are the home or away team
houston_games = df[(df['home_team'] == 'Houston Texans') | (df['away_team'] == 'Houston Texans')]

# Reset index if needed
houston_games = houston_games.reset_index(drop=True)

# Print the filtered DataFrame
print(houston_games)

# Optionally, save the filtered DataFrame to a new CSV file
houston_games.to_csv('data/over-under/houston-texans-games.csv', index=False)
