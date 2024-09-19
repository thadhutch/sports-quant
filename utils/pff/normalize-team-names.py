import pandas as pd
from teams import encoded_teams  # Import the encoded teams dictionary

# Load the dataset
df = pd.read_csv('data/pff/dates_team_data.csv')  # Update with the actual path to your dataset

# Create a reverse mapping of encoded_teams dictionary
reverse_encoded_teams = {v: k for k, v in encoded_teams.items()}

# Function to map team abbreviations to full names
def map_teams(game_string):
    # Split the game string to get team abbreviations
    teams = game_string.split('-')[:2]
    # Map abbreviations to full team names
    team_0 = reverse_encoded_teams.get(teams[0], teams[0])  # Default to abbreviation if not found
    team_1 = reverse_encoded_teams.get(teams[1], teams[1])  # Default to abbreviation if not found
    return team_0, team_1

# Apply the function and create team_0 and team_1 columns
df[['team_0', 'team_1']] = df['game-string'].apply(lambda x: pd.Series(map_teams(x)))

# Display the updated DataFrame
print(df.head())

# Save the updated DataFrame if needed
df.to_csv('data/pff/normalized_team_data.csv', index=False)  # Update with the desired save path
