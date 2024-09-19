import pandas as pd

# Load the first CSV with game details
df1 = pd.read_csv('data/pfr/final_pfr_odds.csv')  # Update with the path to your first CSV file

# Load the second CSV with game statistics
df2 = pd.read_csv('data/pff/normalized_team_data.csv')  # Update with the path to your second CSV file

# Merge the dataframes on 'Formatted Date', 'away_team', and 'home_team'
merged_df = pd.merge(
    df1,
    df2,
    left_on=['Formatted Date', 'away_team', 'home_team'],
    right_on=['date', 'team_0', 'team_1'],
    how='inner'  # Change to 'outer' if you want to include non-matching rows as well
)

# Drop any columns that are redundant after merging, if necessary
merged_df.drop(columns=['date', 'team_0', 'team_1'], inplace=True)

# Display the merged dataframe
print(merged_df.head())

# Save the merged dataframe to a CSV file if needed
merged_df.to_csv('data/pff_and_pfr_data.csv', index=False)  # Update with the desired save path
