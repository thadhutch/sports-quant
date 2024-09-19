import os
import pandas as pd
from dateutil.parser import parse

# Load the CSV file into a DataFrame
df = pd.read_csv('data/pfr/regular_game_data.csv')

# Function to extract and format the date
def extract_date(title):
    # Find the date portion in the title (after the second hyphen)
    date_str = title.split('-')[-1].strip()
    # Parse the date and format it as 'MM/DD/YYYY'
    parsed_date = parse(date_str).strftime('%m/%d/%Y')
    return parsed_date

# Apply the function to extract dates from the Title column
df['Formatted Date'] = df['Title'].apply(extract_date)

# Display the DataFrame with the formatted dates
print(df[['Title', 'Formatted Date']])

# Define the output folder and file path
output_folder = 'data/pfr'
output_file = os.path.join(output_folder, 'normalized_pfr_odds.csv')

df.to_csv(output_file, index=False)

