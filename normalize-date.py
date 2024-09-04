import json
from datetime import datetime

def normalize_date(date_float):
    # Convert the float to a string, remove the decimal part, and parse it as a date
    date_str = str(int(date_float))
    normalized_date = datetime.strptime(date_str, "%Y%m%d").strftime("%m/%d/%Y")
    return normalized_date

# Load the JSON data
with open('data/nfl_odds_data.json', 'r') as file:
    data = json.load(file)

# Normalize the dates in the JSON data
for item in data:
    item['date'] = normalize_date(item['date'])

# Save the modified JSON data back to a file
with open('data/normalized_nfl_odds_data.json', 'w') as file:
    json.dump(data, file, indent=2)

print("Dates have been normalized and saved to 'normalized_dates.json'.")
