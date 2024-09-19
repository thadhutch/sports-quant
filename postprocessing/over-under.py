import pandas as pd

# Load the dataset
df = pd.read_csv('data/pff_and_pfr_data.csv')

# Function to determine the 'total' value based on the 'Over/Under' column
def set_total(over_under):
    if '(under)' in over_under:
        return 0
    elif '(over)' in over_under:
        return 1
    return None  # In case the value doesn't match either condition

# Function to extract the numeric value from the 'Over/Under' column
def extract_ou_line(over_under):
    # Split the string and try to convert the first part to a float
    try:
        return float(over_under.split()[0])
    except ValueError:
        return None  # Return None if conversion fails

# Apply the functions to create the 'total' and 'ou_line' columns
df['total'] = df['Over/Under'].apply(set_total)
df['ou_line'] = df['Over/Under'].apply(extract_ou_line)

# Display the first few rows to verify the new columns
print(df.head())

# Save the updated DataFrame back to the CSV file if needed
df.to_csv('data/over-under/v1-dataset.csv', index=False)

print("New columns 'total' and 'ou_line' added based on 'Over/Under' values and CSV saved successfully.")
