import pandas as pd
import requests

# Function to fetch new data
def fetch_new_data():
    # Example of getting new data from an API
    url = 'https://api.example.com/data'
    response = requests.get(url)
    new_data = response.json()  # Assuming the response is in JSON format
    return pd.DataFrame(new_data)

# Load existing data
csv_file_path = 'C:/Users/mohamed/Desktop/stage pfe Orange/Orange_data1.csv'
existing_data = pd.read_csv(csv_file_path)

# Fetch new data
new_data = fetch_new_data()

# Append new data to existing data
updated_data = existing_data.append(new_data, ignore_index=True)

# Save updated data back to CSV
updated_data.to_csv(csv_file_path, index=False)
