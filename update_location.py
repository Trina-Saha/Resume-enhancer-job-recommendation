import pandas as pd
import random

# Read existing CSV file
existing_df = pd.read_csv('Job_db.csv')

# Define locations
locations = ['Kolkata', 'Bangalore', 'Chennai', 'Gurgaon', 'Mumbai', 'Pune', 'Noida', 'Hyderabad']

# Fill 'Location' column with random values for all rows
existing_df['Location'] = random.choices(locations, k=len(existing_df))

# Save updated DataFrame to CSV file
existing_df.to_csv('Job_db2.csv', index=False)

print("CSV file updated successfully.")


