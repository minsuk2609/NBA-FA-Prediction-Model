import pandas as pd
import unicodedata

# Load your existing CSV
salary_df = pd.read_csv("NBA_Salary_2010-2025.csv")

# Lowercase columns
salary_df.columns = salary_df.columns.str.lower().str.strip()

# Normalize player names
def normalize_name(name):
    return unicodedata.normalize('NFKD', str(name)).encode('ascii','ignore').decode('utf-8').strip().lower()

salary_df['player'] = salary_df['player'].apply(normalize_name)

# Filter for years 2020-2025
salary_df = salary_df[salary_df['year'].between(2020, 2025)]

# Create season column
salary_df['season'] = salary_df['year'].apply(lambda x: f"{x}-{str(x+1)[-2:]}")

# Save filtered CSV
salary_df.to_csv("salaries_2020_2025.csv", index=False)

print("Filtered CSV saved as salaries_2020_2025.csv")
