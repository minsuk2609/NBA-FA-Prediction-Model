import pandas as pd
import unicodedata

# Parameters
INFLATION_RATE = 0.04  # 4% per year
TARGET_YEAR = 2025     # adjust all salaries to 2025 dollars

# Load CSV
salary_df = pd.read_csv("NBA_Salary_2010-2025.csv")

# Lowercase and strip columns
salary_df.columns = salary_df.columns.str.lower().str.strip()

# Normalize player names
def normalize_name(name):
    return unicodedata.normalize('NFKD', str(name)).encode('ascii','ignore').decode('utf-8').strip().lower()

salary_df['player'] = salary_df['player'].apply(normalize_name)

# Filter for 2020-2025
salary_df = salary_df[salary_df['year'].between(2020, 2025)]

# Create season column
salary_df['season'] = salary_df['year'].apply(lambda x: f"{x}-{str(x+1)[-2:]}")

# Create avg_salary column if it doesn't exist
if 'avg_salary' not in salary_df.columns:
    if 'salary' in salary_df.columns:
        salary_df['avg_salary'] = salary_df['salary']
    else:
        raise ValueError("No salary column found to create avg_salary!")

# Adjust for inflation
salary_df['adjusted_salary'] = salary_df.apply(
    lambda row: row['avg_salary'] * ((1 + INFLATION_RATE) ** (TARGET_YEAR - row['year'])),
    axis=1
)

# Save to new CSV
salary_df.to_csv("salaries_2020_2025_adjusted.csv", index=False)

print("CSV saved as salaries_2020_2025_adjusted.csv with adjusted salaries")
