import pandas as pd

# Load and clean
vegas = pd.read_csv('nba_preseason_odds.csv')

# Keep only what we need
vegas = vegas[['Season', 'Team', 'W-L O/U']].copy()

# Rename for clarity
vegas.rename(columns={'W-L O/U': 'Vegas_OU'}, inplace=True)

# Convert to numeric (remove any non-numeric characters)
vegas['Vegas_OU'] = pd.to_numeric(vegas['Vegas_OU'], errors='coerce')

# Remove rows with missing values
vegas = vegas.dropna(subset=['Vegas_OU'])

# Save cleaned version
vegas.to_csv('nba_vegas_odds_clean.csv', index=False)

print(f"Cleaned {len(vegas)} records")
print(vegas.head(10))
