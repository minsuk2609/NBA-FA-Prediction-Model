import pandas as pd
import unicodedata
import glob
import os

# ----------------------------
# Helper functions
# ----------------------------
def normalize_name(name):
    """Remove accents, lowercase, strip spaces."""
    return unicodedata.normalize('NFKD', str(name)).encode('ascii','ignore').decode('utf-8').strip().lower()

def normalize_season(season_str):
    """Convert '2010' -> '2010-11' format."""
    year = int(season_str)
    return f"{year}-{str(year+1)[-2:]}"

# ----------------------------
# Load main player CSV
# ----------------------------
player_df = pd.read_csv("player_stat_2010-2024.csv")

# Normalize column names
player_df.columns = player_df.columns.str.strip().str.lower()

# Normalize player names and season strings
player_df['player'] = player_df['player'].apply(normalize_name)
player_df['season'] = player_df['season'].apply(lambda x: x[:4] + '-' + x[-2:])

# ----------------------------
# Load all advanced stat CSVs
# ----------------------------
adv_files = sorted(glob.glob("advanced_stat_*.csv"))
all_adv_dfs = []

for file in adv_files:
    year_str = os.path.basename(file).split("_")[2].split(".")[0]
    season = normalize_season(year_str)
    
    adv_df = pd.read_csv(file)
    adv_df.columns = adv_df.columns.str.strip().str.lower()
    adv_df['player'] = adv_df['player'].apply(normalize_name)
    adv_df['season'] = season

    # Keep only relevant columns
    cols_needed = ['player','season','ows','dws','ws','ws/48','obpm','dbpm','bpm','vorp','awards','team']
    adv_df = adv_df[[c for c in cols_needed if c in adv_df.columns]]

    # If player was traded, keep the 'TOT' row first
    adv_df = adv_df.sort_values(by='team')
    adv_df = adv_df.drop_duplicates(subset=['player','season'], keep='first')

    all_adv_dfs.append(adv_df)

# Combine all seasons
adv_stats_df = pd.concat(all_adv_dfs, ignore_index=True)

# ----------------------------
# Merge
# ----------------------------
merged_df = pd.merge(player_df, adv_stats_df.drop(columns=['team']),
                     on=['player','season'],
                     how='left')

# ----------------------------
# Summary / debug
# ----------------------------
matched_count = merged_df['ows'].notna().sum()
print(f"Advanced stats matched for {matched_count}/{len(player_df)} players")

# Optional: see players missing advanced stats
missing_players = merged_df[merged_df['ows'].isna()][['player','season']]
print(f"Missing {len(missing_players)} players")
print(missing_players.head(20))

# ----------------------------
# Save merged CSV
# ----------------------------
merged_df.to_csv("player_stat_2010-2024_with_advanced.csv", index=False)
print("Merged CSV saved: player_stat_2010-2024_with_advanced.csv")
