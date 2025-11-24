import json
import pandas as pd

# Load your JSON file
with open("2025-26.NBA.Roster.json", "r", encoding="utf-8") as f:
    data = json.load(f)

players = data.get("players", [])

# Full NBA team names mapped to tid (confirmed from your file)
TEAM_MAP_FULL = {
    0: "Atlanta Hawks",
    1: "Boston Celtics",
    2: "Brooklyn Nets",
    3: "Charlotte Hornets",
    4: "Chicago Bulls",
    5: "Cleveland Cavaliers",
    6: "Dallas Mavericks",
    7: "Denver Nuggets",
    8: "Detroit Pistons",
    9: "Golden State Warriors",
    10: "Houston Rockets",
    11: "Indiana Pacers",
    12: "LA Clippers",
    13: "Los Angeles Lakers",
    14: "Memphis Grizzlies",
    15: "Miami Heat",
    16: "Milwaukee Bucks",
    17: "Minnesota Timberwolves",
    18: "New Orleans Pelicans",
    19: "New York Knicks",
    20: "Oklahoma City Thunder",
    21: "Orlando Magic",
    22: "Philadelphia 76ers",
    23: "Phoenix Suns",
    24: "Portland Trail Blazers",
    25: "Sacramento Kings",
    26: "San Antonio Spurs",
    27: "Toronto Raptors",
    28: "Utah Jazz",
    29: "Washington Wizards",
}

rows = []

for p in players:

    # Ensure this is a player dict with a name
    if not isinstance(p, dict):
        continue
    if "name" not in p or "tid" not in p:
        continue

    name = p["name"]
    tid = p["tid"]

    # Skip unassigned players (FA, G-League, retired)
    if tid not in TEAM_MAP_FULL:
        continue

    team_name = TEAM_MAP_FULL[tid]

    rows.append({
        "Player": name,
        "Team": team_name,
    })

# Create DataFrame
df = pd.DataFrame(rows)

# Sort for readability
df = df.sort_values(["Team", "Player"]).reset_index(drop=True)

# Save CSV output
df.to_csv("nba_roster_2025_26_bgm.csv", index=False)

print("✔ Completed: nba_roster_2025_26_bgm.csv has been created")
