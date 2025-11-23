import json
import pandas as pd

json_path = "2025-26.NBA.Roster.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

injury_rows = []

# Iterate through teams in the uploaded JSON
for team in data["teams"]:
    team_name = team["name"]

    # Every player on the roster
    for player in team.get("players", []):
        player_name = player.get("name", "Unknown Player")

        injury = player.get("injury")

        # Only process if an injury object exists
        if injury and isinstance(injury, dict):

            injury_type = injury.get("type", "Unknown")

            # gamesRemaining → expected games out
            games_remaining = injury.get("gamesRemaining", 0)

            # Cap to 82
            if games_remaining > 82:
                games_remaining = 82

            injury_rows.append({
                "Player": player_name,
                "Team": team_name,
                "Injury": injury_type,
                "Games_Missed": games_remaining
            })

# Create DataFrame
injury_df = pd.DataFrame(injury_rows)

print(injury_df.head())
print(f"\nTotal Injuries Found: {len(injury_df)}")

# Optional CSV output
injury_df.to_csv("nba_injuries_2025_26_preseason.csv", index=False)
print("\nSaved: nba_injuries_2025_26_preseason.csv")
