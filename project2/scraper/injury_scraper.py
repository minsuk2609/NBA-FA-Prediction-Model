import json
import pandas as pd

def parse_injuries(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

    injury_rows = []

    for team in data.get("teams", []):
        team_name = team.get("name")
        if not team_name:
            continue

        for player in team.get("players", []):
            player_name = player.get("name")
            if not player_name:
                continue

            injury = player.get("injury")
            if not injury or not isinstance(injury, dict):
                continue


            injury_type = injury.get("type", "Unknown")
            games_remaining = injury.get("gamesRemaining", 0)
            if games_remaining > 82:
                games_remaining = 82

            injury_rows.append({
                "Player": player_name,
                "Team": team_name,
                "Injury": injury_type,
                "Games_Missed": games_remaining
            })


    return pd.DataFrame(injury_rows)


def main():
    df_injury = parse_injuries("../data/2025-26.NBA.Roster.json")
    if df_injury is not None:
        try:
            df_injury.to_csv("../data/nba_injuries_2025_26_preseason.csv", index=False)
        except Exception as e:
            print(f"Error saving CSV: {e}")


if __name__ == "__main__":
    main()