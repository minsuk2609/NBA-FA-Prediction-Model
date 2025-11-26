import json
import pandas as pd

def load_roster(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

    players = data.get("players", [])

    TEAM_MAP_FULL = {
        0: "Atlanta Hawks", 1: "Boston Celtics", 2: "Brooklyn Nets",
        3: "Charlotte Hornets", 4: "Chicago Bulls", 5: "Cleveland Cavaliers",
        6: "Dallas Mavericks", 7: "Denver Nuggets", 8: "Detroit Pistons",
        9: "Golden State Warriors", 10: "Houston Rockets", 11: "Indiana Pacers",
        12: "LA Clippers", 13: "Los Angeles Lakers", 14: "Memphis Grizzlies",
        15: "Miami Heat", 16: "Milwaukee Bucks", 17: "Minnesota Timberwolves",
        18: "New Orleans Pelicans", 19: "New York Knicks",
        20: "Oklahoma City Thunder", 21: "Orlando Magic",
        22: "Philadelphia 76ers", 23: "Phoenix Suns",
        24: "Portland Trail Blazers", 25: "Sacramento Kings",
        26: "San Antonio Spurs", 27: "Toronto Raptors",
        28: "Utah Jazz", 29: "Washington Wizards"
    }

    rows = []

    for p in players:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        tid = p.get("tid")
        if name is None or tid is None:
            continue
        if tid not in TEAM_MAP_FULL:
            continue
        rows.append({"Player": name, "Team": TEAM_MAP_FULL[tid]})


    df_roster = pd.DataFrame(rows)
    df_roster = df_roster.sort_values(["Team", "Player"]).reset_index(drop=True)
    return df_roster

def main():
    df_roster = load_roster("../data/2025-26.NBA.Roster.json")
    if df_roster is not None:
        try:
            df_roster.to_csv("../data/nba_roster_2025_26_bgm.csv", index=False)
        except Exception as e:
            print(f"Error saving CSV: {e}")


if __name__ == "__main__":
    main()