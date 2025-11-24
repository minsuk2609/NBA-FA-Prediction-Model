import pandas as pd
import numpy as np

# -----------------------------------------
# TEAM MAPS
# -----------------------------------------

TEAM_ABBREVS = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BRK',
    'Charlotte Hornets': 'CHO', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHO',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

# Reverse mapping
ABBR_TO_TEAM = {v: k for k, v in TEAM_ABBREVS.items()}

# Normalize Nets naming
ABBR_TO_TEAM["BKN"] = "Brooklyn Nets"
ABBR_TO_TEAM["BRK"] = "Brooklyn Nets"


# -----------------------------------------
# LOAD STAR SCORES (2015–2025)
# -----------------------------------------

def load_star_df():
    print("Loading star metrics (WS/48, BPM, VORP, LEBRON, WAR)…")

    adv = pd.read_csv("nba_advanced_stats_2015_2025.csv")
    leb = pd.read_csv("nba_2014_2025_LEBRON.csv")

    adv["Player"] = adv["Player"].astype(str).str.strip()
    adv["Season"] = adv["Season"].astype(str).str.strip()

    leb["Player"] = leb["Player"].astype(str).str.strip()
    leb["Season"] = leb["Season"].astype(str).str.strip()

    # Merge adv + LEBRON metrics
    df = adv.merge(leb[["Season", "Player", "LEBRON", "LEBRON WAR"]],
                   on=["Season", "Player"], how="left")

    for col in ["WS/48", "BPM", "VORP", "LEBRON", "LEBRON WAR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Scale metrics
    metric_cols = ["WS/48", "BPM", "VORP", "LEBRON", "LEBRON WAR"]
    for m in metric_cols:
        mn, mx = df[m].min(), df[m].max()
        df[m + "_scaled"] = (df[m] - mn) / (mx - mn) if mx > mn else 0

    df["StarScore"] = df[[m + "_scaled" for m in metric_cols]].mean(axis=1)

    # keep only columns needed
    df = df[["Season", "Player", "StarScore"]]

    print(f"✓ Loaded star data for {len(df)} season-player rows")
    return df


# -----------------------------------------
# LOAD EXISTING INJURY CSV
# -----------------------------------------

def load_existing_injuries():
    print("Loading existing injury data…")
    df = pd.read_csv("nba_injury_data.csv")
    return df


# -----------------------------------------
# UPDATE WITH 2025-26 PRESEASON Injuries.csv
# -----------------------------------------

def update_injuries_2025_26():
    existing = load_existing_injuries()
    star_df = load_star_df()

    print("Loading preseason injuries (Injuries.csv)…")
    inj = pd.read_csv("Injuries.csv")

    inj["Name"] = inj["Name"].astype(str).str.strip()
    inj["Team"] = inj["Team"].astype(str).str.strip()
    inj["Games"] = pd.to_numeric(inj["Games"], errors="coerce").fillna(0).clip(0, 82)

    # Normalize Nets naming
    inj["Team"] = inj["Team"].replace({"BKN": "Brooklyn Nets", "BRK": "Brooklyn Nets"})

    # Merge star scores from 2024-25
    star_prev = star_df[star_df["Season"] == "2024-25"]
    inj = inj.merge(star_prev, left_on="Name", right_on="Player", how="left")
    inj["StarScore"] = inj["StarScore"].fillna(0)
    inj = inj.drop(columns=["Player"])

    # -----------------------------------------
    # Build injury rows per team
    # -----------------------------------------

    rows = []

    TEAMS = list(TEAM_ABBREVS.keys())  # all 30 teams

    for team in TEAMS:
        team_inj = inj[inj["Team"] == team]

        # injury-prone = any player with Games > 0 (injured)
        injury_prone_count = (team_inj["Games"] > 0).sum()

        # players missing from Injuries.csv → still injury prone? → yes
        if team_inj.empty:
            injury_prone_count = 0  # no reported injuries

        # star-weighted missed (only for players who were in top 8 last season)
        star_weighted_missed = (team_inj["Games"] * (1 + team_inj["StarScore"])).sum()
        plain_missed = team_inj["Games"].sum()

        rows.append({
            "Season": "2025-26",
            "Team": team,
            "Games_Missed_Top8": float(plain_missed),
            "Star_Weighted_Games_Missed_Top8": float(star_weighted_missed),
            "Injury_Prone_Count": int(injury_prone_count),
            "Top_Player_Games": 0
        })

        print(f"  ✓ {team}: missed={plain_missed}, star={star_weighted_missed}, injured={injury_prone_count}")

    # -----------------------------------------
    # Append to existing data
    # -----------------------------------------

    new_df = pd.DataFrame(rows)
    final = pd.concat([existing, new_df], ignore_index=True)

    # Save
    final.to_csv("nba_injury_data.csv", index=False)

    print("\n✓ Updated nba_injury_data.csv with 2025-26 preseason injuries.")
    print(f"Total rows now: {len(final)}")


# -----------------------------------------
# MAIN
# -----------------------------------------

if __name__ == "__main__":
    update_injuries_2025_26()
