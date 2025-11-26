import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime

TEAM_ABBREVS = {
    'Atlanta Hawks': 'ATL','Boston Celtics': 'BOS','Brooklyn Nets': 'BRK',
    'Charlotte Hornets': 'CHO','Chicago Bulls': 'CHI','Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL','Denver Nuggets': 'DEN','Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW','Houston Rockets': 'HOU','Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC','Los Angeles Lakers': 'LAL','Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA','Milwaukee Bucks': 'MIL','Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP','New York Knicks': 'NYK','Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL','Philadelphia 76ers': 'PHI','Phoenix Suns': 'PHO',
    'Portland Trail Blazers': 'POR','Sacramento Kings': 'SAC','San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR','Utah Jazz': 'UTA','Washington Wizards': 'WAS'
}

ABBR_TO_TEAM = {v: k for k, v in TEAM_ABBREVS.items()}


def load_data(adv_file, lebron_file):
    adv = pd.read_csv(adv_file)
    adv["Player"] = adv["Player"].astype(str).str.strip()
    adv["Season"] = adv["Season"].astype(str).str.strip()
    adv["Tm"] = adv["Tm"].astype(str).str.strip()

    def pick_main_row(group):
        multi_tags = {"TOT", "2TM", "3TM", "4TM"}
        if "Tm" in group.columns:
            mask = group["Tm"].isin(multi_tags)
            if mask.any():
                return group[mask].iloc[0]
        if "MP" in group.columns:
            mp = pd.to_numeric(group["MP"], errors="coerce")
            if mp.notna().any():
                return group.iloc[int(mp.fillna(-1).values.argmax())]
        return group.iloc[0]

    adv_main = adv.groupby(["Season", "Player"], as_index=False, group_keys=False).apply(pick_main_row).reset_index(drop=True)


    leb = pd.read_csv(lebron_file)
    leb["Player"] = leb["Player"].astype(str).str.strip()
    leb["Season"] = leb["Season"].astype(str).str.strip()
    leb = leb[["Season", "Player", "LEBRON WAR", "LEBRON"]]

    players = adv_main.merge(leb, on=["Season", "Player"], how="left")


    for col in ["WS/48", "BPM", "VORP", "LEBRON WAR", "LEBRON"]:
        players[col] = pd.to_numeric(players[col], errors="coerce").fillna(0.0)

    metrics = ["WS/48", "BPM", "VORP", "LEBRON", "LEBRON WAR"]
    for m in metrics:
        vmin, vmax = players[m].min(), players[m].max()
        if vmax > vmin:
            players[m + "_scaled"] = (players[m] - vmin) / (vmax - vmin)
        else:
            players[m + "_scaled"] = 0.0


    players["StarScore"] = players[[m + "_scaled" for m in metrics]].mean(axis=1)
    return players[["Season", "Player", "StarScore", "MP"]]




def scrape_roster(start_year, end_year, star_df):
    star_df = star_df.copy()
    star_df["Player"] = star_df["Player"].astype(str).str.strip()
    star_df["Season"] = star_df["Season"].astype(str).str.strip()

    continuity_data = []
    bgm_roster_df = None

    for year in range(start_year + 1, end_year + 1):
        season = f"{year-1}-{str(year)[-2:]}"
        prev_season = f"{year-2}-{str(year-1)[-2:]}"

        for team_name, abbrev in TEAM_ABBREVS.items():
            print(f"Roster for {team_name}")
            
            url_prev = f"https://www.basketball-reference.com/teams/{abbrev}/{year-1}.html"
            resp_prev = requests.get(url_prev, headers={'User-Agent': 'Mozilla/5.0'})
            time.sleep(1.5)

            if resp_prev.status_code != 200:
                print(f"Error loading previous roster for {team_name}")
                continue

            tables_prev = pd.read_html(resp_prev.content)
            roster_prev = next((t for t in tables_prev if "Player" in t.columns and "MP" in t.columns), None)
            if roster_prev is None:
                print(f"Error: no previous roster table for {team_name}")
                continue

            roster_prev = roster_prev[roster_prev["Player"] != "Player"]
            roster_prev["Player"] = roster_prev["Player"].astype(str).str.strip()

            prev_players = {}
            for _, r in roster_prev.iterrows():
                prev_players[r["Player"]] = float(r["MP"])

            if not prev_players:
                continue

            prev_df = pd.DataFrame([{"Player": p, "Minutes": m} for p, m in prev_players.items()])
            stars_prev = star_df[star_df["Season"] == prev_season][["Player", "StarScore"]]
            prev_df = prev_df.merge(stars_prev, on="Player", how="left").fillna({"StarScore": 0.0})

            prev_df = prev_df.sort_values(["StarScore", "Minutes"], ascending=False)
            prev_top8 = prev_df.head(8).copy()

            total_prev_min = prev_top8["Minutes"].sum()
            prev_top8.loc[:, "StarWeight"] = prev_top8["Minutes"] * (1 + prev_top8["StarScore"])
            total_prev_star = prev_top8["StarWeight"].sum()

            if year == 2026:
                if bgm_roster_df is None:
                    bgm_roster_df = pd.read_csv("../data/nba_roster_2025_26_bgm.csv")
                    bgm_roster_df["Player"] = bgm_roster_df["Player"].astype(str).str.strip()
                    bgm_roster_df["Team"] = bgm_roster_df["Team"].astype(str).str.strip()

                lookup_name = "LA Clippers" if team_name == "Los Angeles Clippers" else team_name
                curr_players = bgm_roster_df[bgm_roster_df["Team"] == lookup_name]["Player"].tolist()

            else:
                url_curr = f"https://www.basketball-reference.com/teams/{abbrev}/{year}.html"
                resp_curr = requests.get(url_curr, headers={'User-Agent': 'Mozilla/5.0'})
                time.sleep(1.5)
                if resp_curr.status_code != 200:
                    print(f"Error loading current roster for {team_name}")
                    continue

                tables_curr = pd.read_html(resp_curr.content)
                roster_curr = next((t for t in tables_curr if "Player" in t.columns), None)
                if roster_curr is None:
                    continue

                roster_curr = roster_curr[roster_curr["Player"] != "Player"]
                roster_curr["Player"] = roster_curr["Player"].astype(str).str.strip()
                curr_players = roster_curr["Player"].tolist()

            if not curr_players:
                continue

            ret_min, ret_star, new_top = 0, 0, 0
            for p in curr_players:
                row = prev_top8[prev_top8["Player"] == p]
                if not row.empty:
                    ret_min += float(row["Minutes"].iloc[0])
                    ret_star += float(row["StarWeight"].iloc[0])
                else:
                    new_top += 1

            pct_top8 = ret_min / total_prev_min * 100 if total_prev_min > 0 else 0
            pct_star = ret_star / total_prev_star * 100 if total_prev_star > 0 else pct_top8

            full_prev_min = prev_df["Minutes"].sum()
            ret_full = prev_df[prev_df["Player"].isin(curr_players)]["Minutes"].sum()
            pct_full = ret_full / full_prev_min * 100 if full_prev_min > 0 else 0

            continuity_data.append({
                "Season": season,
                "Team": team_name,
                "Returning_Minutes_Top8_Pct": round(pct_top8, 2),
                "Star_Weighted_Continuity": round(pct_star, 2),
                "New_Players_Top8": new_top,
                "Returning_Minutes_FullRoster_Pct": round(pct_full, 2),
                "Total_Prev_Minutes_Top8": round(total_prev_min, 1),
                "Total_Prev_Minutes_FullRoster": round(full_prev_min, 1)
            })

    df_continuity = pd.DataFrame(continuity_data)
    df_continuity.to_csv("../data/nba_roster_continuity.csv", index=False)
    return df_continuity


def scrape_injury(start_year, end_year, star_df):
    star_df = star_df.copy()
    star_df["Player"] = star_df["Player"].astype(str).str.strip()
    star_df["Season"] = star_df["Season"].astype(str).str.strip()

    injury_data = []
    prev_top8 = {}

    for year in range(start_year, end_year + 1):
        season = f"{year-1}-{str(year)[-2:]}"

        for team_name, abbrev in TEAM_ABBREVS.items():
            print(f"Injuries for {team_name}")

            url = f"https://www.basketball-reference.com/teams/{abbrev}/{year}.html"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            time.sleep(1.5)

            if resp.status_code != 200:
                print(f"Error loading roster for {team_name}")
                continue

            tables = pd.read_html(resp.content)
            roster = next((t for t in tables if "Player" in t.columns and "G" in t.columns), None)
            if roster is None:
                continue

            roster = roster[roster['Player'] != 'Player']
            roster['Player'] = roster['Player'].astype(str).str.strip()
            roster["MP"] = pd.to_numeric(roster["MP"], errors="coerce")
            roster = roster.sort_values("MP", ascending=False)

            total_games = 72 if year == 2021 else 82

            top8 = roster.head(8).copy()
            stars = star_df[star_df["Season"] == season][["Player", "StarScore"]]
            top8 = top8.merge(stars, on="Player", how="left").fillna({"StarScore": 0})

            missed, missed_star, injury_prone = 0, 0, 0

            for _, r in top8.iterrows():
                gp = int(r["G"])
                gm = max(0, total_games - gp)
                missed += gm
                missed_star += gm * (1 + float(r["StarScore"]))

            for _, r in roster.iterrows():
                gp = int(r["G"])
                if total_games - gp >= 20:
                    injury_prone += 1

            top_g = int(top8.iloc[0]["G"]) if len(top8) > 0 else 0

            injury_data.append({
                "Season": season,
                "Team": team_name,
                "Games_Missed_Top8": missed,
                "Star_Weighted_Games_Missed_Top8": missed_star,
                "Injury_Prone_Count": injury_prone,
                "Top_Player_Games": top_g
            })

            if season == "2024-25":
                prev_top8[team_name] = top8[["Player", "G", "MP", "StarScore"]].copy()

    inj = pd.read_csv("../data/Injuries.csv")
    inj["Name"] = inj["Name"].astype(str).str.strip()
    inj["Team"] = inj["Team"].astype(str).str.strip()
    inj["Games"] = pd.to_numeric(inj["Games"], errors="coerce").fillna(0).clip(upper=82).astype(int)

    stars_24 = star_df[star_df["Season"] == "2024-25"][["Player", "StarScore"]]
    inj = inj.merge(stars_24, left_on="Name", right_on="Player", how="left").drop(columns=["Player"])
    inj["StarScore"] = inj["StarScore"].fillna(0)

    preseason_rows = []

    for abbr in inj["Team"].unique():
        dfteam = inj[inj["Team"] == abbr]
        team_name = ABBR_TO_TEAM.get(abbr, abbr)
        prev = prev_top8.get(team_name)

        miss, miss_star, count = 0, 0, 0
        prev_names = set(prev["Player"]) if prev is not None else set()

        for _, r in dfteam.iterrows():
            nm = r["Name"]
            gm = int(r["Games"])
            star = float(r["StarScore"])
            count += 1
            if nm in prev_names:
                miss += gm
                miss_star += gm * (1 + star)

        preseason_rows.append({
            "Season": "2025-26",
            "Team": team_name,
            "Games_Missed_Top8": miss,
            "Star_Weighted_Games_Missed_Top8": miss_star,
            "Injury_Prone_Count": count,
            "Top_Player_Games": 0
        })

    injury_data.extend(preseason_rows)

    df_continuity = pd.DataFrame(injury_data)
    df_continuity.to_csv("../data/nba_injury_data.csv", index=False)
    return df_continuity


def main():
    star_df = load_data(adv_file="../data/nba_advanced_stats_2015_2025.csv",lebron_file="../data/nba_2014_2025_LEBRON.csv")

    scrape_roster(start_year=2015, end_year=2025, star_df=star_df)
    scrape_injury(start_year=2015, end_year=2025, star_df=star_df)


if __name__ == "__main__":
    main()
