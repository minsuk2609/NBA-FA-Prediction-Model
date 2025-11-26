import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

def season_to_bbr_year(season_str):
    start, end = season_str.split('-')
    return int("20" + end) if len(end) == 2 else int(end)

def flat_data(cols):
    new_cols = []
    for c in cols:
        if isinstance(c, tuple):
            c = "_".join([str(x) for x in c if x])
        c = c.replace(" ", "").replace("/", "_")
        new_cols.append(c)
    return new_cols

def find_team(df_net):
    for col in df_net.columns:
        if "Team" in col or "team" in col:
            return col

def scrape_br(bbr_year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{bbr_year}_ratings.html"
    print(f"Scraping {url} ...")

    res = requests.get(url)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table", {"id": "ratings"})
    df_net = pd.read_html(StringIO(str(table)))[0]

    df_net.columns = flat_data(df_net.columns)
    team_col = find_team(df_net)
    df_net = df_net[df_net[team_col] != "Team"]

    df_net[team_col] = df_net[team_col].str.replace("*", "", regex=False).str.strip()

    metric_cols = {}
    for col in df_net.columns:
        if "ORtg_A" in col:
            metric_cols["ORtg_A"] = col
        elif "DRtg_A" in col:
            metric_cols["DRtg_A"] = col
        elif "NRtg_A" in col:
            metric_cols["NRtg_A"] = col

    df_net = df_net[[team_col] + list(metric_cols.values())]

    df_net = df_net.rename(columns={
        team_col: "Team",
        metric_cols.get("ORtg_A"): "BR_ORtg_A",
        metric_cols.get("DRtg_A"): "BR_DRtg_A",
        metric_cols.get("NRtg_A"): "BR_NRtg_A",
    })

    return df_net

def merge_csv(csv_path):
    csv_path = f"../data/{csv_path}"
    df_main = pd.read_csv(csv_path)

    all_ratings = []
    for season in sorted(df_main["Season"].unique()):
        bbr_year = season_to_bbr_year(season)
        ratings_df = scrape_br(bbr_year)
        ratings_df["Season"] = season
        all_ratings.append(ratings_df)

    ratings_all = pd.concat(all_ratings, ignore_index=True)

    df_final = df_main.merge(
        ratings_all,
        on=["Season", "Team"],
        how="left"
    )

    df_final.to_csv(csv_path, index=False)

    return df_final


if __name__ == "__main__":
    merge_csv("../data/nba_team_stats_2020_2025.csv")
