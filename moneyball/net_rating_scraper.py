import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

# =====================================================================
# Helper functions
# =====================================================================

def season_to_bbr_year(season_str):
    """Convert '2024-25' → 2025"""
    start, end = season_str.split('-')
    return int("20" + end) if len(end) == 2 else int(end)


def flatten_columns(cols):
    """Flatten MultiIndex columns into strings."""
    new_cols = []
    for c in cols:
        if isinstance(c, tuple):
            c = "_".join([str(x) for x in c if x])
        c = c.replace(" ", "").replace("/", "_")
        new_cols.append(c)
    return new_cols


def find_team_column(df):
    """Find the column that contains team names."""
    for col in df.columns:
        if "Team" in col or "team" in col:
            return col
    raise ValueError("Team column not found")


def scrape_bbr_team_ratings(bbr_year):
    """
    Scrape ORtg/A, DRtg/A, NRtg/A for a given season.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{bbr_year}_ratings.html"
    print(f"Scraping {url} ...")

    res = requests.get(url)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table", {"id": "ratings"})

    df = pd.read_html(StringIO(str(table)))[0]

    # Flatten MultiIndex columns
    df.columns = flatten_columns(df.columns)

    # Identify team column
    team_col = find_team_column(df)

    # Remove repeated header rows
    df = df[df[team_col] != "Team"]

    # Clean team names
    df[team_col] = df[team_col].str.replace("*", "", regex=False).str.strip()

    # Locate metric columns
    metric_cols = {}
    for col in df.columns:
        if "ORtg_A" in col:
            metric_cols["ORtg_A"] = col
        if "DRtg_A" in col:
            metric_cols["DRtg_A"] = col
        if "NRtg_A" in col:
            metric_cols["NRtg_A"] = col

    # Keep only the needed columns
    keep = [team_col] + list(metric_cols.values())
    df = df[keep]

    # Make consistent column names
    df = df.rename(columns={
        team_col: "Team",
        metric_cols.get("ORtg_A", None): "BR_ORtg_A",
        metric_cols.get("DRtg_A", None): "BR_DRtg_A",
        metric_cols.get("NRtg_A", None): "BR_NRtg_A",
    })

    return df


# =====================================================================
# Main process — OVERWRITE ORIGINAL CSV
# =====================================================================

def merge_ratings_into_main_csv(csv_path="nba_team_stats_2020_2025.csv"):
    df_main = pd.read_csv(csv_path)

    all_ratings = []

    for season in sorted(df_main["Season"].unique()):
        bbr_year = season_to_bbr_year(season)
        ratings_df = scrape_bbr_team_ratings(bbr_year)
        ratings_df["Season"] = season
        all_ratings.append(ratings_df)

    ratings_all = pd.concat(all_ratings, ignore_index=True)

    df_final = df_main.merge(
        ratings_all,
        on=["Season", "Team"],
        how="left"
    )

    # 🔥 Overwrite original CSV
    df_final.to_csv(csv_path, index=False)
    print(f"\n✓ Updated file saved (OVERWRITTEN): {csv_path}")

    return df_final


# =====================================================================

if __name__ == "__main__":
    merge_ratings_into_main_csv()
