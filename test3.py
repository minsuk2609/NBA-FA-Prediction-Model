import pandas as pd
import requests
import time

def scrape_advanced_stats(season_years=range(2015, 2026)):
    """
    Scrapes Basketball-Reference advanced stats per season.
    Handles Tm vs Team, TOT rows, and numeric conversion.
    """

    records = []

    for year in season_years:
        season_label = f"{year-1}-{str(year)[-2:]}"
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        print(f"Scraping advanced stats for {season_label} …")

        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        time.sleep(0.7)

        df_list = pd.read_html(resp.content)
        adv = None

        # Find table
        for df in df_list:
            if "Player" in df.columns and ("WS/48" in df.columns or "WS/48 " in df.columns):
                adv = df
                break

        if adv is None:
            print(f"  ⚠️ No advanced stats table found for {season_label}")
            continue

        # Clean up header repeats
        adv = adv[adv["Player"] != "Player"]

        # Fix Player
        adv["Player"] = adv["Player"].astype(str).str.strip()

        # Fix TEAM column: accept either Tm or Team
        if "Tm" in adv.columns:
            adv["TeamFix"] = adv["Tm"]
        elif "Team" in adv.columns:
            adv["TeamFix"] = adv["Team"]
        else:
            print(f"  ⚠️ No team column found for {season_label}")
            continue

        adv["TeamFix"] = adv["TeamFix"].astype(str).str.strip()

        # Handle "TOT" rows
        adv_sorted = adv.sort_values(["Player", "TeamFix"])
        adv_final = (
            adv_sorted
                .groupby("Player", as_index=False, group_keys=False)
                .apply(lambda g: g[g["TeamFix"] == "TOT"] if "TOT" in g["TeamFix"].values else g.iloc[[0]])
        )


        adv_final["Season"] = season_label

        # Normalize column names
        col_map = {
            "WS/48 ": "WS/48",   # sometimes has trailing space
        }
        adv_final.rename(columns=col_map, inplace=True)

        # Ensure required columns exist
        needed = ["Season", "Player", "TeamFix", "MP", "WS/48", "BPM", "VORP"]
        for c in needed:
            if c not in adv_final.columns:
                adv_final[c] = pd.NA

        # Select final columns
        df_out = adv_final[needed].copy()
        df_out.rename(columns={"TeamFix": "Tm"}, inplace=True)

        # Convert numerics
        for c in ["MP", "WS/48", "BPM", "VORP"]:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

        records.append(df_out)

    # Combine all seasons
    if len(records) == 0:
        print("No data scraped.")
        return pd.DataFrame()

    df_all = pd.concat(records, ignore_index=True)
    return df_all
if __name__ == "__main__":
    df_stats = scrape_advanced_stats(season_years=range(2015, 2026))
    df_stats = df_stats.to_csv("nba_advanced_stats_2015_2025.csv", index=False)
    print(f"\n✓ Scraped {len(df_stats)} advanced stats records")
    print(df_stats.head(10))