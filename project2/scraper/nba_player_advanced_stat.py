import pandas as pd
import requests
import time

def scrape_advanced_stats(season_years):
    records = []

    for year in season_years:
        season_label = f"{year-1}-{str(year)[-2:]}"
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"

        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        except Exception as e:
            print(f"Error fetching {season_label}: {e}")
            continue

        time.sleep(0.7)

        try:
            df_list = pd.read_html(resp.content)
        except Exception as e:
            print(f"Error reading HTML for {season_label}: {e}")
            continue

        adv = None
        for df in df_list:
            if "Player" in df.columns and ("WS/48" in df.columns or "WS/48 " in df.columns):
                adv = df
                break

        if adv is None:
            print(f"No advanced stats table found for {season_label}")
            continue

        adv = adv[adv["Player"] != "Player"]
        adv["Player"] = adv["Player"].astype(str).str.strip()

        if "Tm" in adv.columns:
            adv["TeamFix"] = adv["Tm"]
        elif "Team" in adv.columns:
            adv["TeamFix"] = adv["Team"]
        else:
            print(f"No team column found for {season_label}")
            continue

        adv["TeamFix"] = adv["TeamFix"].astype(str).str.strip()

        try:
            adv_sorted = adv.sort_values(["Player", "TeamFix"])
            adv_final = adv_sorted.groupby("Player", as_index=False, group_keys=False).apply(
                lambda g: g[g["TeamFix"] == "TOT"] if "TOT" in g["TeamFix"].values else g.iloc[[0]]
            )
        except Exception as e:
            print(f"Error processing TOT rows for {season_label}: {e}")
            continue

        adv_final["Season"] = season_label
        adv_final.rename(columns={"WS/48 ": "WS/48"}, inplace=True)

        needed = ["Season", "Player", "TeamFix", "MP", "WS/48", "BPM", "VORP"]
        for c in needed:
            if c not in adv_final.columns:
                adv_final[c] = pd.NA

        df_out = adv_final[needed].copy()
        df_out.rename(columns={"TeamFix": "Tm"}, inplace=True)

        for c in ["MP", "WS/48", "BPM", "VORP"]:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

        records.append(df_out)

    if len(records) == 0:
        print("No data scraped.")
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


if __name__ == "__main__":
    df_stats = scrape_advanced_stats(range(2017, 2025))
    try:
        df_stats.to_csv("../data/nba_advanced_stats_2015_2025.csv", index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}")
