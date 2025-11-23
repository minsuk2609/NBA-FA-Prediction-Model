import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime

# ============================================
# GLOBAL TEAM MAPS
# ============================================

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

ABBR_TO_TEAM = {v: k for k, v in TEAM_ABBREVS.items()}


# ============================================
# 0. LOAD PLAYER IMPACT DATA (WS/48, BPM, VORP, LEBRON, WAR)
# ============================================

def load_player_impact_data(
    adv_file="nba_advanced_stats_2015_2025.csv",
    lebron_file="nba_2014_2025_LEBRON.csv"
):
    """
    Load advanced box-score stats + LEBRON data and build a StarScore per
    (Season, Player).

    StarScore = mean of 5 scaled metrics:
      - WS/48
      - BPM
      - VORP
      - LEBRON
      - LEBRON WAR
    """

    print("=" * 80)
    print("LOADING PLAYER IMPACT DATA (WS/48, BPM, VORP, LEBRON, WAR)")
    print("=" * 80)

    # ---- Load advanced stats ----
    adv = pd.read_csv(adv_file)
    # Expecting columns: Season, Player, Tm, MP, WS/48, BPM, VORP

    # Clean basic fields
    adv["Player"] = adv["Player"].astype(str).str.strip()
    adv["Season"] = adv["Season"].astype(str).str.strip()
    adv["Tm"] = adv["Tm"].astype(str).str.strip()

    # Some players have multiple rows (trades). Prefer a "TOT" / "xTM" type row
    # or else the row with highest MP.
    def pick_main_row(group: pd.DataFrame) -> pd.Series:
        # If dataset has an aggregate row like "TOT"/"2TM"/"3TM"/"4TM", prefer it
        multi_tags = {"TOT", "2TM", "3TM", "4TM"}
        if "Tm" in group.columns:
            mask = group["Tm"].isin(multi_tags)
            if mask.any():
                return group[mask].iloc[0]

        # Fallback: row with maximum MP; handle all-Na MP safely
        if "MP" in group.columns:
            mp = pd.to_numeric(group["MP"], errors="coerce")
            if mp.notna().any():
                # Use positional index (iloc) to avoid KeyError with weird indices
                pos = mp.fillna(-1).values.argmax()
                return group.iloc[pos]

        # If no valid MP, just return the first row
        return group.iloc[0]

    adv_main = (
        adv
        .groupby(["Season", "Player"], as_index=False)
        .apply(pick_main_row)
        .reset_index(drop=True)
    )

    # ---- Load LEBRON data ----
    leb = pd.read_csv(lebron_file)
    # Columns: ENTRY,Season,Player,Age,Team(s),LEBRON WAR,LEBRON,O-LEBRON,D-LEBRON
    leb["Player"] = leb["Player"].astype(str).str.strip()
    leb["Season"] = leb["Season"].astype(str).str.strip()

    # Keep only needed columns
    leb = leb[["Season", "Player", "LEBRON WAR", "LEBRON"]]

    # ---- Merge both ----
    players = adv_main.merge(
        leb,
        on=["Season", "Player"],
        how="left"
    )

    # Fill missing impact metrics with 0
    for col in ["WS/48", "BPM", "VORP", "LEBRON WAR", "LEBRON"]:
        if col not in players.columns:
            players[col] = 0.0
        players[col] = pd.to_numeric(players[col], errors="coerce").fillna(0.0)

    # ---- Scale each metric 0–1 across all seasons/players ----
    metrics = ["WS/48", "BPM", "VORP", "LEBRON", "LEBRON WAR"]
    for m in metrics:
        m_min = players[m].min()
        m_max = players[m].max()
        if m_max > m_min:
            players[m + "_scaled"] = (players[m] - m_min) / (m_max - m_min)
        else:
            players[m + "_scaled"] = 0.0

    # ---- Compute StarScore (simple average, each 0.2 weight) ----
    players["StarScore"] = players[
        [m + "_scaled" for m in metrics]
    ].mean(axis=1)

    print(f"✓ Built StarScore for {len(players)} season-player rows")

    # We only need Season, Player, StarScore (and maybe MP for debugging)
    star_df = players[["Season", "Player", "StarScore", "MP"]].copy()

    return star_df


# ============================================
# 1. ROSTER CONTINUITY (STAR-WEIGHTED)
# ============================================

def scrape_roster_continuity(start_year=2015, end_year=2025, star_df=None):
    """
    Scrape roster continuity data (% of minutes returning),
    and add star-weighted continuity using StarScore.

    Uses Basketball-Reference for historical seasons.
    For the 2025-26 season (year == 2026), uses nba_roster_2025_26_bgm.csv
    which contains opening-night rosters exported from Basketball-GM.
    """

    print("=" * 80)
    print("SCRAPING ROSTER CONTINUITY (WITH STAR WEIGHTING)")
    print("=" * 80)

    if star_df is None:
        raise ValueError("star_df (player StarScore data) must be provided")

    # Ensure clean keys
    star_df = star_df.copy()
    star_df["Player"] = star_df["Player"].astype(str).str.strip()
    star_df["Season"] = star_df["Season"].astype(str).str.strip()

    continuity_data = []
    bgm_roster_df = None  # Only used for 2025-26

    for year in range(start_year + 1, end_year + 1):
        # "season" is the upcoming season label, e.g. "2015-16"
        season = f"{year - 1}-{str(year)[-2:]}"
        # Previous season label, e.g. "2014-15"
        prev_season = f"{year - 2}-{str(year - 1)[-2:]}"
        print(f"\nProcessing {season} (previous season = {prev_season})...")

        for team_name, abbrev in TEAM_ABBREVS.items():
            try:
                # -------------------------------
                # 1) Previous season roster (Basketball-Reference)
                # -------------------------------
                url_prev = f"https://www.basketball-reference.com/teams/{abbrev}/{year-1}.html"
                response_prev = requests.get(url_prev, headers={'User-Agent': 'Mozilla/5.0'})
                time.sleep(2)

                if response_prev.status_code != 200:
                    print(f"  ⚠️  Failed to load prev roster for {team_name} {year-1}")
                    continue

                tables_prev = pd.read_html(response_prev.content)
                roster_prev = None

                for table in tables_prev:
                    if 'Player' in table.columns and 'MP' in table.columns:
                        roster_prev = table
                        break

                if roster_prev is None:
                    print(f"  ⚠️  No roster table for {team_name} {year-1}")
                    continue

                roster_prev = roster_prev[roster_prev['Player'] != 'Player']
                roster_prev['Player'] = roster_prev['Player'].astype(str).str.strip()

                prev_players = {}
                for _, row in roster_prev.iterrows():
                    player = str(row['Player']).strip()
                    try:
                        minutes = float(row['MP'])
                        prev_players[player] = minutes
                    except Exception:
                        continue

                if not prev_players:
                    print(f"  ⚠️  No valid minutes for {team_name} {year-1}")
                    continue

                # Turn into DataFrame
                prev_df = pd.DataFrame([
                    {"Player": p, "Minutes": m} for p, m in prev_players.items()
                ])

                # Merge in StarScore from previous season
                star_prev = star_df[star_df["Season"] == prev_season][["Player", "StarScore"]]
                prev_df = prev_df.merge(star_prev, on="Player", how="left").fillna({"StarScore": 0.0})

                # Keep ONLY top 8 players by StarScore, tie-breaker minutes
                prev_df = prev_df.sort_values(["StarScore", "Minutes"], ascending=False)
                prev_df_top8 = prev_df.head(8).reset_index(drop=True)

                total_prev_minutes = prev_df_top8["Minutes"].sum()
                prev_df_top8["StarWeight"] = prev_df_top8["Minutes"] * (1.0 + prev_df_top8["StarScore"])
                total_prev_star_minutes = prev_df_top8["StarWeight"].sum()

                if total_prev_minutes == 0:
                    print(f"  ⚠️  Zero total prev minutes for {team_name} {prev_season}")
                    continue

                # -------------------------------
                # 2) Current season roster
                # -------------------------------
                if year == 2026:
                    # Use BGM CSV: nba_roster_2025_26_bgm.csv
                    if bgm_roster_df is None:
                        bgm_roster_df = pd.read_csv("nba_roster_2025_26_bgm.csv")
                        bgm_roster_df["Player"] = bgm_roster_df["Player"].astype(str).str.strip()
                        bgm_roster_df["Team"] = bgm_roster_df["Team"].astype(str).str.strip()

                    filter_team_name = team_name
                    if team_name == "Los Angeles Clippers":
                        # In BGM file you used "LA Clippers"
                        filter_team_name = "LA Clippers"

                    curr_roster_list = bgm_roster_df[
                        bgm_roster_df["Team"] == filter_team_name
                    ]["Player"].tolist()
                else:
                    # Scrape Basketball-Reference for current season
                    url_curr = f"https://www.basketball-reference.com/teams/{abbrev}/{year}.html"
                    response_curr = requests.get(url_curr, headers={'User-Agent': 'Mozilla/5.0'})
                    time.sleep()

                    if response_curr.status_code != 200:
                        print(f"  ⚠️  Failed to load current roster for {team_name} {year}")
                        continue

                    tables_curr = pd.read_html(response_curr.content)
                    roster_curr = None

                    for table in tables_curr:
                        if 'Player' in table.columns:
                            roster_curr = table
                            break

                    if roster_curr is None:
                        continue

                    roster_curr = roster_curr[roster_curr['Player'] != 'Player']
                    roster_curr['Player'] = roster_curr['Player'].astype(str).str.strip()
                    curr_roster_list = roster_curr['Player'].tolist()

                if not curr_roster_list:
                    print(f"  ⚠️  No current roster list for {team_name} {season}")
                    continue

                # -------------------------------
                # 3) Calculate continuity (plain + star-weighted)
                # -------------------------------
                returning_minutes = 0.0
                returning_star_minutes = 0.0
                new_players = 0

                for player in curr_roster_list:
                    row = prev_df_top8[prev_df_top8["Player"] == player]
                    if not row.empty:
                        mins = float(row["Minutes"].iloc[0])
                        star_weight = float(row["StarWeight"].iloc[0])
                        returning_minutes += mins
                        returning_star_minutes += star_weight
                    else:
                        new_players += 1

                plain_pct = (returning_minutes / total_prev_minutes * 100.0) if total_prev_minutes > 0 else 0.0
                if total_prev_star_minutes > 0:
                    star_pct = (returning_star_minutes / total_prev_star_minutes * 100.0)
                else:
                    star_pct = plain_pct

                continuity_data.append({
                    "Season": season,
                    "Team": team_name,
                    "Returning_Minutes_Pct": round(plain_pct, 2),
                    "Star_Weighted_Continuity": round(star_pct, 2),
                    "New_Players_Count": new_players,
                    "Total_Prev_Minutes": round(total_prev_minutes, 1)
                })

                print(f"  ✓ {team_name}: {plain_pct:.1f}% mins returning | {star_pct:.1f}% star-weighted")

            except Exception as e:
                print(f"  ✗ Error with {team_name} {season}: {e}")
                continue

    df = pd.DataFrame(continuity_data)
    df.to_csv('nba_roster_continuity.csv', index=False)
    print(f"\n✓ Saved: nba_roster_continuity.csv ({len(df)} records)")
    return df


# ============================================
# 3. INJURY DATA SCRAPER (STAR-WEIGHTED)
# ============================================

def scrape_injury_data(start_year=2015, end_year=2025, star_df=None):
    """
    Scrape injury data from previous seasons based on games played, and add
    star-weighted games missed for top-8 players (by minutes).

    Then incorporate 2025-26 preseason injuries from Injuries.csv and
    weight them by StarScore using previous season (2024-25).
    """
    print("\n" + "=" * 80)
    print("SCRAPING INJURY DATA (WITH STAR WEIGHTING)")
    print("=" * 80)

    if star_df is None:
        raise ValueError("star_df (player StarScore data) must be provided")

    star_df = star_df.copy()
    star_df["Player"] = star_df["Player"].astype(str).str.strip()
    star_df["Season"] = star_df["Season"].astype(str).str.strip()

    injury_data = []
    prev_top8_by_team_2024_25 = {}  # to reuse for 2025-26 preseason Injuries.csv

    for year in range(start_year, end_year + 1):
        season = f"{year - 1}-{str(year)[-2:]}"
        print(f"\nProcessing {season}...")

        for team_name, abbrev in TEAM_ABBREVS.items():
            try:
                url = f"https://www.basketball-reference.com/teams/{abbrev}/{year}.html"
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                time.sleep(2)

                if response.status_code != 200:
                    print(f"  ⚠️  Failed to load roster for {team_name} {year}")
                    continue

                tables = pd.read_html(response.content)
                roster = None

                for table in tables:
                    if 'Player' in table.columns and 'G' in table.columns:
                        roster = table
                        break

                if roster is None:
                    continue

                roster = roster[roster['Player'] != 'Player']
                roster['Player'] = roster['Player'].astype(str).str.strip()

                total_games = 82 if year < 2020 or year > 2021 else 72  # COVID seasons

                # Sort by minutes played
                roster["MP"] = pd.to_numeric(roster["MP"], errors="coerce")
                roster = roster.sort_values("MP", ascending=False)

                top_8_players = roster.head(8).copy()

                # Merge StarScore for current season (impact this year)
                stars_season = star_df[star_df["Season"] == season][["Player", "StarScore"]]
                top_8_players = top_8_players.merge(
                    stars_season, on="Player", how="left"
                ).fillna({"StarScore": 0.0})

                games_missed_top8_plain = 0.0
                games_missed_top8_star = 0.0
                injury_prone_count = 0

                # Count games missed & star-weighted missed for top-8
                for _, player_row in top_8_players.iterrows():
                    try:
                        gp = int(player_row["G"])
                    except Exception:
                        continue
                    games_missed = max(0, total_games - gp)
                    games_missed_top8_plain += games_missed
                    games_missed_top8_star += games_missed * (1.0 + float(player_row["StarScore"]))

                # Count injury-prone players (missed 20+ games) across full roster
                for _, player_row in roster.iterrows():
                    try:
                        gp = int(player_row["G"])
                    except Exception:
                        continue
                    if (total_games - gp) >= 20:
                        injury_prone_count += 1

                top_player_games = 0
                if len(top_8_players) > 0:
                    try:
                        top_player_games = int(top_8_players.iloc[0]["G"])
                    except Exception:
                        top_player_games = 0

                injury_data.append({
                    "Season": season,
                    "Team": team_name,
                    "Games_Missed_Top8": games_missed_top8_plain,
                    "Star_Weighted_Games_Missed_Top8": games_missed_top8_star,
                    "Injury_Prone_Count": injury_prone_count,
                    "Top_Player_Games": top_player_games
                })

                print(
                    f"  ✓ {team_name}: Top-8 missed {games_missed_top8_plain:.1f} games "
                    f"(star-weighted {games_missed_top8_star:.1f})"
                )

                # Save 2024-25 top-8 for later preseason (2025-26) injuries mapping
                if season == "2024-25":
                    prev_top8_by_team_2024_25[team_name] = top_8_players[
                        ["Player", "G", "MP", "StarScore"]
                    ].copy()

            except Exception as e:
                print(f"  ✗ Error with {team_name} {season}: {e}")
                continue

    # -----------------------------------------
    # Add 2025-26 preseason injuries from Injuries.csv
    # -----------------------------------------
    print("\n" + "=" * 80)
    print("ADDING 2025-26 PRESEASON INJURIES (STAR-WEIGHTED TOP-8)")
    print("=" * 80)

    try:
        inj = pd.read_csv("Injuries.csv")
        # Example columns: Name,Pos,Team,Age,Ovr,Pot,G,PTS,TRB,AST,Type,Games

        inj["Name"] = inj["Name"].astype(str).str.strip()
        inj["Team"] = inj["Team"].astype(str).str.strip()

        # Cap games at 82, handle missing
        inj["Games"] = pd.to_numeric(inj["Games"], errors="coerce").fillna(0).astype(int)
        inj["Games"] = inj["Games"].clip(upper=82)

        # Merge StarScore from previous season (2024-25)
        stars_2024_25 = star_df[star_df["Season"] == "2024-25"][["Player", "StarScore"]]
        inj = inj.merge(
            stars_2024_25,
            left_on="Name",
            right_on="Player",
            how="left"
        ).drop(columns=["Player"])
        inj["StarScore"] = inj["StarScore"].fillna(0.0)

        # Group by team abbreviation (e.g. BOS, DAL, etc.)
        preseason_rows = []
        for team_abbr in inj["Team"].unique():
            team_inj = inj[inj["Team"] == team_abbr]

            # Map abbreviation to full team name used elsewhere
            team_full = ABBR_TO_TEAM.get(team_abbr, team_abbr)
            top8_prev = prev_top8_by_team_2024_25.get(team_full)

            games_missed_top8_plain = 0.0
            games_missed_top8_star = 0.0
            injury_prone_count = 0

            top8_names = set()
            if top8_prev is not None:
                top8_names = set(top8_prev["Player"].astype(str).str.strip().tolist())

            # Loop injured players
            for _, row in team_inj.iterrows():
                name = row["Name"]
                games = int(row["Games"])
                star = float(row["StarScore"])
                injury_prone_count += 1  # any preseason injury counts

                if name in top8_names:
                    games_missed_top8_plain += games
                    games_missed_top8_star += games * (1.0 + star)

            preseason_rows.append({
                "Season": "2025-26",
                "Team": team_full,
                "Games_Missed_Top8": games_missed_top8_plain,
                "Star_Weighted_Games_Missed_Top8": games_missed_top8_star,
                "Injury_Prone_Count": injury_prone_count,
                "Top_Player_Games": 0  # unknown preseason
            })

            print(
                f"  ✓ Preseason {team_full}: Top-8 missed {games_missed_top8_plain:.1f} "
                f"(star-weighted {games_missed_top8_star:.1f}), {injury_prone_count} injured players"
            )

        injury_data.extend(preseason_rows)
        print(f"✓ Added preseason injuries for {len(preseason_rows)} teams")

    except FileNotFoundError:
        print("⚠️  Injuries.csv not found – skipping preseason 2025-26 injuries")
    except Exception as e:
        print(f"✗ Failed to import preseason injuries: {e}")

    df = pd.DataFrame(injury_data)
    df.to_csv('nba_injury_data.csv', index=False)
    print(f"\n✓ Saved: nba_injury_data.csv ({len(df)} records)")
    return df


# ============================================
# 5. MERGE ALL DATA SOURCES
# ============================================

def merge_all_data_sources(base_stats_file='nba_team_stats_2015_2025.csv'):
    """
    Merge all scraped data with base team stats.
    """
    print("\n" + "=" * 80)
    print("MERGING ALL DATA SOURCES INTO nba_complete_dataset.csv")
    print("=" * 80)

    df_base = pd.read_csv(base_stats_file)
    print(f"✓ Loaded base stats: {df_base.shape}")

    # Roster continuity
    try:
        df_roster = pd.read_csv('nba_roster_continuity.csv')
        df_base = df_base.merge(df_roster, on=['Season', 'Team'], how='left')
        print(f"✓ Merged roster continuity: {df_roster.shape}")
    except FileNotFoundError:
        print("⚠️  nba_roster_continuity.csv not found")

    # Injury data
    try:
        df_injury = pd.read_csv('nba_injury_data.csv')
        df_base = df_base.merge(df_injury, on=['Season', 'Team'], how='left')
        print(f"✓ Merged injury data: {df_injury.shape}")
    except FileNotFoundError:
        print("⚠️  nba_injury_data.csv not found")

    df_base.to_csv('nba_complete_dataset.csv', index=False)
    print(f"\n✓ Saved complete dataset: nba_complete_dataset.csv")
    print(f"  Final shape: {df_base.shape}")

    print("\n📊 DATA COMPLETENESS:")
    for col in df_base.columns:
        missing_pct = df_base[col].isnull().sum() / len(df_base) * 100
        if missing_pct > 0:
            print(f"  {col}: {missing_pct:.1f}% missing")

    return df_base


# ============================================
# 6. MAIN EXECUTION
# ============================================

def main():
    """
    Run star-weighted roster continuity + injuries scrapers.
    """
    print("\n" + "=" * 80)
    print("NBA ADDITIONAL DATA SCRAPER (STAR-WEIGHTED CONTINUITY & INJURIES)")
    print("=" * 80)
    print("\nThis will scrape:")
    print("  1. Star-weighted Roster Continuity (top-8)")
    print("  2. Star-weighted Injury Data (top-8)")
    print("\n⚠️  This will take time due to web scraping (Basketball-Reference)")
    print("=" * 80)

    proceed = input("\nProceed? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Cancelled.")
        return

    # Load player impact data once
    star_df = load_player_impact_data(
        adv_file="nba_advanced_stats_2015_2025.csv",
        lebron_file="nba_2014_2025_LEBRON.csv"
    )

    # 1. Roster continuity (up to 2025-26 season, i.e., year=2026)
    print("\n" + "=" * 80)
    print("STEP 1/2: ROSTER CONTINUITY")
    print("=" * 80)
    scrape_roster_continuity(start_year=2015, end_year=2026, star_df=star_df)

    # 2. Injury data (up to 2024-25 season + 2025-26 preseason)
    print("\n" + "=" * 80)
    print("STEP 2/2: INJURY DATA")
    print("=" * 80)
    scrape_injury_data(start_year=2015, end_year=2025, star_df=star_df)

    print("\n" + "=" * 80)
    print("SCRAPING COMPLETE!")
    print("=" * 80)
    print("\n📝 OPTIONAL NEXT STEP:")
    print("  → Run merge_all_data_sources('nba_team_stats_2015_2025.csv')")
    print("    to produce nba_complete_dataset.csv with all features merged.")
    print("=" * 80)


if __name__ == "__main__":
    main()
    # After this, if you want the combined dataset:
    # merge_all_data_sources('nba_team_stats_2015_2025.csv')
