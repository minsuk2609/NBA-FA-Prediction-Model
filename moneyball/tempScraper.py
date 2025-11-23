import pandas as pd
import numpy as np

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
    adv_raw = pd.read_csv(adv_file)

    # Expected columns: Season, Player, Tm, MP, WS/48, BPM, VORP
    for col in ["Season", "Player", "Tm"]:
        adv_raw[col] = adv_raw[col].astype(str).str.strip()

    for col in ["MP", "WS/48", "BPM", "VORP"]:
        if col in adv_raw.columns:
            adv_raw[col] = pd.to_numeric(adv_raw[col], errors="coerce").fillna(0.0)
        else:
            adv_raw[col] = 0.0

    # Aggregate per (Season, Player)
    adv = (
        adv_raw
        .groupby(["Season", "Player"], as_index=False)
        .agg({
            "MP": "sum",
            "WS/48": "mean",
            "BPM": "mean",
            "VORP": "sum"
        })
    )

    # ---- Load LEBRON data ----
    leb = pd.read_csv(lebron_file)
    # Columns: ENTRY,Season,Player,Age,Team(s),LEBRON WAR,LEBRON,O-LEBRON,D-LEBRON
    leb["Player"] = leb["Player"].astype(str).str.strip()
    leb["Season"] = leb["Season"].astype(str).str.strip()

    leb = leb[["Season", "Player", "LEBRON WAR", "LEBRON"]]

    # ---- Merge both ----
    players = adv.merge(leb, on=["Season", "Player"], how="left")

    # Fill missing metrics with 0
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
    players["StarScore"] = players[[m + "_scaled" for m in metrics]].mean(axis=1)

    print(f"✓ Built StarScore for {len(players)} season-player rows")

    # Only keep what we need
    star_df = players[["Season", "Player", "StarScore"]].copy()
    return star_df


# ============================================
# Helper: Build seasons list from advanced stats
# ============================================

def get_sorted_seasons(adv_raw: pd.DataFrame):
    seasons = sorted(
        adv_raw["Season"].astype(str).unique(),
        key=lambda s: int(s.split("-")[0])
    )
    return seasons


# ============================================
# Helper: Build top-8 previous-season players per team
# ============================================

def build_top8_by_team_for_season(adv_raw, star_df, season_label):
    """
    Returns dict:
      key: team_abbr (e.g. 'BOS')
      value: DataFrame with columns [Player, Minutes, StarScore, StarWeight]
    """
    star_season = star_df[star_df["Season"] == season_label][["Player", "StarScore"]].copy()
    star_season["Player"] = star_season["Player"].astype(str).str.strip()

    top8_dict = {}

    for team_abbr, team_full in ABBR_TO_TEAM.items():
        team_adv = adv_raw[
            (adv_raw["Season"] == season_label) &
            (adv_raw["Tm"] == team_abbr)
        ].copy()

        if team_adv.empty:
            continue

        team_adv["Player"] = team_adv["Player"].astype(str).str.strip()
        # Sum MP per player on that team
        prev_df = (
            team_adv
            .groupby("Player", as_index=False)["MP"]
            .sum()
            .rename(columns={"MP": "Minutes"})
        )

        prev_df = prev_df.merge(star_season, on="Player", how="left").fillna({"StarScore": 0.0})
        prev_df = prev_df.sort_values(["StarScore", "Minutes"], ascending=False)

        prev_df_top8 = prev_df.head(8).copy().reset_index(drop=True)
        prev_df_top8["StarWeight"] = prev_df_top8["Minutes"] * (1.0 + prev_df_top8["StarScore"])

        top8_dict[team_abbr] = prev_df_top8

    return top8_dict


# ============================================
# 1. ROSTER CONTINUITY (STAR-WEIGHTED, NO SCRAPING)
# ============================================

def scrape_roster_continuity_from_files(
    adv_file="nba_advanced_stats_2015_2025.csv",
    bgm_roster_file="nba_roster_2025_26_bgm.csv",
    star_df=None
):
    """
    Compute roster continuity (plain & star-weighted) using ONLY local files:

    - nba_advanced_stats_2015_2025.csv  (Season, Player, Tm, MP, WS/48, BPM, VORP)
    - nba_2014_2025_LEBRON.csv          (already baked into star_df)
    - nba_roster_2025_26_bgm.csv        (Player, Team) for 2025-26

    For each pair of consecutive seasons in the advanced stats file:
      - previous season: define top-8 per team by StarScore & minutes
      - current season: define roster from advanced stats (except 2025-26)
      - 2025-26: define roster from BGM CSV
    """
    if star_df is None:
        raise ValueError("star_df (player StarScore data) must be provided")

    print("=" * 80)
    print("COMPUTING ROSTER CONTINUITY (LOCAL CSV ONLY)")
    print("=" * 80)

    adv_raw = pd.read_csv(adv_file)
    for col in ["Season", "Player", "Tm"]:
        adv_raw[col] = adv_raw[col].astype(str).str.strip()
    adv_raw["MP"] = pd.to_numeric(adv_raw["MP"], errors="coerce").fillna(0.0)

    continuity_rows = []

    seasons = get_sorted_seasons(adv_raw)

    # ---- 1) Historical seasons continuity (within adv_raw) ----
    for i in range(1, len(seasons)):
        prev_season = seasons[i - 1]
        curr_season = seasons[i]

        print(f"\nProcessing continuity: {prev_season} -> {curr_season}")

        # Build top-8 from previous season
        prev_top8 = build_top8_by_team_for_season(adv_raw, star_df, prev_season)

        for team_full, team_abbr in TEAM_ABBREVS.items():
            prev_df_top8 = prev_top8.get(team_abbr)
            if prev_df_top8 is None or prev_df_top8.empty:
                continue

            total_prev_minutes = prev_df_top8["Minutes"].sum()
            total_prev_star_minutes = prev_df_top8["StarWeight"].sum()

            if total_prev_minutes <= 0:
                continue

            # Current season roster (from advanced stats)
            curr_team = adv_raw[
                (adv_raw["Season"] == curr_season) &
                (adv_raw["Tm"] == team_abbr)
            ].copy()

            curr_team["Player"] = curr_team["Player"].astype(str).str.strip()
            curr_roster_list = sorted(curr_team["Player"].unique().tolist())

            if not curr_roster_list:
                continue

            returning_minutes = 0.0
            returning_star_minutes = 0.0
            new_players = 0

            for p in curr_roster_list:
                row = prev_df_top8[prev_df_top8["Player"] == p]
                if not row.empty:
                    mins = float(row["Minutes"].iloc[0])
                    starw = float(row["StarWeight"].iloc[0])
                    returning_minutes += mins
                    returning_star_minutes += starw
                else:
                    new_players += 1

            plain_pct = (returning_minutes / total_prev_minutes * 100.0) if total_prev_minutes > 0 else 0.0
            star_pct = (returning_star_minutes / total_prev_star_minutes * 100.0) if total_prev_star_minutes > 0 else plain_pct

            continuity_rows.append({
                "Season": curr_season,
                "Team": team_full,
                "Returning_Minutes_Pct": round(plain_pct, 2),
                "Star_Weighted_Continuity": round(star_pct, 2),
                "New_Players_Count": new_players,
                "Total_Prev_Minutes": round(total_prev_minutes, 1)
            })

            print(f"  ✓ {team_full}: {plain_pct:.1f}% mins | {star_pct:.1f}% star-weighted")

    # ---- 2) Add 2025-26 continuity (prev season = 2024-25; roster from BGM CSV) ----
    # Only if we have 2024-25 in adv_raw
    if "2024-25" in adv_raw["Season"].unique():
        print("\nProcessing continuity: 2024-25 -> 2025-26 (using BGM roster)")
        prev_top8_2024_25 = build_top8_by_team_for_season(adv_raw, star_df, "2024-25")

        bgm = pd.read_csv(bgm_roster_file)
        bgm["Player"] = bgm["Player"].astype(str).str.strip()
        bgm["Team"] = bgm["Team"].astype(str).str.strip()

        for team_full, team_abbr in TEAM_ABBREVS.items():
            prev_df_top8 = prev_top8_2024_25.get(team_abbr)
            if prev_df_top8 is None or prev_df_top8.empty:
                continue

            total_prev_minutes = prev_df_top8["Minutes"].sum()
            total_prev_star_minutes = prev_df_top8["StarWeight"].sum()
            if total_prev_minutes <= 0:
                continue

            filter_team = team_full
            # Handle Clippers naming difference
            if team_full == "Los Angeles Clippers":
                filter_team = "LA Clippers"

            curr_roster_list = sorted(
                bgm[bgm["Team"] == filter_team]["Player"].unique().tolist()
            )

            if not curr_roster_list:
                continue

            returning_minutes = 0.0
            returning_star_minutes = 0.0
            new_players = 0

            for p in curr_roster_list:
                row = prev_df_top8[prev_df_top8["Player"] == p]
                if not row.empty:
                    mins = float(row["Minutes"].iloc[0])
                    starw = float(row["StarWeight"].iloc[0])
                    returning_minutes += mins
                    returning_star_minutes += starw
                else:
                    new_players += 1

            plain_pct = (returning_minutes / total_prev_minutes * 100.0) if total_prev_minutes > 0 else 0.0
            star_pct = (returning_star_minutes / total_prev_star_minutes * 100.0) if total_prev_star_minutes > 0 else plain_pct

            continuity_rows.append({
                "Season": "2025-26",
                "Team": team_full,
                "Returning_Minutes_Pct": round(plain_pct, 2),
                "Star_Weighted_Continuity": round(star_pct, 2),
                "New_Players_Count": new_players,
                "Total_Prev_Minutes": round(total_prev_minutes, 1)
            })

            print(f"  ✓ {team_full}: {plain_pct:.1f}% mins | {star_pct:.1f}% star-weighted")

    df_cont = pd.DataFrame(continuity_rows)
    df_cont.to_csv("nba_roster_continuity.csv", index=False)
    print(f"\n✓ Saved: nba_roster_continuity.csv ({len(df_cont)} records)")
    return df_cont


# ============================================
# 2. INJURY DATA (ADD 2025-26 PRESEASON, STAR-WEIGHTED)
# ============================================

def add_preseason_injuries_2025_26(
    adv_file="nba_advanced_stats_2015_2025.csv",
    star_df=None,
    injuries_file="Injuries.csv",
    existing_injury_file="nba_injury_data.csv"
):
    """
    DO NOT rescrape historical injuries.

    Instead:
      - Optionally load existing nba_injury_data.csv (historical)
      - Build 2024-25 top-8 per team (from adv_file + star_df)
      - Read Injuries.csv (preseason 2025-26 injured players with Games)
      - For each team:
          * if player was in 2024-25 top-8 -> contribute to Games_Missed_Top8
          * Star-weight that by StarScore
      - Append 2025-26 rows and save nba_injury_data.csv
    """
    if star_df is None:
        raise ValueError("star_df (player StarScore data) must be provided")

    print("\n" + "=" * 80)
    print("ADDING 2025-26 PRESEASON INJURIES (STAR-WEIGHTED) FROM LOCAL CSV")
    print("=" * 80)

    # Load adv_raw to define 2024-25 top-8
    adv_raw = pd.read_csv(adv_file)
    for col in ["Season", "Player", "Tm"]:
        adv_raw[col] = adv_raw[col].astype(str).str.strip()
    adv_raw["MP"] = pd.to_numeric(adv_raw["MP"], errors="coerce").fillna(0.0)

    if "2024-25" not in adv_raw["Season"].unique():
        print("⚠️  No 2024-25 season in advanced stats; cannot build top-8 for preseason injuries.")
        prev_top8_by_team = {}
    else:
        prev_top8_by_team = build_top8_by_team_for_season(adv_raw, star_df, "2024-25")

    # Load existing injury file if it exists
    try:
        injury_hist = pd.read_csv(existing_injury_file)
        print(f"✓ Loaded existing {existing_injury_file} with {len(injury_hist)} rows")
    except FileNotFoundError:
        print(f"⚠️  {existing_injury_file} not found; will create new file with only 2025-26 preseason injuries")
        injury_hist = pd.DataFrame(columns=[
            "Season", "Team", "Games_Missed_Top8",
            "Star_Weighted_Games_Missed_Top8",
            "Injury_Prone_Count", "Top_Player_Games"
        ])

    # Load Injuries.csv
    try:
        inj = pd.read_csv(injuries_file, on_bad_lines="skip")
    except FileNotFoundError:
        print(f"⚠️  {injuries_file} not found; skipping preseason 2025-26 injuries")
        injury_hist.to_csv(existing_injury_file, index=False)
        return injury_hist

    # Expect columns: Name,Pos,Team,Age,Ovr,Pot,G,PTS,TRB,AST,Type,Games
    inj["Name"] = inj["Name"].astype(str).str.strip()
    inj["Team"] = inj["Team"].astype(str).str.strip()
    inj["Games"] = pd.to_numeric(inj["Games"], errors="coerce").fillna(0).astype(int)
    inj["Games"] = inj["Games"].clip(upper=82)

    # Merge StarScore for 2024-25
    stars_2024_25 = star_df[star_df["Season"] == "2024-25"][["Player", "StarScore"]].copy()
    stars_2024_25["Player"] = stars_2024_25["Player"].astype(str).str.strip()

    inj = inj.merge(
        stars_2024_25,
        left_on="Name",
        right_on="Player",
        how="left"
    ).drop(columns=["Player"])
    inj["StarScore"] = inj["StarScore"].fillna(0.0)

    preseason_rows = []

    for team_abbr in inj["Team"].unique():
        team_inj = inj[inj["Team"] == team_abbr]

        team_full = ABBR_TO_TEAM.get(team_abbr, team_abbr)
        prev_top8 = prev_top8_by_team.get(team_abbr)

        games_missed_top8_plain = 0.0
        games_missed_top8_star = 0.0
        injury_prone_count = 0

        top8_names = set()
        if prev_top8 is not None:
            top8_names = set(prev_top8["Player"].astype(str).str.strip().tolist())

        for _, row in team_inj.iterrows():
            name = row["Name"]
            games = int(row["Games"])
            star = float(row["StarScore"])

            injury_prone_count += 1

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
            f"  ✓ {team_full}: Top-8 missed {games_missed_top8_plain:.1f} "
            f"(star-weighted {games_missed_top8_star:.1f}), injured players={injury_prone_count}"
        )

    injury_2025_26 = pd.DataFrame(preseason_rows)

    # Drop any existing 2025-26 rows to avoid duplicates then append
    injury_hist = injury_hist[injury_hist["Season"] != "2025-26"]
    injury_all = pd.concat([injury_hist, injury_2025_26], ignore_index=True)

    injury_all.to_csv(existing_injury_file, index=False)
    print(f"\n✓ Saved: {existing_injury_file} ({len(injury_all)} rows total)")
    return injury_all


# ============================================
# 3. MERGE ALL DATA SOURCES
# ============================================

def merge_all_data_sources(base_stats_file='nba_team_stats_2015_2025.csv'):
    """
    Merge all scraped data with base team stats into nba_complete_dataset.csv
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
# 4. MAIN EXECUTION
# ============================================

def main():
    """
    Run star-weighted roster continuity + add 2025-26 preseason injuries.
    No web scraping, only local CSVs.
    """
    print("\n" + "=" * 80)
    print("NBA ADDITIONAL DATA (STAR-WEIGHTED CONTINUITY & INJURIES) - LOCAL ONLY")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Build StarScore from advanced stats + LEBRON")
    print("  2. Compute star-weighted roster continuity (historical + 2025-26 via BGM)")
    print("  3. Add 2025-26 preseason injuries to nba_injury_data.csv (star-weighted)")
    print("=" * 80)

    proceed = input("\nProceed? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Cancelled.")
        return

    # 0. Load StarScore
    star_df = load_player_impact_data(
        adv_file="nba_advanced_stats_2015_2025.csv",
        lebron_file="nba_2014_2025_LEBRON.csv"
    )

    # 1. Roster continuity
    scrape_roster_continuity_from_files(
        adv_file="nba_advanced_stats_2015_2025.csv",
        bgm_roster_file="nba_roster_2025_26_bgm.csv",
        star_df=star_df
    )

    # 2. Add 2025-26 preseason injuries
    add_preseason_injuries_2025_26(
        adv_file="nba_advanced_stats_2015_2025.csv",
        star_df=star_df,
        injuries_file="Injuries.csv",
        existing_injury_file="nba_injury_data.csv"
    )

    print("\n" + "=" * 80)
    print("DONE. If you want, now run:")
    print("  merge_all_data_sources('nba_team_stats_2015_2025.csv')")
    print("to produce nba_complete_dataset.csv with all features merged.")
    print("=" * 80)


if __name__ == "__main__":
    main()
    # Optionally afterwards:
    # merge_all_data_sources('nba_team_stats_2015_2025.csv')
