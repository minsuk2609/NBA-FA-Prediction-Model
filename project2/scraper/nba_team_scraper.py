import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import time
from io import StringIO


def scrape_basketball_reference_seasons(start_year, end_year):
    all_seasons_data = []

    for year in range(start_year, end_year + 1):
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}.html"

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        season_tables = {}

        table_configs = [
            ('per_game-team', 'PG_Team_'),
            ('per_game-opponent', 'PG_Opp_'),
            ('totals-team', 'Tot_Team_'),
            ('totals-opponent', 'Tot_Opp_'),
            ('advanced-team', 'Adv_'),
            ('shooting-team', 'Shoot_Team_'),
            ('shooting-opponent', 'Shoot_Opp_'),
        ]

        for table_id, prefix in table_configs:
            df_team = extract_table(soup, table_id)

            if df_team is not None:
                if 'Team' not in df_team.columns:
                    print(f"No team column: {df_team.columns.tolist()}")
                    continue
                df_team = add_prefix_to_columns(df_team, prefix, exclude=['Team'])
                season_tables[table_id] = df_team

        if season_tables:
            season_df = merge_season_tables(season_tables, year)
            if season_df is not None:
                all_seasons_data.append(season_df)

        time.sleep(1.5)

    if all_seasons_data:
        return pd.concat(all_seasons_data, ignore_index=True)
    else:
        print("Error combining season stats")
        return None


def extract_table(soup, table_id):
    table = soup.find('table', {'id': table_id})

    if table is None:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            if table_id in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table', {'id': table_id})
                if table:
                    break

    if table is None:
        return None

    df_team = pd.read_html(StringIO(str(table)))[0]
    df_team = clean_dataframe(df_team)
    return df_team


def clean_dataframe(df_team):
    if isinstance(df_team.columns, pd.MultiIndex):
        new_cols = []
        for col in df_team.columns:
            parts = [str(x) for x in col if str(x) != 'nan' and 'Unnamed' not in str(x)]
            new_cols.append('_'.join(parts) if parts else '')
        df_team.columns = new_cols

    df_team.columns = [str(col).strip() for col in df_team.columns]

    if "Team" in df_team.columns:
        df_team["Team"] = df_team["Team"].astype(str)
        df_team = df_team[df_team["Team"] != "Team"]
        df_team = df_team[df_team["Team"] != ""]
        df_team = df_team[df_team["Team"] != "nan"]
        df_team["Team"] = df_team["Team"].str.replace("*", "", regex=False).str.strip()

    drops = [c for c in df_team.columns if c == "Rk" or c.startswith("Unnamed")]
    df_team = df_team.drop(drops, axis=1, errors="ignore")

    for col in df_team.columns:
        if col == "Team":
            continue
        if not isinstance(df_team[col], pd.Series):
            continue

        if df_team[col].dtype == 'O' and df_team[col].apply(lambda x: isinstance(x, (dict, list, pd.DataFrame))).any():
            continue

        df_team[col] = pd.to_numeric(df_team[col], errors='coerce')

    df_team = df_team.reset_index(drop=True)
    return df_team



def add_prefix_to_columns(df_team, prefix, exclude=None):
    if exclude is None:
        exclude = []

    new_columns = {
        col: (prefix + col if col not in exclude else col)
        for col in df_team.columns
    }
    return df_team.rename(columns=new_columns)


def merge_season_tables(tables_dict, year):
    merged_df = None

    for df_team in tables_dict.values():
        if df_team is not None and not df_team.empty:
            if 'Team' not in df_team.columns:
                continue

            merged_df = df_team.copy() if merged_df is None else pd.merge(
                merged_df, df_team, on='Team', how='outer'
            )

    if merged_df is not None:
        merged_df.insert(0, 'Season', f"{year-1}-{str(year)[-2:]}")
        cols = merged_df.columns.tolist()
        cols.remove('Team')
        cols.insert(1, 'Team')
        merged_df = merged_df[cols]

    return merged_df

if __name__ == "__main__":
    df_team = scrape_basketball_reference_seasons(start_year=2018, end_year=2025)
    if df_team is not None:
        try:
            df_team.to_csv("../data/nba_team_stats_2020_2025.csv", index=False)
        except Exception as e:
            print(f"Error saving file: {e}")
