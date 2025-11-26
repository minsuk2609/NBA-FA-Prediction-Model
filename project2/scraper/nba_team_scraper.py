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
            df = extract_table(soup, table_id)

            if df is not None:
                if 'Team' not in df.columns:
                    print(f"No team column: {df.columns.tolist()}")
                    continue
                df = add_prefix_to_columns(df, prefix, exclude=['Team'])
                season_tables[table_id] = df

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

    df = pd.read_html(StringIO(str(table)))[0]
    df = clean_dataframe(df)
    return df


def clean_dataframe(df):
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            parts = [str(x) for x in col if str(x) != 'nan' and 'Unnamed' not in str(x)]
            new_cols.append('_'.join(parts) if parts else '')
        df.columns = new_cols

    df.columns = [str(col).strip() for col in df.columns]

    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str)
        df = df[df["Team"] != "Team"]
        df = df[df["Team"] != ""]
        df = df[df["Team"] != "nan"]
        df["Team"] = df["Team"].str.replace("*", "", regex=False).str.strip()

    drop_cols = [c for c in df.columns if c == "Rk" or c.startswith("Unnamed")]
    df = df.drop(drop_cols, axis=1, errors="ignore")

    for col in df.columns:
        if col == "Team":
            continue
        if not isinstance(df[col], pd.Series):
            continue

        if df[col].dtype == 'O' and df[col].apply(lambda x: isinstance(x, (dict, list, pd.DataFrame))).any():
            continue

        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.reset_index(drop=True)
    return df



def add_prefix_to_columns(df, prefix, exclude=None):
    if exclude is None:
        exclude = []

    new_columns = {
        col: (prefix + col if col not in exclude else col)
        for col in df.columns
    }
    return df.rename(columns=new_columns)


def merge_season_tables(tables_dict, year):
    merged_df = None

    for df in tables_dict.values():
        if df is not None and not df.empty:
            if 'Team' not in df.columns:
                continue

            merged_df = df.copy() if merged_df is None else pd.merge(
                merged_df, df, on='Team', how='outer'
            )

    if merged_df is not None:
        merged_df.insert(0, 'Season', f"{year-1}-{str(year)[-2:]}")
        cols = merged_df.columns.tolist()
        cols.remove('Team')
        cols.insert(1, 'Team')
        merged_df = merged_df[cols]

    return merged_df

if __name__ == "__main__":
    df = scrape_basketball_reference_seasons(start_year=2018, end_year=2025)
    if df is not None:
        try:
            df.to_csv("../data/nba_team_stats_2020_2025.csv", index=False)
        except Exception as e:
            print(f"Error saving file: {e}")
