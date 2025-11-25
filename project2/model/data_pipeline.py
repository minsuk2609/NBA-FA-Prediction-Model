import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def load_data(csv_file):
    df = pd.read_csv(f"data/{csv_file}")

    try:
        injury_df = pd.read_csv("data/nba_injury_data.csv")
        df = df.merge(injury_df, on=['Season', 'Team'], how='left')
    except Exception:
        print("Injury data not found, proceeding without it.")

    try:
        roster_df = pd.read_csv("data/nba_roster_continuity.csv")
        df = df.merge(roster_df, on=['Season', 'Team'], how='left')
    except Exception:
        print("Roster continuity data not found, proceeding without it.")

    return df


def create_rows(df):
    latest_season = df['Season'].max()
    teams = df[df['Season'] == latest_season]['Team'].unique()
    df_2025_26 = pd.DataFrame([{'Season': '2025-26', 'Team': t} for t in teams])
    df = pd.concat([df, df_2025_26], ignore_index=True)
    return df.sort_values(['Team', 'Season']).reset_index(drop=True)


def lagged_features(df):
    df = df.sort_values(['Team', 'Season']).reset_index(drop=True)

    base_features = [
        'PG_Team_PTS', 'PG_Opp_PTS', 'PG_Team_FG%', 'PG_Team_3P%', 'PG_Team_TRB',
        'PG_Team_AST', 'PG_Team_STL', 'PG_Team_TOV', 'PG_Opp_FG%', 'PG_Opp_3P%',
        'PG_Opp_TOV', 'Adv_W', 'Adv_MOV', 'Adv_SRS', 'Adv_ORtg', 'Adv_DRtg',
        'Adv_NRtg', 'Adv_TS%', 'Adv_3PAr', 'Adv_Offense Four Factors_eFG%',
        'Adv_Offense Four Factors_TOV%', 'Adv_Defense Four Factors_eFG%',
        'Adv_Defense Four Factors_DRB%',
    ]

    if 'Games_Missed_Top8' in df.columns:
        base_features += [
            'Games_Missed_Top8', 'Star_Weighted_Games_Missed_Top8',
            'Injury_Prone_Count', 'Top_Player_Games'
        ]

    if 'Returning_Minutes_Pct' in df.columns:
        base_features += [
            'Returning_Minutes_Pct', 'Star_Weighted_Continuity',
            'New_Players_Count'
        ]

    for col in base_features:
        if col in df.columns:
            df[f'Prev_{col}'] = df.groupby('Team')[col].shift(1)

    key_cols = ['Adv_W', 'BR_ORtg_A', 'BR_DRtg_A', 'BR_NRtg_A', 'Adv_MOV', 'Adv_SRS']
    for col in key_cols:
        if col in df.columns:
            df[f'Avg2Y_{col}'] = (
                df.groupby('Team')[col]
                .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
            )

    if 'Vegas_OU' in df.columns:
        df['Prev_Vegas_OU'] = df.groupby('Team')['Vegas_OU'].shift(1)

    df['Target_Wins'] = df['Adv_W']
    return df


def engineer_features(df):
    try:
        x = 14
        pf = df['Prev_PG_Team_PTS']
        pa = df['Prev_PG_Opp_PTS']
        df['Prev_Pythag_WinPct'] = (pf ** x) / ((pf ** x) + (pa ** x))
        df['Prev_Pythag_Exp_Wins'] = df['Prev_Pythag_WinPct'] * 82
        df['Prev_Pythag_WinDiff'] = df['Prev_Pythag_Exp_Wins'] - df['Prev_Adv_W']
    except Exception:
        df['Prev_Pythag_WinPct'] = np.nan
        df['Prev_Pythag_Exp_Wins'] = np.nan
        df['Prev_Pythag_WinDiff'] = np.nan

    try:
        df['Prev_SOS'] = df['Prev_Adv_SRS'] - df['Prev_Adv_MOV']
        df['SOS_Effect'] = df['BR_NRtg_A'] + df['Prev_SOS']
    except Exception:
        df['Prev_SOS'] = np.nan
        df['SOS_Effect'] = np.nan

    if 'Prev_Games_Missed_Top8' in df.columns:
        max_games = 8 * 82
        df['Health_Score'] = max_games - df['Prev_Games_Missed_Top8']
        df['Low_Injury'] = df['BR_NRtg_A'] * (df['Health_Score'] / max_games)

    if 'Prev_Returning_Minutes_Pct' in df.columns:
        pct = df['Prev_Returning_Minutes_Pct'] / 100.0
        df['High_Continuity'] = df['BR_NRtg_A'] * pct

    percentile_cols = [
        'BR_ORtg_A', 'BR_DRtg_A', 'BR_NRtg_A', 'Prev_Adv_SRS', 'Prev_Adv_MOV',
        'Prev_PG_Team_PTS', 'Prev_PG_Opp_PTS', 'Prev_Pythag_Exp_Wins',
        'Prev_Adv_Offense Four Factors_eFG%',
        'Prev_Adv_Defense Four Factors_eFG%'
    ]

    for col in percentile_cols:
        if col in df.columns:
            df[f'{col}_pct'] = df.groupby('Season')[col].rank(pct=True)
            df[f'{col}_pct_delta'] = df.groupby('Team')[f'{col}_pct'].diff()

    return df


def train_data(df, prediction_season):
    feature_cols = [
        col for col in df.columns
        if (
            col.startswith('Prev_')
            or col.startswith('Avg')
            or col.endswith('_pct')
            or col.endswith('_pct_delta')
            or col in [
                'OffDef_Ratio', 'Prev_Point_Diff', 'Prev_eFG_Diff',
                'Prev_Pythag_WinPct', 'Prev_Pythag_Exp_Wins',
                'Prev_Pythag_WinDiff', 'Prev_SOS', 'SOS_Effect',
                'Health_Score', 'Low_Injury', 'High_Continuity'
            ]
        )
        and col not in ['Prev_Adv_W', 'Prev_Adv_L']
    ]

    feature_cols = [c for c in feature_cols if 'Vegas' not in c]
    feature_cols = [c for c in feature_cols if df[c].notna().sum() > 50]

    allowed = ['2022-23', '2023-24', '2024-25']

    historical = df[
        df['Season'].isin(allowed)
        & df['Target_Wins'].notna()
        & df['Vegas_OU'].notna()
        & df['Vegas_Error_Target'].notna()
    ].copy()

    prediction = df[df['Season'] == prediction_season].copy()

    return historical, prediction, feature_cols
