import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def load_data(csv_name):
    df_load = pd.read_csv("data/" + csv_name)

    inj = pd.read_csv("data/nba_injury_data.csv")
    df_load = df_load.merge(inj, on=["Season","Team"], how="left")

    rc = pd.read_csv("data/nba_roster_continuity.csv")
    df_load = df_load.merge(rc, on=["Season","Team"], how="left")

    return df_load


def create_rows(df_raw):
    prev_season = df_raw["Season"].max()
    tm_list = df_raw[df_raw["Season"]==prev_season]["Team"].unique()

    temp_rows = []
    for t in tm_list:
        temp_rows.append({"Season":"2025-26","Team":t})

    df_new = pd.concat([df_raw, pd.DataFrame(temp_rows)], ignore_index=True)
    df_new = df_new.sort_values(["Team","Season"]).reset_index(drop=True)
    return df_new


def lagged_features(df_raw):
    df_raw = df_raw.sort_values(["Team","Season"]).reset_index(drop=True)

    feats = ["PG_Team_PTS","PG_Opp_PTS","PG_Team_FG%",
        "PG_Team_3P%","PG_Team_TRB","PG_Team_AST","PG_Team_STL","PG_Team_TOV","PG_Opp_FG%",
        "PG_Opp_3P%","PG_Opp_TOV","Adv_W","Adv_MOV","Adv_SRS","Adv_ORtg","Adv_DRtg","Adv_NRtg","Adv_TS%",
        "Adv_3PAr","Adv_Offense Four Factors_eFG%",
        "Adv_Offense Four Factors_TOV%",
        "Adv_Defense Four Factors_eFG%","Adv_Defense Four Factors_DRB%"
    ]
    feats.extend(["Games_Missed_Top8","Star_Weighted_Games_Missed_Top8","Injury_Prone_Count","Top_Player_Games"])
    feats.extend(["Returning_Minutes_Pct","Star_Weighted_Continuity","New_Players_Count"])

    for c in feats:
        if c in df_raw.columns:
            df_raw["Prev_"+c] = df_raw.groupby("Team")[c].shift(1)

    klist = ["Adv_W","BR_ORtg_A","BR_DRtg_A","BR_NRtg_A","Adv_MOV","Adv_SRS"]
    for c in klist:
        if c in df_raw.columns:
            df_raw["Avg2Y_"+c] = (df_raw.groupby("Team")[c].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean()))

    if "Vegas_OU" in df_raw.columns:
        df_raw["Prev_Vegas_OU"] = df_raw.groupby("Team")["Vegas_OU"].shift(1)

    df_raw["Target_Wins"] = df_raw["Adv_W"]
    return df_raw


def engineer_features(df_raw):
    xval = 14
    p_for = df_raw["Prev_PG_Team_PTS"]
    p_against = df_raw["Prev_PG_Opp_PTS"]

    df_raw["Prev_Pythag_WinPct"] = (p_for**xval) / ((p_for**xval)+(p_against**xval))
    df_raw["Prev_Pythag_Exp_Wins"] = df_raw["Prev_Pythag_WinPct"] * 82
    df_raw["Prev_Pythag_WinDiff"] = df_raw["Prev_Pythag_Exp_Wins"] - df_raw["Prev_Adv_W"]
    df_raw["Prev_Pythag_WinPct"] = np.nan
    df_raw["Prev_Pythag_Exp_Wins"] = np.nan
    df_raw["Prev_Pythag_WinDiff"] = np.nan


    df_raw["Prev_SOS"] = df_raw["Prev_Adv_SRS"] - df_raw["Prev_Adv_MOV"]
    df_raw["SOS_Effect"] = df_raw["BR_NRtg_A"] + df_raw["Prev_SOS"]
    df_raw["Prev_SOS"] = np.nan
    df_raw["SOS_Effect"] = np.nan


    if "Prev_Games_Missed_Top8" in df_raw.columns:
        maxg = 8*82
        df_raw["Health_Score"] = maxg - df_raw["Prev_Games_Missed_Top8"]
        df_raw["Low_Injury"] = df_raw["BR_NRtg_A"] * (df_raw["Health_Score"]/maxg)

    if "Prev_Returning_Minutes_Pct" in df_raw.columns:
        pct = df_raw["Prev_Returning_Minutes_Pct"] / 100.0
        df_raw["High_Continuity"] = df_raw["BR_NRtg_A"] * pct


    pct_cols = ["BR_ORtg_A","BR_DRtg_A","BR_NRtg_A","Prev_Adv_SRS","Prev_Adv_MOV","Prev_PG_Team_PTS","Prev_PG_Opp_PTS","Prev_Pythag_Exp_Wins","Prev_Adv_Offense Four Factors_eFG%","Prev_Adv_Defense Four Factors_eFG%"]

    for c in pct_cols:
        if c in df_raw.columns:
            df_raw[c+"_pct"] = df_raw.groupby("Season")[c].rank(pct=True)
            df_raw[c+"_pct_delta"] = df_raw.groupby("Team")[c+"_pct"].diff()

    return df_raw


def train_data(df_raw, pred_season):
    all_feats = []
    
    for col in df_raw.columns:
        if (col.startswith("Prev_") or
            col.startswith("Avg") or
            col.endswith("_pct") or
            col.endswith("_pct_delta") or
            col in ["OffDef_Ratio","Prev_Point_Diff","Prev_eFG_Diff","Prev_Pythag_WinPct","Prev_Pythag_Exp_Wins",
                    "Prev_Pythag_WinDiff","Prev_SOS","SOS_Effect","Health_Score","Low_Injury","High_Continuity"]):
            if col not in ["Prev_Adv_W","Prev_Adv_L"]:
                all_feats.append(col)

    all_feats = [c for c in all_feats if "Vegas" not in c]
    all_feats = [c for c in all_feats if df_raw[c].notna().sum() > 50]

    season_window = ["2022-23","2023-24","2024-25"]
    hist = df_raw[df_raw["Season"].isin(season_window) & df_raw["Target_Wins"].notna()& df_raw["Vegas_OU"].notna() & df_raw["Vegas_Error_Target"].notna()].copy()

    pred = df_raw[df_raw["Season"] == pred_season].copy()

    return hist, pred, all_feats