import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 1. LOAD & MERGE DATA
# =====================================================================

def load_and_prepare_data(csv_file='nba_team_stats_2020_2025.csv'):
    df = pd.read_csv(csv_file)
    print("=" * 80)
    print("DATA LOADED")
    print("=" * 80)
    print(f"Shape: {df.shape}")

    # Merge injury data (optional)
    try:
        injury_df = pd.read_csv('nba_injury_data.csv')
        df = df.merge(injury_df, on=['Season', 'Team'], how='left')
        print("✓ Injury data merged")
    except Exception as e:
        print(f"⚠️  No injury data or merge failed: {e}")

    # Merge roster continuity (optional)
    try:
        roster_df = pd.read_csv('nba_roster_continuity.csv')
        df = df.merge(roster_df, on=['Season', 'Team'], how='left')
        print("✓ Roster continuity merged")
    except Exception as e:
        print(f"⚠️  No roster data or merge failed: {e}")

    return df

# =====================================================================
# 2. CREATE FUTURE SEASON ROWS (2025-26)
# =====================================================================

def create_2025_26_rows(df):
    latest_season = df['Season'].max()
    teams = df[df['Season'] == latest_season]['Team'].unique()

    rows_2025_26 = [{'Season': '2025-26', 'Team': team} for team in teams]
    df_2025_26 = pd.DataFrame(rows_2025_26)

    df_combined = pd.concat([df, df_2025_26], ignore_index=True)
    df_combined = df_combined.sort_values(['Team', 'Season']).reset_index(drop=True)

    print("\n2025-26 rows created")
    return df_combined

# =====================================================================
# 3. CREATE LAGGED FEATURES (NO VEGAS IN FEATURES)
# =====================================================================

def create_lagged_features(df):
    df = df.sort_values(['Team', 'Season']).reset_index(drop=True)

    # Core per-game & advanced features to lag
    feature_cols = [
        'PG_Team_PTS', 'PG_Opp_PTS', 'PG_Team_FG%', 'PG_Team_3P%',
        'PG_Team_TRB', 'PG_Team_AST', 'PG_Team_STL', 'PG_Team_TOV',
        'PG_Opp_FG%', 'PG_Opp_3P%', 'PG_Opp_TOV',
        'Adv_W', 'Adv_MOV', 'Adv_SRS',
        'Adv_ORtg', 'Adv_DRtg', 'Adv_NRtg',
        'Adv_TS%', 'Adv_3PAr',
        'Adv_Offense Four Factors_eFG%', 'Adv_Offense Four Factors_TOV%',
        'Adv_Defense Four Factors_eFG%', 'Adv_Defense Four Factors_DRB%',
    ]

    # Injury-related features (Top-8, star-weighted)
    if 'Games_Missed_Top8' in df.columns:
        feature_cols.extend([
            'Games_Missed_Top8',
            'Star_Weighted_Games_Missed_Top8',
            'Injury_Prone_Count',
            'Top_Player_Games'
        ])

    # Roster continuity features (if present)
    if 'Returning_Minutes_Pct' in df.columns:
        feature_cols.extend([
            'Returning_Minutes_Pct',
            'Star_Weighted_Continuity',
            'New_Players_Count'
        ])

    # Create lagged features by team
    for col in feature_cols:
        if col in df.columns:
            df[f'Prev_{col}'] = df.groupby('Team')[col].shift(1)

    # 2-year rolling averages for key advanced metrics
    key_cols = ['Adv_W', 'Adv_ORtg', 'Adv_DRtg', 'Adv_NRtg', 'Adv_MOV', 'Adv_SRS']
    for col in key_cols:
        if col in df.columns:
            df[f'Avg2Y_{col}'] = (
                df.groupby('Team')[col]
                  .transform(lambda s: s.shift(1).rolling(window=2, min_periods=1).mean())
            )

    # Optional: diagnostics only, NOT used as feature
    if 'Vegas_OU' in df.columns:
        df['Prev_Vegas_OU'] = df.groupby('Team')['Vegas_OU'].shift(1)

    # Target = actual wins for that season
    df['Target_Wins'] = df['Adv_W']

    print("Lagged features created")
    return df

# =====================================================================
# 4. ENGINEER HIGHER-LEVEL FEATURES (INCL. PERCENTILES)
# =====================================================================

def engineer_features(df):
    # Year-over-year trends
    df['Win_Change_1Y'] = df['Prev_Adv_W'] - df.groupby('Team')['Adv_W'].shift(2)
    df['NetRtg_Trend'] = df['Prev_Adv_NRtg'] - df.groupby('Team')['Adv_NRtg'].shift(2)

    # Basic differentials
    df['OffDef_Ratio'] = df['Prev_Adv_ORtg'] / df['Prev_Adv_DRtg'].replace(0, np.nan)
    df['Prev_Point_Diff'] = df['Prev_PG_Team_PTS'] - df['Prev_PG_Opp_PTS']
    df['Prev_eFG_Diff'] = (
        df['Prev_Adv_Offense Four Factors_eFG%'] -
        df['Prev_Adv_Defense Four Factors_eFG%']
    )

    # 1. Pythagorean expected wins
    try:
        x = 14  # modern NBA exponent
        pf = df['Prev_PG_Team_PTS']
        pa = df['Prev_PG_Opp_PTS']

        df['Prev_Pythag_WinPct'] = (pf ** x) / ((pf ** x) + (pa ** x))
        df['Prev_Pythag_Exp_Wins'] = df['Prev_Pythag_WinPct'] * 82
        df['Prev_Pythag_WinDiff'] = df['Prev_Pythag_Exp_Wins'] - df['Prev_Adv_W']
    except Exception:
        df['Prev_Pythag_WinPct'] = np.nan
        df['Prev_Pythag_Exp_Wins'] = np.nan
        df['Prev_Pythag_WinDiff'] = np.nan

    # 2. Strength of Schedule Effect (SOS)
    try:
        df['Prev_SOS'] = df['Prev_Adv_SRS'] - df['Prev_Adv_MOV']
        df['SOS_Trend'] = df['Prev_SOS'] - df.groupby('Team')['Adv_SRS'].shift(2)
        df['SOS_Effect'] = df['Prev_Adv_NRtg'] + df['Prev_SOS']
    except Exception:
        df['Prev_SOS'] = np.nan
        df['SOS_Trend'] = np.nan
        df['SOS_Effect'] = np.nan

    # Injury-related engineered features (Top-8 based)
    if 'Prev_Games_Missed_Top8' in df.columns:
        max_games_top8 = 8 * 82  # 8 players * 82 games
        df['Health_Score'] = max_games_top8 - df['Prev_Games_Missed_Top8']
        df['Health_Score'] = df['Health_Score'].clip(lower=0)
        df['Star_Healthy'] = (df['Prev_Top_Player_Games'] >= 60).astype(float)
        df['Healthy_Elite'] = df['Prev_Adv_NRtg'] * (df['Health_Score'] / max_games_top8)

    # Roster continuity features
    if 'Prev_Returning_Minutes_Pct' in df.columns:
        continuity_pct = df['Prev_Returning_Minutes_Pct'] / 100.0
        df['High_Continuity'] = (df['Prev_Returning_Minutes_Pct'] >= 70).astype(float)
        df['Major_Turnover'] = (df['Prev_Returning_Minutes_Pct'] < 50).astype(float)
        df['Elite_Continuity'] = df['Prev_Adv_NRtg'] * continuity_pct
        df['Continuity_Adjusted_Wins'] = (
            df['Prev_Adv_W'] * continuity_pct + 41 * (1 - continuity_pct)
        )

    # Stats that vary heavily by season -> percentile normalize
    percentile_cols = [
        'Prev_Adv_ORtg',
        'Prev_Adv_DRtg',
        'Prev_Adv_NRtg',
        'Prev_Adv_SRS',
        'Prev_Adv_MOV',
        'Prev_PG_Team_PTS',
        'Prev_PG_Opp_PTS',
        'Prev_Pythag_Exp_Wins',
        'Prev_Adv_Offense Four Factors_eFG%',
        'Prev_Adv_Defense Four Factors_eFG%'
    ]

    for col in percentile_cols:
        if col in df.columns:
            # Percentile rank *within each season* (0–1)
            df[f'{col}_pct'] = df.groupby('Season')[col].rank(pct=True)
            # YoY change in percentile for each team
            df[f'{col}_pct_delta'] = df.groupby('Team')[f'{col}_pct'].diff()

    print("Features engineered")
    return df

# =====================================================================
# 5. PREPARE FEATURE LIST (EXCLUDING VEGAS FROM FEATURES)
# =====================================================================

def prepare_train_test_data(df, prediction_season='2025-26'):
    # Base feature selection:
    feature_cols = [
        col for col in df.columns
        if (
            col.startswith('Prev_') or
            col.startswith('Avg') or
            col.endswith('_pct') or
            col.endswith('_pct_delta') or
            col in [
                'Win_Change_1Y', 'NetRtg_Trend', 'OffDef_Ratio',
                'Prev_Point_Diff', 'Prev_eFG_Diff',
                'Prev_Pythag_WinPct', 'Prev_Pythag_Exp_Wins', 'Prev_Pythag_WinDiff',
                'Prev_SOS', 'SOS_Trend', 'SOS_Effect',
                'Health_Score', 'Star_Healthy', 'Healthy_Elite',
                'High_Continuity', 'Major_Turnover', 'Elite_Continuity',
                'Continuity_Adjusted_Wins'
            ]
        )
        and col not in ['Prev_Adv_W', 'Prev_Adv_L']
    ]

    # Remove any Vegas-related columns from features to keep training clean
    feature_cols = [col for col in feature_cols if 'Vegas' not in col]

    # Filter out features with too many missing values
    feature_cols = [
        col for col in feature_cols
        if col in df.columns and df[col].notna().sum() > 50
    ]

    # We’ll train on 2022-23, 2023-24, 2024-25 for final model
    allowed_train_seasons = ['2022-23', '2023-24', '2024-25']

    historical_data = df[
        df['Season'].isin(allowed_train_seasons)
        & df['Target_Wins'].notna()
        & df['Vegas_OU'].notna()
        & df['Vegas_Error_Target'].notna()
    ].copy()

    prediction_data = df[df['Season'] == prediction_season].copy()

    print(f"\nTraining seasons used: {allowed_train_seasons}")
    print(f"Historical rows: {len(historical_data)} | Features: {len(feature_cols)}")
    print("Some features:", feature_cols[:10])

    return historical_data, prediction_data, feature_cols

# =====================================================================
# 6. XGBOOST TRAINING (RECENCY + “HARD CASES” WEIGHTING)
# =====================================================================

def train_xgboost_model(X_train, y_train, season_train=None):
    # Base: all ones
    sample_weights = np.ones(len(y_train), dtype=float)

    # Recency weighting: 2024-25 > 2023-24 > 2022-23
    if season_train is not None:
        season_train = np.array(season_train)
        mask_22 = (season_train == '2022-23')
        mask_23 = (season_train == '2023-24')
        mask_24 = (season_train == '2024-25')

        sample_weights[mask_22] *= 1.0
        sample_weights[mask_23] *= 2.0
        sample_weights[mask_24] *= 5.0

    # Extra weighting for large Vegas errors (we care more about
    # learning how Vegas is wrong when it's off by a lot)
    abs_err = np.abs(y_train)
    sample_weights[abs_err >= 8] *= 1.5
    sample_weights[abs_err >= 12] *= 2.0

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 20,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'min_child_weight': 0.5,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'random_state': 42,
        'n_jobs': -1
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    return model

# =====================================================================
# 7. OPTIONAL: VALIDATE ON 2024-25 (USING RESIDUAL MODEL)
# =====================================================================

def validate_on_2024_25(df, feature_cols):
    print("\n" + "=" * 80)
    print("2024-25 VALIDATION (RESIDUAL MODEL: VEGAS + CORRECTION)")
    print("=" * 80)

    valid_training_seasons = [
    '2017-18',
    '2018-19',
    '2022-23',
    '2023-24',
    '2024-25'
    ]

    train_data = df[
        df['Season'].isin(valid_training_seasons)
        & df['Vegas_OU'].notna()
        & df['Target_Wins'].notna()
        & df['Vegas_Error_Target'].notna()
    ].copy()

    test_data = df[
        (df['Season'] == '2024-25')
        & df['Vegas_OU'].notna()
        & df['Target_Wins'].notna()
    ].copy()

    test_data_clean = test_data.dropna(subset=feature_cols)
    if len(test_data_clean) == 0:
        print("No clean test rows for 2024-25.")
        return pd.DataFrame(), 0, 0, 0

    X_train = train_data[feature_cols]
    y_train = train_data['Vegas_Error_Target']
    X_test = test_data_clean[feature_cols]
    y_test_wins = test_data_clean['Target_Wins']
    vegas_test = test_data_clean['Vegas_OU']

    model = train_xgboost_model(X_train, y_train, season_train=train_data['Season'].values)
    y_pred_error = model.predict(X_test)

    # Convert residual prediction back to wins
    y_pred_wins = vegas_test.values + y_pred_error

    mae = mean_absolute_error(y_test_wins, y_pred_wins)
    rmse = np.sqrt(mean_squared_error(y_test_wins, y_pred_wins))
    r2 = r2_score(y_test_wins, y_pred_wins)
    within_3 = np.mean(np.abs(y_test_wins - y_pred_wins) <= 3) * 100
    within_5 = np.mean(np.abs(y_test_wins - y_pred_wins) <= 5) * 100

    print(f"MODEL (Vegas + correction) -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    print(f"Within 3: {within_3:.1f}% | Within 5: {within_5:.1f}%")

    # VEGAS-only performance
    mae_vegas = mean_absolute_error(y_test_wins, vegas_test)
    rmse_vegas = np.sqrt(mean_squared_error(y_test_wins, vegas_test))
    print(f"VEGAS ONLY -> MAE: {mae_vegas:.2f} | RMSE: {rmse_vegas:.2f}")

    comparison = pd.DataFrame({
        'Team': test_data_clean['Team'].values,
        'Actual': y_test_wins.values,
        'Vegas_OU': vegas_test.values,
        'Model_Pred': np.round(y_pred_wins, 1),
        'Vegas_Error': np.round(vegas_test.values - y_test_wins.values, 1),
        'Model_Error': np.round(y_pred_wins - y_test_wins.values, 1),
    })

    comparison = comparison.sort_values('Actual', ascending=False).reset_index(drop=True)
    print("\n" + comparison.head(10).to_string(index=False))

    return comparison, mae, rmse, r2

# =====================================================================
# 8. TIME-SERIES CROSS VALIDATION (RESIDUAL MODEL)
# =====================================================================

def time_series_cross_validation(df, feature_cols, n_splits=2):
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION (TIME-SERIES BY SEASON, VEGAS + CORRECTION)")
    print("=" * 80)

    cv_seasons = ['2022-23', '2023-24', '2024-25']

    df_hist = df[
        df['Season'].isin(cv_seasons)
        & df['Target_Wins'].notna()
        & df['Vegas_OU'].notna()
        & df['Vegas_Error_Target'].notna()
    ].copy()

    seasons = sorted(df_hist['Season'].unique())
    results = []

    for i in range(len(seasons) - n_splits, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        if len(train_seasons) == 0:
            continue

        train_data = df_hist[df_hist['Season'].isin(train_seasons)]
        test_data = df_hist[df_hist['Season'] == test_season]

        train_data = train_data[train_data['Team'] != 'League Average']

        X_train = train_data[feature_cols]
        y_train = train_data['Vegas_Error_Target']
        X_test = test_data[feature_cols]
        y_test_wins = test_data['Target_Wins']
        vegas_test = test_data['Vegas_OU']

        model = train_xgboost_model(
            X_train, y_train, season_train=train_data['Season'].values
        )
        y_pred_error = model.predict(X_test)
        y_pred_wins = vegas_test.values + y_pred_error

        mae = mean_absolute_error(y_test_wins, y_pred_wins)
        r2 = r2_score(y_test_wins, y_pred_wins)
        within_3 = np.mean(np.abs(y_test_wins - y_pred_wins) <= 3) * 100

        results.append({
            'Season': test_season,
            'MAE': mae,
            'R2': r2,
            'Within_3': within_3
        })

        print(f"{test_season}: MAE={mae:.2f} | R²={r2:.3f} | Within 3={within_3:.1f}%")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        print(
            f"\nAverage: MAE={results_df['MAE'].mean():.2f} | "
            f"R²={results_df['R2'].mean():.3f}"
        )

    return results_df

# =====================================================================
# 9. PREDICT 2025-26 USING VEGAS + RESIDUAL MODEL
# =====================================================================

def predict_2025_26_season(df, feature_cols):
    print("\n" + "=" * 80)
    print("2025-26 PREDICTIONS (VEGAS + LEARNED CORRECTION)")
    print("=" * 80)

    allowed_train_seasons = ['2022-23', '2023-24', '2024-25']

    train_data = df[
        df['Season'].isin(allowed_train_seasons)
        & df['Target_Wins'].notna()
        & df['Vegas_OU'].notna()
        & df['Vegas_Error_Target'].notna()
    ].copy()

    # Predict only for real teams in 2025-26
    predict_data = df[
        (df['Season'] == '2025-26')
        & (df['Team'] != 'League Average')
    ].copy()

    if len(predict_data) == 0:
        print("No rows for 2025-26.")
        return None, None

    # Make sure they all have Vegas_OU
    if 'Vegas_OU' not in predict_data.columns:
        print("✗ No Vegas_OU for 2025-26 – cannot use residual model.")
        return None, None

    X_train = train_data[feature_cols]
    y_train = train_data['Vegas_Error_Target']
    X_predict = predict_data[feature_cols].copy()
    vegas_2025_26 = predict_data['Vegas_OU'].values

    # Train residual model
    model = train_xgboost_model(X_train, y_train, season_train=train_data['Season'].values)
    pred_error = model.predict(X_predict)

    # Raw predicted wins = Vegas + correction
    wins_raw = vegas_2025_26 + pred_error

    # --- Force league total wins to be correct ---
    n_teams = len(predict_data)
    target_total_wins = 41.0 * n_teams

    total_raw = wins_raw.sum()
    scale_model = target_total_wins / total_raw if total_raw > 0 else 1.0
    wins_scaled = wins_raw * scale_model

    print(f"Model sum before fix: {total_raw:.1f}")
    print(f"Model sum after fix:  {wins_scaled.sum():.1f}")
    print(f"League average now:   {wins_scaled.mean():.2f}")

    results = pd.DataFrame({
        'Team': predict_data['Team'].values,
        'Vegas_OU_2025_26': vegas_2025_26,
        'Model_Only_Wins': np.round(wins_raw, 1),
        'Predicted_Wins': np.round(wins_scaled).astype(int),
        'Previous_Wins': predict_data['Prev_Adv_W'].values,
        'Predicted_Correction': np.round(pred_error, 1),
        'Vegas_Error_if_unchanged': np.round(vegas_2025_26 - predict_data['Prev_Adv_W'].values, 1),
        'Change_from_Prev': np.round(wins_scaled - predict_data['Prev_Adv_W'].values, 1),
        'Prev_NetRtg': np.round(predict_data['Prev_Adv_NRtg'].values, 1)
    })

    results = results.sort_values('Predicted_Wins', ascending=False).reset_index(drop=True)
    results.index = results.index + 1

    print(results[['Team', 'Predicted_Wins', 'Vegas_OU_2025_26', 'Model_Only_Wins', 'Previous_Wins']].to_string())
    print(f"\nProjected Playoff Teams (>41 wins): {len(results[results['Predicted_Wins'] > 41])}")

    return results, model

# =====================================================================
# 10. FEATURE IMPORTANCE
# =====================================================================

def analyze_feature_importance(model, feature_cols, top_n=15):
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n" + "=" * 80)
    print(f"TOP {top_n} FEATURES (PREDICTING VEGAS ERROR)")
    print("=" * 80)
    print(importance_df.head(top_n).to_string(index=False))

    return importance_df

# =====================================================================
# 11. MAIN PIPELINE
# =====================================================================

def main():
    print("\n" + "=" * 80)
    print("NBA 2025-26 WIN PREDICTION - VEGAS RESIDUAL MODEL")
    print("=" * 80)

    # 1. Load core team stats and optional injuries/roster continuity
    df = load_and_prepare_data('nba_team_stats_2020_2025.csv')

    # Drop synthetic “League Average” row
    if 'Team' in df.columns:
        df = df[df['Team'] != 'League Average'].copy()

    df = create_2025_26_rows(df)

    # 2. Merge Vegas odds
    print("\n" + "=" * 80)
    print("MERGING VEGAS ODDS")
    print("=" * 80)


    try:
        vegas_df = pd.read_csv('nba_vegas_odds_clean.csv')
        print(f"Vegas CSV loaded: {len(vegas_df)} records")
        print(f"Vegas seasons: {sorted(vegas_df['Season'].unique())}")

        # -----------------------------------------------
        # FILTER TO ONLY MODERN VEGAS SEASONS (IMPORTANT)
        # -----------------------------------------------
        allowed_vegas_seasons = ['2022-23', '2023-24', '2024-25', '2025-26']
        vegas_df = vegas_df[vegas_df['Season'].isin(allowed_vegas_seasons)]

        print(f"Vegas seasons kept: {sorted(vegas_df['Season'].unique())}")
        print(f"Records after filtering: {len(vegas_df)}")

        # Check mismatched team names
        df_teams = set(df['Team'].unique())
        vegas_teams = set(vegas_df['Team'].unique())

        if df_teams != vegas_teams:
            print("\n⚠️  Team name mismatch detected")
            only_in_df = df_teams - vegas_teams
            only_in_vegas = vegas_teams - df_teams
            if only_in_df:
                print(f"  Only in main data (sample): {list(only_in_df)[:5]}")
            if only_in_vegas:
                print(f"  Only in Vegas (sample): {list(only_in_vegas)[:5]}")

        before_cols = len(df.columns)
        df = df.merge(vegas_df, on=['Season', 'Team'], how='left')
        after_cols = len(df.columns)

        if after_cols == before_cols:
            print("✗ Vegas merge failed - no new columns added!")
        else:
            vegas_total = df['Vegas_OU'].notna().sum()
            vegas_2025_26 = df[df['Season'] == '2025-26']['Vegas_OU'].notna().sum()
            print("✓ Vegas odds merged successfully")
            print(f"  Total with Vegas: {vegas_total}/{len(df)} records")
            print(f"  2025-26 with Vegas: {vegas_2025_26} teams")

    except FileNotFoundError:
        print("✗ nba_vegas_odds_clean.csv not found! Residual model cannot run.")
    except Exception as e:
        print(f"✗ Error merging Vegas: {e}")

    # 3. Create lagged & engineered features
    df = create_lagged_features(df)
    df = engineer_features(df)

    # 4. Build Vegas residual target
    if 'Vegas_OU' in df.columns:
        df['Vegas_Error_Target'] = df['Target_Wins'] - df['Vegas_OU']
    else:
        df['Vegas_Error_Target'] = np.nan

    # 5. Diagnostics: Vegas coverage & correlation vs actual wins
    print("\n" + "=" * 80)
    print("VEGAS DIAGNOSTICS")
    print("=" * 80)

    if 'Vegas_OU' in df.columns:
        vegas_coverage = df.groupby('Season')['Vegas_OU'].apply(lambda x: x.notna().sum())
        print("Vegas coverage by season:")
        print(vegas_coverage)

        historical = df[
            df['Target_Wins'].notna()
            & df['Season'].isin(['2022-23', '2023-24', '2024-25'])
        ].copy()

        if historical['Vegas_OU'].notna().sum() > 0:
            corr = historical[['Vegas_OU', 'Target_Wins']].corr().iloc[0, 1]
            print(f"\nVegas_OU vs Actual Wins correlation: {corr:.3f}")

            mae_vegas = mean_absolute_error(
                historical['Target_Wins'].values,
                historical['Vegas_OU'].fillna(41).values
            )
            rmse_vegas = np.sqrt(mean_squared_error(
                historical['Target_Wins'].values,
                historical['Vegas_OU'].fillna(41).values
            ))
            print(f"Vegas alone (historical, 2022-23 to 2024-25) -> MAE: {mae_vegas:.2f} | RMSE: {rmse_vegas:.2f}")
    else:
        print("No Vegas_OU column found after merge.")

    # 6. Prepare training/prediction sets and feature list
    historical_data, _, feature_cols = prepare_train_test_data(df)

    # 7. Time-series cross-validation (residual model)
    cv_results = time_series_cross_validation(df, feature_cols, n_splits=2)

    # 8. Predict 2025-26 (Vegas + learned correction)
    preds_2025_26, final_model = predict_2025_26_season(df, feature_cols)

    # 9. Feature importance
    if final_model is not None:
        analyze_feature_importance(final_model, feature_cols, top_n=15)

    # 10. Save predictions
    if preds_2025_26 is not None:
        preds_2025_26.to_csv('nba_2025_26_predictions.csv', index=False)
        print("\n✓ Saved: nba_2025_26_predictions.csv")

    # 11. Save CV results
    if cv_results is not None and not cv_results.empty:
        cv_results.to_csv('cross_validation_results.csv', index=False)
        print("✓ Saved: cross_validation_results.csv")

    # 12. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if cv_results is not None and not cv_results.empty:
        print(f"CV Average: MAE={cv_results['MAE'].mean():.2f} | "
              f"R²={cv_results['R2'].mean():.3f} | "
              f"Within 3={cv_results['Within_3'].mean():.1f}%")

    return df, preds_2025_26, final_model, cv_results

if __name__ == "__main__":
    df, predictions, model, cv_results = main()
