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
# 3. CREATE LAGGED FEATURES (NO VEGAS USED FOR TRAINING)
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

    # Injury-related features (if present)
    if 'Games_Missed_Top3' in df.columns:
        feature_cols.extend(['Games_Missed_Top3', 'Injury_Prone_Count', 'Top_Player_Games'])

    # Roster continuity features (if present)
    if 'Returning_Minutes_Pct' in df.columns:
        feature_cols.extend(['Returning_Minutes_Pct', 'New_Players_Count'])

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
# 4. ENGINEER HIGHER-LEVEL FEATURES
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

    # Injury-related engineered features
    if 'Prev_Games_Missed_Top3' in df.columns:
        df['Health_Score'] = 246 - df['Prev_Games_Missed_Top3']
        df['Star_Healthy'] = (df['Prev_Top_Player_Games'] >= 60).astype(float)
        df['Healthy_Elite'] = df['Prev_Adv_NRtg'] * (df['Health_Score'] / 246)

    # Roster continuity features
    if 'Prev_Returning_Minutes_Pct' in df.columns:
        continuity_pct = df['Prev_Returning_Minutes_Pct'] / 100.0
        df['High_Continuity'] = (df['Prev_Returning_Minutes_Pct'] >= 70).astype(float)
        df['Major_Turnover'] = (df['Prev_Returning_Minutes_Pct'] < 50).astype(float)
        df['Elite_Continuity'] = df['Prev_Adv_NRtg'] * continuity_pct
        df['Continuity_Adjusted_Wins'] = (
            df['Prev_Adv_W'] * continuity_pct + 41 * (1 - continuity_pct)
        )
    
    

    print("Features engineered")
    return df

# =====================================================================
# 5. PREPARE TRAIN / TEST DATA (EXCLUDING VEGAS FROM FEATURES)
# =====================================================================

def prepare_train_test_data(df, prediction_season='2025-26'):
    # Base feature selection: all lagged/engineered features EXCEPT:
    # - any Vegas-based columns
    # - 'Prev_Adv_W' and 'Prev_Adv_L' (to avoid trivial leakage-like behavior)
    feature_cols = [
        col for col in df.columns
        if (
            (col.startswith('Prev_') or col.startswith('Avg') or
             col in [
                 'Win_Change_1Y', 'NetRtg_Trend', 'OffDef_Ratio',
                 'Prev_Point_Diff', 'Prev_eFG_Diff',
                 'Prev_Pythag_WinPct', 'Prev_Pythag_Exp_Wins', 'Prev_Pythag_WinDiff',
                 'Prev_SOS', 'SOS_Trend', 'SOS_Effect',
                 'Prev_Wins_Weighted',
                 'Health_Score', 'Star_Healthy', 'Healthy_Elite',
                 'High_Continuity', 'Major_Turnover', 'Elite_Continuity',
                 'Continuity_Adjusted_Wins'
             ])
            and col not in ['Prev_Adv_W', 'Prev_Adv_L']
        )
    ]

    # Remove any Vegas-related columns from features to keep training clean
    feature_cols = [
        col for col in feature_cols
        if 'Vegas' not in col
    ]

    # Filter out features with too many missing values
    feature_cols = [
        col for col in feature_cols
        if col in df.columns and df[col].notna().sum() > 50
    ]

    # Train only on 2022-23 and 2023-24
    allowed_train_seasons = ['2022-23', '2023-24']

    historical_data = df[
        df['Season'].isin(allowed_train_seasons) &
        df['Target_Wins'].notna()
    ].copy()

    prediction_data = df[df['Season'] == prediction_season].copy()

    print(f"\nTraining seasons used: {allowed_train_seasons}")
    print(f"Historical rows: {len(historical_data)} | Features: {len(feature_cols)}")
    print("Some features:", feature_cols[:10])

    return historical_data, prediction_data, feature_cols

# =====================================================================
# 6. XGBOOST TRAINING (WITH RECENCY + EXTREMES WEIGHTING)
# =====================================================================

def train_xgboost_model(X_train, y_train, season_train=None):
    # Base: all ones
    sample_weights = np.ones(len(y_train), dtype=float)

    # Recency weighting: most recent (2023-24) gets boosted vs 2022-23
    if season_train is not None:
        season_train = np.array(season_train)
        recent_mask = (season_train == '2023-24')
        older_mask = (season_train == '2022-23')

        sample_weights[older_mask] *= 1.0   # base
        sample_weights[recent_mask] *= 3.0  # double weight for most recent season

    # Extra weighting for extreme teams (very good / very bad),
    # applied multiplicatively on top of recency.
    sample_weights[y_train >= 55] *= 2.0
    sample_weights[y_train >= 60] *= 3.0
    sample_weights[y_train <= 25] *= 2.0
    sample_weights[y_train <= 20] *= 3.0

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
# 7. VALIDATE ON 2024-25 (MODEL vs VEGAS vs BLEND)
# =====================================================================

def validate_on_2024_25(df, feature_cols):
    print("\n" + "=" * 80)
    print("2024-25 VALIDATION (MODEL vs VEGAS vs BLEND)")
    print("=" * 80)

    allowed_train_seasons = ['2022-23', '2023-24']

    train_data = df[
        df['Season'].isin(allowed_train_seasons) &
        df['Target_Wins'].notna()
    ]
    test_data = df[df['Season'] == '2024-25'].copy()

    test_data_clean = test_data.dropna(subset=feature_cols)
    if len(test_data_clean) == 0:
        print("No clean test rows for 2024-25.")
        return pd.DataFrame(), 0, 0, 0

    X_train = train_data[feature_cols]
    y_train = train_data['Target_Wins']
    X_test = test_data_clean[feature_cols]
    y_test = test_data_clean['Target_Wins']

    model = train_xgboost_model(X_train, y_train, season_train=train_data['Season'].values)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    within_3 = np.mean(np.abs(y_test - y_pred) <= 3) * 100
    within_5 = np.mean(np.abs(y_test - y_pred) <= 5) * 100

    print(f"MODEL -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    print(f"MODEL -> Within 3: {within_3:.1f}% | Within 5: {within_5:.1f}%")

    # VEGAS-only performance (if available for that season)
    if 'Vegas_OU' in test_data_clean.columns and test_data_clean['Vegas_OU'].notna().sum() > 0:
        vegas = test_data_clean['Vegas_OU'].fillna(41)
        mae_vegas = mean_absolute_error(y_test, vegas)
        rmse_vegas = np.sqrt(mean_squared_error(y_test, vegas))
        print(f"VEGAS -> MAE: {mae_vegas:.2f} | RMSE: {rmse_vegas:.2f}")

        # Blend: α * Vegas + (1-α) * Model
        alpha = 0.7
        blend = alpha * vegas.values + (1 - alpha) * y_pred
        mae_blend = mean_absolute_error(y_test, blend)
        rmse_blend = np.sqrt(mean_squared_error(y_test, blend))
        print(f"BLEND (70% Vegas, 30% Model) -> MAE: {mae_blend:.2f} | RMSE: {rmse_blend:.2f}")

    comparison = pd.DataFrame({
        'Team': test_data_clean['Team'].values,
        'Actual': y_test.values,
        'Model_Pred': np.round(y_pred, 1),
        'Error_Model': np.round(y_pred - y_test.values, 1)
    })

    if 'Vegas_OU' in test_data_clean.columns:
        comparison['Vegas_OU'] = test_data_clean['Vegas_OU'].values
        comparison['Error_Vegas'] = np.round(test_data_clean['Vegas_OU'].values - y_test.values, 1)

    comparison = comparison.sort_values('Actual', ascending=False).reset_index(drop=True)
    print("\n" + comparison.head(10).to_string(index=False))

    return comparison, mae, rmse, r2

# =====================================================================
# 8. TIME-SERIES CROSS VALIDATION (MODEL ONLY)
# =====================================================================

def time_series_cross_validation(df, feature_cols, n_splits=3):
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION (TIME-SERIES BY SEASON)")
    print("=" * 80)

    df_hist = df[df['Target_Wins'].notna()].copy()
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
        y_train = train_data['Target_Wins']
        X_test = test_data[feature_cols]
        y_test = test_data['Target_Wins']

        model = train_xgboost_model(X_train, y_train, season_train=train_data['Season'].values)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        within_3 = np.mean(np.abs(y_test - y_pred) <= 3) * 100

        results.append({
            'Season': test_season,
            'MAE': mae,
            'R2': r2,
            'Within_3': within_3
        })

        print(f"{test_season}: MAE={mae:.2f} | R²={r2:.3f} | Within 3={within_3:.1f}%")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        print(f"\nAverage: MAE={results_df['MAE'].mean():.2f} | R²={results_df['R2'].mean():.3f}")

    return results_df

# =====================================================================
# 9. PREDICT 2025-26 USING MODEL + VEGAS BLEND
# =====================================================================

def predict_2025_26_season(df, feature_cols, vegas_weight=0.7):
    print("\n" + "=" * 80)
    print("2025-26 PREDICTIONS (BLEND: VEGAS + MODEL)")
    print("=" * 80)

    allowed_train_seasons = ['2022-23', '2023-24']

    train_data = df[
        df['Season'].isin(allowed_train_seasons) &
        df['Target_Wins'].notna()
    ].copy()

    # Predict only for real teams in 2025-26
    predict_data = df[df['Season'] == '2025-26'].copy()
    predict_data = predict_data[predict_data['Team'] != 'League Average'].copy()

    if len(predict_data) == 0:
        print("No rows for 2025-26.")
        return None, None

    X_train = train_data[feature_cols]
    y_train = train_data['Target_Wins']
    X_predict = predict_data[feature_cols].copy()

    # Train model on historical seasons (NO Vegas in features)
    model = train_xgboost_model(X_train, y_train, season_train=train_data['Season'].values)
    xgb_predictions = model.predict(X_predict)

    # --- Force league total wins to be correct ---
    n_teams = len(predict_data)
    target_total_wins = 41.0 * n_teams

    model_raw = xgb_predictions.copy()
    total_raw = model_raw.sum()
    scale_model = target_total_wins / total_raw if total_raw > 0 else 1.0
    xgb_predictions = model_raw * scale_model

    print(f"Model sum before fix: {total_raw:.1f}")
    print(f"Model sum after fix:  {xgb_predictions.sum():.1f}")
    print(f"League average now:   {xgb_predictions.mean():.2f}")

    # Get current Vegas odds (used ONLY here, not in training)
    if 'Vegas_OU' in predict_data.columns:
        current_vegas = predict_data['Vegas_OU'].fillna(41).values
    else:
        current_vegas = np.full(len(predict_data), 41.0)

    print(f"Using Vegas odds for {np.isfinite(current_vegas).sum()} teams")
    if np.isfinite(current_vegas).sum() > 0:
        print(f"Vegas range: {np.nanmin(current_vegas):.1f} - {np.nanmax(current_vegas):.1f}")

    # Ensemble: vegas_weight * Vegas + (1 - vegas_weight) * Model
    blend_raw = vegas_weight * current_vegas + (1 - vegas_weight) * xgb_predictions
    total_blend_raw = blend_raw.sum()
    scale_blend = target_total_wins / total_blend_raw if total_blend_raw > 0 else 1.0
    final_predictions = blend_raw * scale_blend

    print(f"Blend sum before fix: {total_blend_raw:.1f}")
    print(f"Blend sum after fix:  {final_predictions.sum():.1f}")
    print(f"League avg (blend):   {final_predictions.mean():.2f}")
    print(f"Ensemble: {int(vegas_weight * 100)}% Vegas + {int((1 - vegas_weight) * 100)}% Model")

    results = pd.DataFrame({
        'Team': predict_data['Team'].values,
        'Predicted_Wins': np.round(final_predictions).astype(int),
        'Previous_Wins': predict_data['Prev_Adv_W'].values,
        'Vegas_OU_2025_26': current_vegas,
        'Model_Only': np.round(xgb_predictions, 1),
        'Change_from_Prev': np.round(final_predictions - predict_data['Prev_Adv_W'].values, 1),
        'Prev_NetRtg': np.round(predict_data['Prev_Adv_NRtg'].values, 1)
    })

    results = results.sort_values('Predicted_Wins', ascending=False).reset_index(drop=True)
    results.index = results.index + 1

    print(results[['Team', 'Predicted_Wins', 'Vegas_OU_2025_26', 'Model_Only', 'Previous_Wins']].to_string())
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
    print(f"TOP {top_n} FEATURES")
    print("=" * 80)
    print(importance_df.head(top_n).to_string(index=False))

    return importance_df

# =====================================================================
# 11. MAIN PIPELINE
# =====================================================================

def main():
    print("\n" + "=" * 80)
    print("NBA 2025-26 WIN PREDICTION - NO VEGAS LEAK")
    print("=" * 80)

    # 1. Load core team stats and optional injuries/roster continuity
    df = load_and_prepare_data('nba_team_stats_2020_2025.csv')

    # Drop synthetic “League Average” row
    if 'Team' in df.columns:
        df = df[df['Team'] != 'League Average'].copy()

    df = create_2025_26_rows(df)

    # 2. Merge Vegas odds (for evaluation & blending ONLY)
    print("\n" + "=" * 80)
    print("MERGING VEGAS ODDS")
    print("=" * 80)

    try:
        vegas_df = pd.read_csv('nba_vegas_odds_clean.csv')
        print(f"Vegas CSV loaded: {len(vegas_df)} records")
        print(f"Vegas seasons: {sorted(vegas_df['Season'].unique())}")

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
        print("✗ nba_vegas_odds_clean.csv not found! Model will still run without blending.")
    except Exception as e:
        print(f"✗ Error merging Vegas: {e}")

    # 3. Create lagged & engineered features
    df = create_lagged_features(df)
    df = engineer_features(df)

    # 4. Diagnostics: Vegas coverage & correlation vs actual wins (historical)
    print("\n" + "=" * 80)
    print("VEGAS DIAGNOSTICS")
    print("=" * 80)

    if 'Vegas_OU' in df.columns:
        vegas_coverage = df.groupby('Season')['Vegas_OU'].apply(lambda x: x.notna().sum())
        print("Vegas coverage by season:")
        print(vegas_coverage)

        historical = df[df['Target_Wins'].notna()].copy()
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
            print(f"Vegas alone (historical) -> MAE: {mae_vegas:.2f} | RMSE: {rmse_vegas:.2f}")
    else:
        print("No Vegas_OU column found after merge.")

    # 5. Prepare training/prediction sets and feature list (NO Vegas columns)
    historical_data, _, feature_cols = prepare_train_test_data(df)

    # 6. Validate on 2024-25 (model vs Vegas vs blend)
    comp_2024, mae_2024, rmse_2024, r2_2024 = validate_on_2024_25(df, feature_cols)

    # 7. Time-series cross-validation (model only)
    cv_results = time_series_cross_validation(historical_data, feature_cols, n_splits=3)

    # 8. Predict 2025-26 (blend model + Vegas)
    preds_2025_26, final_model = predict_2025_26_season(df, feature_cols, vegas_weight=0.7)

    # 9. Feature importance
    if final_model is not None:
        analyze_feature_importance(final_model, feature_cols, top_n=15)

    # 10. Save outputs
    if preds_2025_26 is not None:
        preds_2025_26.to_csv('nba_2025_26_predictions.csv', index=False)
        print("\n✓ Saved: nba_2025_26_predictions.csv")

    if len(comp_2024) > 0:
        comp_2024.to_csv('2024_25_validation.csv', index=False)
        print("✓ Saved: 2024_25_validation.csv")

    # 11. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"2024-25 MODEL: MAE={mae_2024:.2f} | RMSE={rmse_2024:.2f} | R²={r2_2024:.3f}")
    if not cv_results.empty:
        print(f"CV Average: MAE={cv_results['MAE'].mean():.2f} | R²={cv_results['R2'].mean():.3f}")

    return df, preds_2025_26, final_model, cv_results

if __name__ == "__main__":
    df, predictions, model, cv_results = main()
