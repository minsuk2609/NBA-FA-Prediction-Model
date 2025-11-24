import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_file='nba_team_stats_2020_2025.csv'):
    df = pd.read_csv(csv_file)
    print("="*80)
    print("DATA LOADED")
    print("="*80)
    print(f"Shape: {df.shape}")
    
    try:
        injury_df = pd.read_csv('nba_injury_data.csv')
        df = df.merge(injury_df, on=['Season', 'Team'], how='left')
        print(f"✓ Injury data merged")
    except:
        print("⚠️  No injury data")
    
    try:
        roster_df = pd.read_csv('nba_roster_continuity.csv')
        df = df.merge(roster_df, on=['Season', 'Team'], how='left')
        print(f"✓ Roster continuity merged")
    except:
        print("⚠️  No roster data")
    
    return df






def create_2025_26_rows(df):
    latest_season = df['Season'].max()
    teams = df[df['Season'] == latest_season]['Team'].unique()
    
    rows_2025_26 = [{'Season': '2025-26', 'Team': team} for team in teams]
    df_2025_26 = pd.DataFrame(rows_2025_26)
    df_combined = pd.concat([df, df_2025_26], ignore_index=True)
    df_combined = df_combined.sort_values(['Team', 'Season']).reset_index(drop=True)
    
    print("\n2025-26 rows created")
    return df_combined

def create_lagged_features(df):
    df = df.sort_values(['Team', 'Season']).reset_index(drop=True)
   
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
   
    if 'Games_Missed_Top3' in df.columns:
        feature_cols.extend(['Games_Missed_Top3', 'Injury_Prone_Count', 'Top_Player_Games'])
   
    if 'Returning_Minutes_Pct' in df.columns:
        feature_cols.extend(['Returning_Minutes_Pct', 'New_Players_Count'])
    
    # ADD: Vegas odds (don't lag it yet)
    if 'Vegas_OU' in df.columns:
        feature_cols.append('Vegas_OU')
   
    # Create lagged features
    for col in feature_cols:
        if col in df.columns:
            df[f'Prev_{col}'] = df.groupby('Team')[col].shift(1)
   
    key_cols = ['Adv_W', 'Adv_ORtg', 'Adv_DRtg', 'Adv_NRtg', 'Adv_MOV', 'Adv_SRS']
    for col in key_cols:
        if col in df.columns:
            df[f'Avg2Y_{col}'] = df.groupby('Team')[col].shift(1).rolling(window=2, min_periods=1).mean()
   
    df['Target_Wins'] = df['Adv_W']
   
    print("Lagged features created")
    return df



def engineer_features(df):
    df['Win_Change_1Y'] = df['Prev_Adv_W'] - df.groupby('Team')['Adv_W'].shift(2)
    df['NetRtg_Trend'] = df['Prev_Adv_NRtg'] - df.groupby('Team')['Adv_NRtg'].shift(2)
    df['OffDef_Ratio'] = df['Prev_Adv_ORtg'] / df['Prev_Adv_DRtg'].replace(0, np.nan)
    df['Prev_Point_Diff'] = df['Prev_PG_Team_PTS'] - df['Prev_PG_Opp_PTS']
    df['Prev_eFG_Diff'] = df['Prev_Adv_Offense Four Factors_eFG%'] - df['Prev_Adv_Defense Four Factors_eFG%']
    
    # Use weighted wins (80% actual, 20% MOV-based estimate)
    df['Prev_Wins_Weighted'] = 0.9 * df['Prev_Adv_W'] + 0.1 * (41 + df['Prev_Adv_MOV'] * 2.5)
    
    # Injury features
    if 'Prev_Games_Missed_Top3' in df.columns:
        df['Health_Score'] = 246 - df['Prev_Games_Missed_Top3']
        df['Star_Healthy'] = (df['Prev_Top_Player_Games'] >= 70).astype(float)
        df['Healthy_Elite'] = df['Prev_Adv_NRtg'] * (df['Health_Score'] / 246)
    
    # Roster continuity features
    if 'Prev_Returning_Minutes_Pct' in df.columns:
        continuity_pct = df['Prev_Returning_Minutes_Pct'] / 100
        df['High_Continuity'] = (df['Prev_Returning_Minutes_Pct'] >= 70).astype(float)
        df['Major_Turnover'] = (df['Prev_Returning_Minutes_Pct'] < 50).astype(float)
        df['Elite_Continuity'] = df['Prev_Adv_NRtg'] * continuity_pct
        df['Continuity_Adjusted_Wins'] = df['Prev_Adv_W'] * continuity_pct + 41 * (1 - continuity_pct)
    
    print("Features engineered")
    return df



def prepare_train_test_data(df, prediction_season='2025-26'):
    feature_cols = [col for col in df.columns
                    if (col.startswith('Prev_') or col.startswith('Avg') or
                        col in ['Win_Change_1Y', 'NetRtg_Trend', 'OffDef_Ratio',
                               'Prev_Point_Diff', 'Prev_eFG_Diff',
                               'Prev_Wins_Weighted',
                               'Health_Score', 'Star_Healthy', 'Healthy_Elite',
                               'High_Continuity', 'Major_Turnover', 'Elite_Continuity',
                               'Continuity_Adjusted_Wins'])
                    and col not in ['Prev_Adv_W', 'Prev_Adv_L']]
    
    # Filter out features with too many missing values
    feature_cols = [col for col in feature_cols if col in df.columns and df[col].notna().sum() > 50]
    
    # Check if Vegas is included
    if 'Prev_Vegas_OU' in feature_cols:
        print(f"✓ Prev_Vegas_OU is in features")
    elif 'Prev_Vegas_OU' in df.columns:
        non_null = df['Prev_Vegas_OU'].notna().sum()
        print(f"⚠️  Prev_Vegas_OU exists but excluded (only {non_null} non-null values)")
        if non_null > 50:
            feature_cols.append('Prev_Vegas_OU')
            print(f"  → Manually added to features")
    else:
        print(f"⚠️  Prev_Vegas_OU does not exist in dataframe")
    
    historical_data = df[(df['Season'] != prediction_season) & (df['Target_Wins'].notna())].copy()
    prediction_data = df[df['Season'] == prediction_season].copy()
    
    print(f"\nHistorical: {len(historical_data)} | Features: {len(feature_cols)}")
    return historical_data, prediction_data, feature_cols


def train_xgboost_model(X_train, y_train):
    # Heavy weighting for extreme teams
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train >= 60] = 6.0
    sample_weights[y_train >= 55] = 2.0
    sample_weights[y_train <= 20] = 6.0
    sample_weights[y_train <= 25] = 2.0
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'min_child_weight': 1,
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



def validate_on_2024_25(df, feature_cols):
    print("\n" + "="*80)
    print("2024-25 VALIDATION")
    print("="*80)
    
    train_data = df[(df['Season'] != '2024-25') & (df['Season'] != '2025-26') & (df['Target_Wins'].notna())]
    test_data = df[df['Season'] == '2024-25'].copy()
    test_data_clean = test_data.dropna(subset=feature_cols)
    
    if len(test_data_clean) == 0:
        return pd.DataFrame(), 0, 0, 0
    
    X_train = train_data[feature_cols]
    y_train = train_data['Target_Wins']
    X_test = test_data_clean[feature_cols]
    y_test = test_data_clean['Target_Wins']
    
    model = train_xgboost_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    within_3 = np.mean(np.abs(y_test - y_pred) <= 3) * 100
    within_5 = np.mean(np.abs(y_test - y_pred) <= 5) * 100
    
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    print(f"Within 3: {within_3:.1f}% | Within 5: {within_5:.1f}%")
    
    comparison = pd.DataFrame({
        'Team': test_data_clean['Team'].values,
        'Actual': y_test.values,
        'Predicted': np.round(y_pred, 1),
        'Error': np.round(y_pred - y_test.values, 1)
    })
    
    comparison = comparison.sort_values('Actual', ascending=False).reset_index(drop=True)
    print("\n" + comparison.head(10).to_string(index=False))
    
    return comparison, mae, rmse, r2

def time_series_cross_validation(df, feature_cols, n_splits=3):
    print("\n" + "="*80)
    print("CROSS-VALIDATION")
    print("="*80)
    
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
        
        X_train = train_data[feature_cols]
        y_train = train_data['Target_Wins']
        X_test = test_data[feature_cols]
        y_test = test_data['Target_Wins']
        
        model = train_xgboost_model(X_train, y_train)
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
        
        print(f"{test_season}: MAE={mae:.2f} | R²={r2:.3f} | Within3={within_3:.1f}%")
    
    results_df = pd.DataFrame(results)
    print(f"\nAverage: MAE={results_df['MAE'].mean():.2f} | R²={results_df['R2'].mean():.3f}")
    
    return results_df

def predict_2025_26_season(df, feature_cols):
    print("\n" + "="*80)
    print("2025-26 PREDICTIONS")
    print("="*80)
    
    train_data = df[(df['Season'] != '2025-26') & (df['Target_Wins'].notna())]
    predict_data = df[df['Season'] == '2025-26']
    
    if len(predict_data) == 0:
        return None, None
    
    X_train = train_data[feature_cols]
    y_train = train_data['Target_Wins']
    X_predict = predict_data[feature_cols].copy()
    
    # Get current Vegas odds
    current_vegas = predict_data['Vegas_OU'].fillna(41).values
    
    if 'Prev_Vegas_OU' in X_predict.columns:
        X_predict['Prev_Vegas_OU'] = current_vegas
        print(f"✓ Using 2025-26 preseason odds (range: {np.nanmin(current_vegas):.1f} - {np.nanmax(current_vegas):.1f})")
    
    model = train_xgboost_model(X_train, y_train)
    xgb_predictions = model.predict(X_predict)
    
    # ENSEMBLE: 40% XGBoost + 60% Vegas
    final_predictions = 0.3 * xgb_predictions + 0.7 * current_vegas
    
    print(f"Ensemble: 40% Model + 60% Vegas")
    
    results = pd.DataFrame({
        'Team': predict_data['Team'].values,
        'Predicted_Wins': np.round(final_predictions).astype(int),
        'Previous_Wins': predict_data['Prev_Adv_W'].values,
        'Vegas_OU_2025_26': current_vegas,
        'Model_Only': np.round(xgb_predictions, 1),
        'Change': np.round(final_predictions - predict_data['Prev_Adv_W'].values, 1),
        'Prev_NetRtg': np.round(predict_data['Prev_Adv_NRtg'].values, 1)
    })
    
    results = results.sort_values('Predicted_Wins', ascending=False).reset_index(drop=True)
    results.index = results.index + 1
    
    print(results[['Team', 'Predicted_Wins', 'Vegas_OU_2025_26', 'Model_Only', 'Previous_Wins']].to_string())
    print(f"\nProjected Playoff Teams (>41 wins): {len(results[results['Predicted_Wins'] > 41])}")
    
    return results, model






def analyze_feature_importance(model, feature_cols, top_n=15):
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*80)
    print(f"TOP {top_n} FEATURES")
    print("="*80)
    print(importance_df.head(top_n).to_string(index=False))
    
    return importance_df

def main():
    print("\n" + "="*80)
    print("NBA 2025-26 WIN PREDICTION - IMPROVED")
    print("="*80)
   
    df = load_and_prepare_data('nba_team_stats_2020_2025.csv')
    df = create_2025_26_rows(df)
   
    # IMPROVED Vegas merge with better error handling
    print("\n" + "="*80)
    print("MERGING VEGAS ODDS")
    print("="*80)
    
    try:
        vegas_df = pd.read_csv('nba_vegas_odds_clean.csv')
        print(f"Vegas CSV loaded: {len(vegas_df)} records")
        print(f"Vegas seasons: {sorted(vegas_df['Season'].unique())}")
        
        # Check for team name mismatches before merge
        df_teams = set(df['Team'].unique())
        vegas_teams = set(vegas_df['Team'].unique())
        
        if df_teams != vegas_teams:
            print(f"\n⚠️  Team name mismatch detected")
            only_in_df = df_teams - vegas_teams
            only_in_vegas = vegas_teams - df_teams
            if only_in_df:
                print(f"  Only in main data: {list(only_in_df)[:3]}")
            if only_in_vegas:
                print(f"  Only in Vegas: {list(only_in_vegas)[:3]}")
        
        # Merge
        before_cols = len(df.columns)
        df = df.merge(vegas_df, on=['Season', 'Team'], how='left')
        after_cols = len(df.columns)
        
        if after_cols == before_cols:
            print("✗ Vegas merge failed - no new columns added!")
            print(f"  Check if 'Vegas_OU' already exists: {'Vegas_OU' in df.columns}")
        else:
            vegas_total = df['Vegas_OU'].notna().sum()
            vegas_2025_26 = df[df['Season'] == '2025-26']['Vegas_OU'].notna().sum()
            print(f"✓ Vegas odds merged successfully")
            print(f"  Total: {vegas_total}/{len(df)} records")
            print(f"  2025-26: {vegas_2025_26}/31 teams")
            
    except FileNotFoundError:
        print("✗ nba_vegas_odds_clean.csv not found!")
    except Exception as e:
        print(f"✗ Error merging Vegas: {e}")
   
   
    df = create_lagged_features(df)
   
    print("\n" + "="*80)
    print("VEGAS DEBUG")
    print("="*80)
    print(f"Vegas_OU in df: {'Vegas_OU' in df.columns}")
    print(f"Prev_Vegas_OU in df: {'Prev_Vegas_OU' in df.columns}")
   
    if 'Prev_Vegas_OU' in df.columns:
        non_null = df['Prev_Vegas_OU'].notna().sum()
        print(f"Prev_Vegas_OU non-null: {non_null}/{len(df)}")
        if non_null > 0:
            print(f"\nSample Prev_Vegas_OU:")
            sample = df[['Team', 'Season', 'Vegas_OU', 'Prev_Vegas_OU']].dropna(subset=['Prev_Vegas_OU']).head(5)
            print(sample.to_string(index=False))
    
    # ADD EXTENDED DIAGNOSTIC:
    print("\n" + "="*80)
    print("VEGAS COVERAGE BY SEASON")
    print("="*80)
    vegas_coverage = df.groupby('Season')['Vegas_OU'].apply(lambda x: x.notna().sum())
    print(vegas_coverage)
    
    # Check correlation
    historical = df[df['Target_Wins'].notna()].copy()
    if 'Vegas_OU' in historical.columns and historical['Vegas_OU'].notna().sum() > 0:
        corr = historical[['Vegas_OU', 'Target_Wins']].corr().iloc[0, 1]
        print(f"\nVegas_OU vs Actual Wins correlation: {corr:.3f}")
        
        mae_vegas = mean_absolute_error(
            historical['Target_Wins'].values, 
            historical['Vegas_OU'].fillna(41).values
        )
        print(f"Vegas alone MAE: {mae_vegas:.2f} wins")
    
    df = engineer_features(df)
   
    historical_data, _, feature_cols = prepare_train_test_data(df)
   
    comp_2024, mae_2024, rmse_2024, r2_2024 = validate_on_2024_25(df, feature_cols)
    cv_results = time_series_cross_validation(historical_data, feature_cols, n_splits=3)
    preds_2025_26, final_model = predict_2025_26_season(df, feature_cols)
   
    if final_model:
        analyze_feature_importance(final_model, feature_cols, top_n=15)
   
    if preds_2025_26 is not None:
        preds_2025_26.to_csv('nba_2025_26_predictions.csv', index=False)
        print("\n✓ Saved: nba_2025_26_predictions.csv")
   
    if len(comp_2024) > 0:
        comp_2024.to_csv('2024_25_validation.csv', index=False)
        print("✓ Saved: 2024_25_validation.csv")
   
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"2024-25: MAE={mae_2024:.2f} | R²={r2_2024:.3f}")
    print(f"CV Average: MAE={cv_results['MAE'].mean():.2f} | R²={cv_results['R2'].mean():.3f}")
   
    return df, preds_2025_26, final_model, cv_results




if __name__ == "__main__":
    df, predictions, model, cv_results = main()


