import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD DATA
# ============================================

def load_and_prepare_data(csv_file='nba_team_stats_2020_2025.csv'):
    df = pd.read_csv(csv_file)
    print("="*80)
    print("DATA LOADED")
    print("="*80)
    print(f"Shape: {df.shape}")
    print(f"Seasons: {sorted(df['Season'].unique())}")
    return df


# ============================================
# STEP 2: CREATE 2025-26 ROWS
# ============================================

def create_2025_26_rows(df):
    latest_season = df['Season'].max()
    teams = df[df['Season'] == latest_season]['Team'].unique()
    
    rows_2025_26 = [{'Season': '2025-26', 'Team': team} for team in teams]
    df_2025_26 = pd.DataFrame(rows_2025_26)
    df_combined = pd.concat([df, df_2025_26], ignore_index=True)
    df_combined = df_combined.sort_values(['Team', 'Season']).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("2025-26 ROWS CREATED")
    print("="*80)
    print(f"Added {len(teams)} teams")
    return df_combined


# ============================================
# STEP 3: CREATE LAGGED FEATURES (FIXED)
# ============================================

def create_lagged_features(df):
    df = df.sort_values(['Team', 'Season']).reset_index(drop=True)
    
    # Core stats to lag
    feature_cols = [
        'PG_Team_PTS', 'PG_Opp_PTS', 'PG_Team_FG%', 'PG_Team_3P%', 'PG_Team_FT%',
        'PG_Team_TRB', 'PG_Team_AST', 'PG_Team_STL', 'PG_Team_BLK', 'PG_Team_TOV',
        'PG_Opp_FG%', 'PG_Opp_3P%', 'PG_Opp_TRB', 'PG_Opp_AST', 'PG_Opp_TOV',
        'Adv_W', 'Adv_L', 'Adv_MOV', 'Adv_SOS', 'Adv_SRS',
        'Adv_ORtg', 'Adv_DRtg', 'Adv_NRtg', 'Adv_Pace', 'Adv_TS%', 'Adv_FTr', 'Adv_3PAr',
        'Adv_Offense Four Factors_eFG%', 'Adv_Offense Four Factors_TOV%',
        'Adv_Offense Four Factors_ORB%', 'Adv_Offense Four Factors_FT/FGA',
        'Adv_Defense Four Factors_eFG%', 'Adv_Defense Four Factors_TOV%',
        'Adv_Defense Four Factors_DRB%', 'Adv_Defense Four Factors_FT/FGA',
    ]
    
    # Create 1-year lag
    for col in feature_cols:
        if col in df.columns:
            df[f'Prev_{col}'] = df.groupby('Team')[col].shift(1)
    
    # 2-year and 3-year averages
    key_cols = ['Adv_W', 'Adv_ORtg', 'Adv_DRtg', 'Adv_NRtg', 'Adv_MOV', 'Adv_SRS']
    for col in key_cols:
        if col in df.columns:
            df[f'Avg2Y_{col}'] = df.groupby('Team')[col].shift(1).rolling(window=2, min_periods=1).mean()
            df[f'Avg3Y_{col}'] = df.groupby('Team')[col].shift(1).rolling(window=3, min_periods=1).mean()
    
    df['Target_Wins'] = df['Adv_W']
    
    print("\n" + "="*80)
    print("LAGGED FEATURES CREATED")
    print("="*80)
    
    return df


# ============================================
# STEP 4: FEATURE ENGINEERING (FIXED - REMOVED BAD FEATURES)
# ============================================

def engineer_features(df):
    """Create features WITHOUT the bugs"""
    
    # 1. Win trends
    df['Win_Change_1Y'] = df['Prev_Adv_W'] - df['Avg2Y_Adv_W']
    
    # 2. Rating trends
    df['NetRtg_Trend'] = df['Prev_Adv_NRtg'] - df['Avg2Y_Adv_NRtg']
    df['OffDef_Ratio'] = df['Prev_Adv_ORtg'] / df['Prev_Adv_DRtg'].replace(0, np.nan)
    
    # 3. Point differential
    df['Prev_Point_Diff'] = df['Prev_PG_Team_PTS'] - df['Prev_PG_Opp_PTS']
    
    # 4. Four Factors differentials
    df['Prev_eFG_Diff'] = df['Prev_Adv_Offense Four Factors_eFG%'] - df['Prev_Adv_Defense Four Factors_eFG%']
    df['Prev_TOV_Diff'] = df['Prev_Adv_Defense Four Factors_TOV%'] - df['Prev_Adv_Offense Four Factors_TOV%']
    
    # 5. Pythagorean wins (FIXED FORMULA)
    # Formula: Wins = Games * (PF^14) / (PF^14 + PA^14)
    # Simplified using MOV
    df['Prev_Pythag_Wins'] = 41 + (df['Prev_Adv_MOV'] * 2.7)  # Linear approximation
    df['Prev_Pythag_Wins'] = df['Prev_Pythag_Wins'].clip(0, 82)
    
    # 6. Luck factor
    df['Prev_Luck_Factor'] = df['Prev_Adv_W'] - df['Prev_Pythag_Wins']
    
    # 7. SRS trend
    df['SRS_Trend'] = df['Prev_Adv_SRS'] - df['Avg2Y_Adv_SRS']
    
    # 8. Team tier indicators
    df['Is_Elite_Team'] = (df['Prev_Adv_W'] >= 55).astype(float)
    df['Is_Playoff_Team'] = ((df['Prev_Adv_W'] >= 42) & (df['Prev_Adv_W'] < 55)).astype(float)
    df['Is_Bad_Team'] = (df['Prev_Adv_W'] <= 30).astype(float)
    
    # 9. Win percentage
    total_games = df['Prev_Adv_W'] + df['Prev_Adv_L']
    df['Prev_Win_Pct'] = df['Prev_Adv_W'] / total_games.replace(0, np.nan)
    
    # 10. Weighted recent performance
    df['Weighted_Wins'] = 0.6 * df['Prev_Adv_W'] + 0.4 * df['Avg2Y_Adv_W']
    
    print("\n" + "="*80)
    print("FEATURES ENGINEERED")
    print("="*80)
    
    # DEBUG: Check for extreme values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].abs().max() > 1000:
            print(f"⚠️  WARNING: {col} has extreme values (max: {df[col].abs().max():.2f})")
    
    return df


# ============================================
# STEP 5: PREPARE DATA
# ============================================

def prepare_train_test_data(df, prediction_season='2025-26'):
    feature_cols = [col for col in df.columns 
                    if (col.startswith('Prev_') or col.startswith('Avg') or 
                        col in ['Win_Change_1Y', 'NetRtg_Trend', 'OffDef_Ratio',
                               'Prev_Point_Diff', 'Prev_eFG_Diff', 'Prev_TOV_Diff',
                               'Prev_Pythag_Wins', 'Prev_Luck_Factor', 'SRS_Trend',
                               'Is_Elite_Team', 'Is_Playoff_Team', 'Is_Bad_Team',
                               'Prev_Win_Pct', 'Weighted_Wins'])
                    and col not in ['Prev_Adv_W', 'Prev_Adv_L']]
    
    # Remove features with too many NaN
    feature_cols = [col for col in feature_cols if col in df.columns and df[col].notna().sum() > 50]
    
    historical_data = df[(df['Season'] != prediction_season) & (df['Target_Wins'].notna())].copy()
    prediction_data = df[df['Season'] == prediction_season].copy()
    
    print("\n" + "="*80)
    print("DATA SPLIT")
    print("="*80)
    print(f"Historical: {len(historical_data)} | Prediction: {len(prediction_data)} | Features: {len(feature_cols)}")
    
    return historical_data, prediction_data, feature_cols


# ============================================
# STEP 6: TRAIN MODEL (SIMPLER PARAMETERS)
# ============================================

def train_xgboost_model(X_train, y_train):
    """Train with SAFE parameters"""
    params = {
        'objective': 'reg:squarederror',  # Back to standard loss
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 400,
        'min_child_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.5,
        'reg_lambda': 1.5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


# ============================================
# STEP 7: VALIDATE ON 2024-25
# ============================================

def validate_on_2024_25(df, feature_cols):
    print("\n" + "="*80)
    print("VALIDATING ON 2024-25")
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
    
    print(f"Validating on {len(test_data_clean)} teams")
    
    # DEBUG: Check for extreme values in training data
    print(f"Training data range: {X_train.min().min():.2f} to {X_train.max().max():.2f}")
    print(f"Test data range: {X_test.min().min():.2f} to {X_test.max().max():.2f}")
    
    model = train_xgboost_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # DEBUG: Check predictions
    print(f"Prediction range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    within_3 = np.mean(np.abs(y_test - y_pred) <= 3) * 100
    within_5 = np.mean(np.abs(y_test - y_pred) <= 5) * 100
    
    print(f"\n2024-25 Performance:")
    print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    print(f"  Within 3: {within_3:.1f}% | Within 5: {within_5:.1f}%")
    
    comparison = pd.DataFrame({
        'Team': test_data_clean['Team'].values,
        'Actual': y_test.values,
        'Predicted': np.round(y_pred, 1),
        'Error': np.round(y_pred - y_test.values, 1),
        'Abs_Error': np.round(np.abs(y_pred - y_test.values), 1)
    })
    
    comparison = comparison.sort_values('Actual', ascending=False).reset_index(drop=True)
    print("\n" + comparison.head(10).to_string(index=False))
    
    return comparison, mae, rmse, r2


# ============================================
# STEP 8: CROSS-VALIDATION (CONTINUED)
# ============================================

def time_series_cross_validation(df, feature_cols, n_splits=3):
    print("\n" + "="*80)
    print("CROSS-VALIDATION")
    print("="*80)
    
    df_hist = df[df['Target_Wins'].notna()].copy()
    seasons = sorted(df_hist['Season'].unique())
    
    results = []
    predictions_list = []
    
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
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        within_3 = np.mean(np.abs(y_test - y_pred) <= 3) * 100
        within_5 = np.mean(np.abs(y_test - y_pred) <= 5) * 100
        
        results.append({
            'Test_Season': test_season,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Within_3': within_3,
            'Within_5': within_5
        })
        
        pred_df = test_data[['Season', 'Team', 'Target_Wins']].copy()
        pred_df['Predicted_Wins'] = y_pred
        pred_df['Error'] = y_pred - y_test
        predictions_list.append(pred_df)
        
        print(f"{test_season}: MAE={mae:.2f} | R²={r2:.3f} | Within3={within_3:.1f}%")
    
    results_df = pd.DataFrame(results)
    all_preds = pd.concat(predictions_list, ignore_index=True)
    
    print(f"\nAverage: MAE={results_df['MAE'].mean():.2f} | R²={results_df['R2'].mean():.3f}")
    
    return results_df, all_preds


# ============================================
# STEP 9: PREDICT 2025-26
# ============================================

def predict_2025_26_season(df, feature_cols):
    print("\n" + "="*80)
    print("PREDICTING 2025-26 SEASON")
    print("="*80)
    
    train_data = df[(df['Season'] != '2025-26') & (df['Target_Wins'].notna())]
    predict_data = df[df['Season'] == '2025-26']
    
    if len(predict_data) == 0:
        return None, None
    
    X_train = train_data[feature_cols]
    y_train = train_data['Target_Wins']
    X_predict = predict_data[feature_cols]
    
    print(f"Training on {len(train_data)} records...")
    model = train_xgboost_model(X_train, y_train)
    
    predictions = model.predict(X_predict)
    
    results = pd.DataFrame({
        'Team': predict_data['Team'].values,
        'Predicted_Wins': np.round(predictions, 1),
        'Previous_Wins': predict_data['Prev_Adv_W'].values,
        'Change': np.round(predictions - predict_data['Prev_Adv_W'].values, 1),
        'Prev_NetRtg': np.round(predict_data['Prev_Adv_NRtg'].values, 1)
    })
    
    results = results.sort_values('Predicted_Wins', ascending=False).reset_index(drop=True)
    results.index = results.index + 1
    
    print("\n2025-26 PREDICTIONS:")
    print(results.to_string())
    
    playoff = results[results['Predicted_Wins'] > 41]
    print(f"\nProjected Playoff Teams: {len(playoff)}")
    
    return results, model


# ============================================
# STEP 10: FEATURE IMPORTANCE
# ============================================

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


# ============================================
# STEP 11: VISUALIZATIONS
# ============================================

def visualize_2024_25(comparison_df):
    if len(comparison_df) == 0:
        return
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df['Actual'], width, label='Actual', alpha=0.8)
    plt.bar(x + width/2, comparison_df['Predicted'], width, label='Predicted', alpha=0.8)
    
    plt.xlabel('Teams')
    plt.ylabel('Wins')
    plt.title('2024-25: Actual vs Predicted Wins', fontweight='bold')
    plt.xticks(x, comparison_df['Team'], rotation=90, fontsize=8)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('2024_25_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 2024_25_comparison.png")
    plt.close()


# ============================================
# STEP 12: MAIN PIPELINE
# ============================================

def main():
    print("\n" + "="*80)
    print("NBA 2025-26 WIN PREDICTION - FIXED VERSION")
    print("="*80)
    
    # Load and prepare
    df = load_and_prepare_data('nba_team_stats_2020_2025.csv')
    df = create_2025_26_rows(df)
    df = create_lagged_features(df)
    df = engineer_features(df)
    
    historical_data, prediction_data, feature_cols = prepare_train_test_data(df)
    
    # Validate on 2024-25
    comp_2024, mae_2024, rmse_2024, r2_2024 = validate_on_2024_25(df, feature_cols)
    visualize_2024_25(comp_2024)
    
    # Cross-validation
    cv_results, cv_preds = time_series_cross_validation(historical_data, feature_cols, n_splits=3)
    
    # Predict 2025-26
    preds_2025_26, final_model = predict_2025_26_season(df, feature_cols)
    
    # Feature importance
    if final_model:
        feat_imp = analyze_feature_importance(final_model, feature_cols, top_n=15)
    
    # Save results
    if preds_2025_26 is not None:
        preds_2025_26.to_csv('nba_2025_26_predictions.csv', index=False)
        print("\n✓ Saved: nba_2025_26_predictions.csv")
    
    if len(comp_2024) > 0:
        comp_2024.to_csv('2024_25_validation.csv', index=False)
        print("✓ Saved: 2024_25_validation.csv")
    
    cv_results.to_csv('cv_results.csv', index=False)
    print("✓ Saved: cv_results.csv")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if len(comp_2024) > 0:
        print(f"\n2024-25 Validation:")
        print(f"  MAE: {mae_2024:.2f} wins")
        print(f"  RMSE: {rmse_2024:.2f} wins")
        print(f"  R²: {r2_2024:.3f}")
    
    print(f"\nCross-Validation:")
    print(f"  MAE: {cv_results['MAE'].mean():.2f} wins")
    print(f"  RMSE: {cv_results['RMSE'].mean():.2f} wins")
    print(f"  R²: {cv_results['R2'].mean():.3f}")
    print(f"  Within 3 wins: {cv_results['Within_3'].mean():.1f}%")
    print(f"  Within 5 wins: {cv_results['Within_5'].mean():.1f}%")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    
    return df, preds_2025_26, final_model, cv_results, comp_2024


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    df, predictions, model, cv_results, validation = main()



