import pandas as pd
import numpy as np
import joblib

def clean_salary(val):
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return float(val)
    return float(str(val).replace('$', '').replace(',', ''))

def predict_2026_salaries():
    print("="*80)
    print("2026 PG SALARY PREDICTIONS")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = joblib.load('salary_model.pkl')
    qt = joblib.load('salary_transformer.pkl')
    training_features = joblib.load('feature_names.pkl')
    print(f"✓ Model loaded ({len(training_features)} features)")
    
    # Load data
    print("\nLoading data...")
    fa_data = pd.read_csv("2026 Free Agent Prediction Data.csv")
    fa_list = pd.read_csv("2026 Free Agent List.csv")
    fa_list.columns = fa_list.columns.str.lower()
    
    print(f"✓ Loaded {len(fa_data)} free agents")
    
    
    # Add position dummies
    position_cols = ['pos_C-PF', 'pos_PF', 'pos_PF-C', 'pos_PF-SF', 'pos_PG', 
                     'pos_PG-SG', 'pos_SF', 'pos_SF-PF', 'pos_SF-SG', 'pos_SG', 
                     'pos_SG-PG', 'pos_SG-SF']
    
    for col in position_cols:
        fa_data[col] = 0
    fa_data['pos_PG'] = 1
    
    # KEY FIX: Add current salary as adjusted_salary feature
    fa_list['player_lower'] = fa_list['player'].str.lower()
    fa_data['player_lower'] = fa_data['player'].str.lower()
    
    # Create salary mapping
    salary_map = dict(zip(fa_list['player_lower'], fa_list['prev aav']))
    fa_data['adjusted_salary'] = fa_data['player_lower'].map(salary_map)
    fa_data['adjusted_salary'] = fa_data['adjusted_salary'].apply(clean_salary)
    
    
    # Extract features
    X = fa_data[training_features].fillna(0)
    
    print(f"✓ Feature matrix: {X.shape}")
    print(f"  Non-zero features (avg): {(X != 0).sum(axis=1).mean():.1f}/{len(training_features)}")
    
    # Predict
    print("\nMaking predictions...")
    y_pred_trans = model.predict(X)
    predicted_salaries = qt.inverse_transform(y_pred_trans.reshape(-1, 1)).ravel()
    
    print(f"✓ Predictions complete")
    print(f"  Range: ${predicted_salaries.min():,.0f} - ${predicted_salaries.max():,.0f}")
    print(f"  Mean: ${predicted_salaries.mean():,.0f}")
    print(f"  Std: ${predicted_salaries.std():,.0f}")
    
    # Create results
    results = pd.DataFrame({
        'Player': fa_data['player'],
        'Age': fa_data['age'],
        'Season': fa_data['season'],
        'GP': fa_data['gp'],
        'PPG': fa_data['pts'],
        'APG': fa_data['ast'],
        'RPG': fa_data['reb'],
        'TS%': (fa_data['ts%'] * 100).round(1),
        'VORP': fa_data['vorp'],
        'Current Salary': fa_data['adjusted_salary'],
        '2026 Salary Prediction': predicted_salaries.round(0)
    })
    
    # Calculate changes
    results['Change'] = results['2026 Salary Prediction'] - results['Current Salary']
    results['Pct_Change'] = ((results['Change'] / results['Current Salary']) * 100).round(1)
    
    # Sort
    results = results.sort_values('2026 Salary Prediction', ascending=False)
    results = results.reset_index(drop=True)
    results.index = results.index + 1
    
    # Display
    print("\n" + "="*80)
    print("2026 SALARY PREDICTIONS")
    print("="*80)
    
    pd.set_option('display.float_format', '{:,.0f}'.format)
    display_cols = ['Player', 'Age', 'PPG', 'APG', 'VORP', 'Current Salary', 
                    '2026 Salary Prediction', 'Change', 'Pct_Change']
    print(results[display_cols].to_string())
    
    # Save
    results.to_csv("2026 PG Salary Predictions.csv", index=False)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Average: ${results['2026 Salary Prediction'].mean():,.0f}")
    print(f"Median: ${results['2026 Salary Prediction'].median():,.0f}")
    print(f"Range: ${results['2026 Salary Prediction'].min():,.0f} - ${results['2026 Salary Prediction'].max():,.0f}")
    
    return results

if __name__ == "__main__":
    predictions = predict_2026_salaries()



