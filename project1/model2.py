import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import QuantileTransformer
import xgboost as xgb
import unicodedata
import joblib

print("="*80)
print("RETRAINING MODEL - SALARY NOT AS FEATURE")
print("="*80)

# Load data
stats_df = pd.read_csv("player_stat_2010-2024_with_advanced.csv")
salaries_df = pd.read_csv("salaries_2020_2025_adjusted.csv")

stats_df.columns = stats_df.columns.str.lower().str.strip()
salaries_df.columns = salaries_df.columns.str.lower().str.strip()

# Normalize names
def normalize_name(name):
    return unicodedata.normalize('NFKD', str(name)).encode('ascii','ignore').decode('utf-8').strip().lower()

stats_df['player'] = stats_df['player'].apply(normalize_name)
salaries_df['player'] = salaries_df['player'].apply(normalize_name)

# Create season column
salaries_df['season'] = salaries_df['year'].apply(lambda x: f"{x}-{str(x+1)[-2:]}")

# Merge
merged_df = stats_df.merge(
    salaries_df[['player','season','adjusted_salary','pos']],
    on=['player','season'],
    how='left'
)

print(f"Salaries matched: {merged_df['adjusted_salary'].notna().sum()} / {len(merged_df)}")

# Filter
merged_df = merged_df[merged_df['adjusted_salary'] > 1_500_000]
print(f"After filtering: {len(merged_df)} records")

# Prepare features - EXCLUDE adjusted_salary
drop_cols = ['player','team','season','awards','salary','adjusted_salary']  # ← KEY CHANGE
feature_cols = [c for c in merged_df.columns if c not in drop_cols]

X = merged_df[feature_cols].copy()

# One-hot encode position
if 'pos' in X.columns:
    X = pd.get_dummies(X, columns=['pos'], drop_first=True)

X = X.fillna(0)
y = merged_df['adjusted_salary']

print(f"\nFeature columns ({len(X.columns)}): {X.columns.tolist()}")
print(f"Salary range: ${y.min():,.0f} - ${y.max():,.0f}")

# Transform
qt = QuantileTransformer(output_distribution='normal')
y_trans = qt.fit_transform(y.values.reshape(-1,1)).ravel()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_trans, test_size=0.2, random_state=42
)

# Train
print("\nTraining model...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, verbose=False)

# Evaluate
y_pred_trans = model.predict(X_test)
y_pred_dollars = qt.inverse_transform(y_pred_trans.reshape(-1, 1)).ravel()
y_actual_dollars = qt.inverse_transform(y_test.reshape(-1, 1)).ravel()

mae = mean_absolute_error(y_actual_dollars, y_pred_dollars)
r2 = r2_score(y_actual_dollars, y_pred_dollars)

print(f"\nModel Performance:")
print(f"  R²: {r2:.4f}")
print(f"  MAE: ${mae:,.0f}")
print(f"  Prediction range: ${y_pred_dollars.min():,.0f} - ${y_pred_dollars.max():,.0f}")
print(f"  Prediction std: ${y_pred_dollars.std():,.0f}")

# Save
joblib.dump(model, 'salary_model_final.pkl')
joblib.dump(qt, 'salary_transformer_final.pkl')
joblib.dump(X.columns.tolist(), 'feature_names_final.pkl')

print("\n✓ Model saved (WITHOUT adjusted_salary as feature):")
print("  - salary_model_final.pkl")
print("  - salary_transformer_final.pkl")
print("  - feature_names_final.pkl")



