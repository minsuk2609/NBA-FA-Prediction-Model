import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import QuantileTransformer
import xgboost as xgb
import joblib
import unicodedata

# -------------------------------
# 1. Load CSVs
# -------------------------------
stats_df = pd.read_csv("player_stat_2010-2024_with_advanced.csv")
salaries_df = pd.read_csv("salaries_2020_2025_adjusted.csv")

# Normalize column names
stats_df.columns = stats_df.columns.str.lower().str.strip()
salaries_df.columns = salaries_df.columns.str.lower().str.strip()

# -------------------------------
# 2. Normalize player names
# -------------------------------
def normalize_name(name):
    return unicodedata.normalize('NFKD', str(name)).encode('ascii','ignore').decode('utf-8').strip().lower()

stats_df['player'] = stats_df['player'].apply(normalize_name)
salaries_df['player'] = salaries_df['player'].apply(normalize_name)

# -------------------------------
# 3. Convert Year to Season for salaries CSV
# -------------------------------
salaries_df['season'] = salaries_df['year'].apply(lambda x: f"{x}-{str(x+1)[-2:]}")

# -------------------------------
# 4. Merge stats + salaries
# -------------------------------
merged_df = stats_df.merge(
    salaries_df[['player','season','adjusted_salary','pos']],  # keep position
    on=['player','season'],
    how='left'
)

print(f"Salaries matched: {merged_df['adjusted_salary'].notna().sum()} / {len(merged_df)}")

# -------------------------------
# 5. Filter out very low salaries to reduce skew
# -------------------------------
merged_df = merged_df[merged_df['adjusted_salary'] > 1500000]  # remove tiny salaries
y = merged_df['adjusted_salary']

# -------------------------------
# 6. Prepare features
# -------------------------------
drop_cols = ['player','team','season','awards','salary']  # non-feature columns
feature_cols = [c for c in merged_df.columns if c not in drop_cols]

X = merged_df[feature_cols].copy()

# Encode position as one-hot
if 'pos' in X.columns:
    X = pd.get_dummies(X, columns=['pos'], drop_first=True)

# Fill missing numeric values
X = X.fillna(0)

# -------------------------------
# 7. Transform target with QuantileTransformer
# -------------------------------
qt = QuantileTransformer(output_distribution='normal')
y_trans = qt.fit_transform(y.values.reshape(-1,1)).ravel()

# -------------------------------
# 8. Cross-validation (optional)
# -------------------------------
cv_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_model, X, y_trans, cv=kf, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
print("Cross-validated MSE:", mse_scores)
print("Mean CV MSE:", mse_scores.mean())
print("Std CV MSE:", mse_scores.std())

# -------------------------------
# 9. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y_trans, merged_df.index, test_size=0.2, random_state=42
)

# -------------------------------
# 10. Train XGBoost model
# -------------------------------
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


joblib.dump(model, 'salary_model.pkl')
joblib.dump(qt, 'salary_transformer.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')

print("\n✓ Model saved: salary_model.pkl")
print("✓ Transformer saved: salary_transformer.pkl")
print("✓ Features saved: feature_names.pkl")

# -------------------------------
# 11. Predict & inverse-transform
# -------------------------------
y_pred_trans = model.predict(X_test)
y_pred_dollars = qt.inverse_transform(y_pred_trans.reshape(-1,1)).ravel()
y_actual_dollars = qt.inverse_transform(y_test.reshape(-1,1)).ravel()

# -------------------------------
# 12. Metrics
# -------------------------------
mse = mean_squared_error(y_test, y_pred_trans)
r2 = r2_score(y_test, y_pred_trans)
mae = mean_absolute_error(y_actual_dollars, y_pred_dollars)
medae = np.median(np.abs(y_actual_dollars - y_pred_dollars))

print(f"Test MSE (transformed target): {mse:.4f}")
print(f"Test R^2 (transformed target): {r2:.4f}")
print(f"Mean Absolute Error in Salary: ${mae:,.0f}")
print(f"Median Absolute Error in Salary: ${medae:,.0f}")

# -------------------------------
# 13. Results DataFrame
# -------------------------------
results = pd.DataFrame({
    'player': merged_df.loc[idx_test, 'player'],
    'season': merged_df.loc[idx_test, 'season'],
    'actual_salary': y_actual_dollars,
    'predicted_salary': y_pred_dollars
})

results['error'] = results['actual_salary'] - results['predicted_salary']
results['abs_error'] = results['error'].abs()

# Round numeric columns for readability
results[['actual_salary','predicted_salary','error','abs_error']] = results[['actual_salary','predicted_salary','error','abs_error']].round(0)

# Sort by absolute error descending
results = results.sort_values(by='abs_error', ascending=False)

# Display nicely
pd.set_option('display.float_format', '{:,.0f}'.format)
print(results.head(10))

# Save to CSV
results.to_csv("salary_predictions.csv", index=False)
