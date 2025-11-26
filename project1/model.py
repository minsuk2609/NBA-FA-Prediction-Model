import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import unicodedata
import joblib

df_stats = pd.read_csv("data/Player Stat Advanced.csv")
df_salary = pd.read_csv("data/Salaries Adjusted.csv")

df_stats.columns = df_stats.columns.str.lower().str.strip()
df_salary.columns = df_salary.columns.str.lower().str.strip()

def clean_name(v):
    v = unicodedata.normalize('NFKD', str(v)).encode('ascii', 'ignore').decode('utf-8')
    return v.strip().lower()

df_stats['player'] = df_stats['player'].apply(clean_name)
df_salary['player'] = df_salary['player'].apply(clean_name)

df_salary['season'] = df_salary['year'].apply(lambda y: f"{y}-{str(y+1)[-2:]}")

df_combined = df_stats.merge(df_salary[['player', 'season', 'adjusted_salary', 'pos']], on=['player','season'], how='left')
df_combined = df_combined[df_combined['adjusted_salary'] > 1_500_000]

remove = ['player','team','season','awards','salary','adjusted_salary']
cols = [c for c in df_combined.columns if c not in remove]

X = df_combined[cols].copy()

if 'pos' in X.columns:
    X = pd.get_dummies(X, columns=['pos'], drop_first=True)

X = X.fillna(0)
y = df_combined['adjusted_salary']

qt = QuantileTransformer(output_distribution='normal')
y_t = qt.fit_transform(y.to_numpy().reshape(-1,1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y_t, test_size=0.2, random_state=42)

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

pred = model.predict(X_test)
pred_d = qt.inverse_transform(pred.reshape(-1,1)).ravel()
y_test_d = qt.inverse_transform(y_test.reshape(-1,1)).ravel()

mae = mean_absolute_error(y_test_d, pred_d)
r2 = r2_score(y_test_d, pred_d)

joblib.dump(model, 'salary_model.pkl')
joblib.dump(qt, 'salary_transformer.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
