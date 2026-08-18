import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def fit_model(x, y, season_train):
    weights = np.ones(len(y))

    season_train = np.array(season_train)
    weights[season_train=='2022-23'] *= 1
    weights[season_train=='2023-24'] *= 2
    weights[season_train=='2024-25'] *= 5

    absDiff = np.abs(y)
    weights[absDiff >= 8] *= 1.5
    weights[absDiff >= 12] *= 2

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 250,
        'min_child_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'reg_alpha': 1,
        'reg_lambda': 8,
        'random_state': 42,
        'n_jobs': -1
    }

    m = xgb.XGBRegressor(**params)
    m.fit(x, y, sample_weight=weights)
    return m

def run_cv(data, cols):
    seasons = ['2022-23','2023-24','2024-25']
    df = data[data['Season'].isin(seasons) & data['Vegas_OU'].notna() & data['Vegas_Error_Target'].notna()].copy()
    df = df[df['Team'] != 'League Average']
    res = []

    for i in range(1,len(seasons)-1):
        train_season = seasons[:i]
        test_season = seasons[i]

        train = df[df['Season'].isin(train_season)]
        test = df[df['Season']==test_season]

        x_train = train[cols]
        y_train = train['Vegas_Error_Target']
        x_test = test[cols]
        vegas_vals = test['Vegas_OU']
        yreal = test['Target_Wins']


        model = fit_model(x_train, y_train, train['Season'])
        pred = model.predict(x_test)
        win_pred = vegas_vals.values + pred

        err = mean_absolute_error(yreal, win_pred)
        r2val = r2_score(yreal, win_pred)


        res.append({'Train Seasons': train_season, 'Test Seasons': test_season, 'MAE': err, 'R2': r2val})

    return pd.DataFrame(res)

def model_io(model, path, mode):
    if mode == 'save':
        joblib.dump(model, path)
    elif mode == 'load':
        return joblib.load(path)
