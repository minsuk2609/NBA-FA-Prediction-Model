import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def fitModel(x, y, season_train):

    weights = np.ones(len(y))

    if season_train is not None:
        season_train = np.array(season_train)
        weights[season_train=='2022-23'] *= 1
        weights[season_train=='2023-24'] *= 2
        weights[season_train=='2024-25'] *= 5

    absDiff = np.abs(y)
    weights[absDiff >= 8] *= 1.5
    weights[absDiff >= 12] *= 2

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'min_child_weight': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 2,
        'reg_alpha': 2,
        'reg_lambda': 5,
        'random_state': 42,
        'n_jobs': -1
    }

    m = xgb.XGBRegressor(**params)
    m.fit(x, y, sample_weight=weights)
    return m

def run_cv(data, cols):
    seasons = ['2022-23','2023-24','2024-25']
    d = data[data['Season'].isin(seasons) & data['Vegas_OU'].notna() & data['Vegas_Error_Target'].notna()].copy()
    d = d[d['Team'] != 'League Average']
    res = []

    for i in range(1,len(seasons)-1):
        train_season = seasons[:i]
        test_season = seasons[i]

        tr = d[d['Season'].isin(train_season)]
        ts = d[d['Season']==test_season]

        x_train = tr[cols]
        y_train = tr['Vegas_Error_Target']
        test = ts[cols]
        vegas_vals = ts['Vegas_OU']
        yreal = ts['Target_Wins']

        model = fitModel(x_train, y_train, tr['Season'])
        pred = model.predict(test)
        win_pred = vegas_vals.values + pred

        err = mean_absolute_error(yreal, win_pred)
        r2val = r2_score(yreal, win_pred)

        res.append({'TrainSzns': train_season, 'TestSzn': test_season, 'MAE': err, 'R2': r2val})

    return pd.DataFrame(res)

def model_io(model, path, mode):
    if mode == 'save':
        joblib.dump(model, path)
    elif mode == 'load':
        return joblib.load(path)
