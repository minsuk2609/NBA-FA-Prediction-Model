import pandas as pd
import numpy as np
import joblib

def clean_format(v):
    if pd.isna(v): 
        return 0
    if isinstance(v,(int,float)):
        return float(v)
    return float(str(v).replace('$','').replace(',',''))

def predict():
    model = joblib.load('salary_model.pkl')
    qt = joblib.load('salary_transformer.pkl')
    feats = joblib.load('feature_names.pkl')

    fa_data = pd.read_csv("data/2026 Free Agent Prediction Data.csv")
    fa_list = pd.read_csv("data/2026 Free Agent List.csv")
    fa_list.columns = fa_list.columns.str.lower()

    pos_cols = ['pos_C-PF','pos_PF','pos_PF-C','pos_PF-SF','pos_PG',
                'pos_PG-SG','pos_SF','pos_SF-PF','pos_SF-SG','pos_SG',
                'pos_SG-PG','pos_SG-SF']

    for c in pos_cols:
        if c not in fa_data.columns:
            fa_data[c] = 0

    fa_data['pos_PG'] = 1

    fa_list['player_lower'] = fa_list['player'].str.lower()
    fa_data['player_lower'] = fa_data['player'].str.lower()

    sal_map = dict(zip(fa_list['player_lower'], fa_list['prev aav']))
    fa_data['adjusted_salary'] = fa_data['player_lower'].map(sal_map)
    fa_data['adjusted_salary'] = fa_data['adjusted_salary'].apply(clean_format)

    X = fa_data[feats].fillna(0)

    y_pred_t = model.predict(X)
    y_dollars = qt.inverse_transform(y_pred_t.reshape(-1,1)).ravel()

    res = pd.DataFrame({
        'Player': fa_data['player'],
        'Age': fa_data['age'],
        'Season': fa_data['season'],
        'GP': fa_data['gp'],
        'PPG': fa_data['pts'],
        'APG': fa_data['ast'],
        'RPG': fa_data['reb'],
        'TS%': (fa_data['ts%']*100).round(1),
        'VORP': fa_data['vorp'],
        'Current Salary': fa_data['adjusted_salary'],
        '2026 Salary Prediction': y_dollars.round(0)
    })

    res['Change'] = res['2026 Salary Prediction'] - res['Current Salary']
    res['Pct_Change'] = ((res['Change'] / res['Current Salary']) * 100).round(1)

    res = res.sort_values('2026 Salary Prediction', ascending=False)
    res = res.reset_index(drop=True)
    res.index = res.index + 1

    res.to_csv("2026 PG Salary Predictions.csv", index=False)

    return res

if __name__ == "__main__":
    out = predict()
