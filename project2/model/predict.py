import numpy as np
import pandas as pd
from .create_model import model_io


def predict(data, feats):
    pred_data = data[(data['Season']=='2025-26') & (data['Team']!='League Average')].copy()
    
    if 'Vegas_OU' not in pred_data.columns:
        return None, None

    model = model_io(None, "saved_models/xgb_model.pkl", mode='load')
    x_input = pred_data[feats]
    vegas_vals = pred_data['Vegas_OU'].values


    err_pred = model.predict(x_input)
    raw_wins = vegas_vals + err_pred

    nteams = len(pred_data)
    total_wins = 41 * nteams

    if raw_wins.sum() <= 0:
        raw_wins = 1

    win_scale = total_wins / raw_wins.sum()
    scaled_wins = raw_wins * win_scale

    out = pd.DataFrame({
        'Team': pred_data['Team'],
        'Prediction': np.round(scaled_wins).astype(int),
        '2024-2025 Wins': pred_data['Prev_Adv_W']
    })

    out = out.sort_values('Prediction', ascending=False).reset_index(drop=True)

    return out, model
