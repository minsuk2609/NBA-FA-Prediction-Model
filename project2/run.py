import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from model.data_pipeline import load_data, create_rows, lagged_features, engineer_features, train_data
from model.model import fitModel, run_cv, model_io
from model.predict import predict

def eval_2024(df, feat_cols, model_path):

    train_df = df[(df['Season'].isin(['2022-23','2023-24'])) & df['Vegas_OU'].notna() & df['Vegas_Error_Target'].notna()].copy()
    test_df = df[df['Season']=='2024-25'].copy()

    if test_df.empty:
        print("No 2024-25 rows found")
        return None

    xtrain = train_df[feat_cols]
    ytrain = train_df['Vegas_Error_Target']
    xtest = test_df[feat_cols]
    vegas_vals = test_df['Vegas_OU']
    actuals = test_df['Target_Wins']

    model = fitModel(xtrain, ytrain, season_train=train_df['Season'])
    pred_err = model.predict(xtest)
    pred_wins = vegas_vals.values + pred_err

    mae = mean_absolute_error(actuals, pred_wins)
    r2 = r2_score(actuals, pred_wins)

    out = pd.DataFrame({
        'Team': test_df['Team'].values,
        'Actual_Wins': actuals.values,
        'Predicted_Wins': np.round(pred_wins,1)
    }).sort_values('Actual_Wins', ascending=False)

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(14,10))
    df_plot = out.melt(id_vars='Team', value_vars=['Actual_Wins','Predicted_Wins'])

    sns.barplot(data=df_plot, x='value', y='Team', hue='variable')
    plt.title("2024–25 Model Evaluation: Actual vs Predicted Wins")
    plt.xlabel("Wins")
    plt.ylabel("Teams")
    plt.legend(title='')

    tx = df_plot['value'].max() * 0.6
    ty = len(out) - 1

    plt.text(tx, ty, f"MAE: {mae:.3f}\nR2: {r2:.3f}", fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.tight_layout()
    out_path = "results/validate_2024_25.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"MAE: {mae:.3f}, R2: {r2:.3f}")

    return out


def heat_corr(df, fcols):
    os.makedirs("results", exist_ok=True)

    corr = df[fcols].corr()

    corr_flat = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_flat.columns = ['Feature1', 'Feature2', 'Correlation']
    corr_flat['AbsCorr'] = corr_flat['Correlation'].abs()

    top12 = corr_flat.sort_values('AbsCorr', ascending=False).head(12)

    top_features = pd.unique(top12[['Feature1', 'Feature2']].values.ravel())

    corr_subset = corr.loc[top_features, top_features]
    scale = (corr_subset + 1) / 2

    plt.figure(figsize=(14, 10))
    sns.heatmap(scale, cmap='coolwarm', annot=True, vmin=0, vmax=1, center=0)
    plt.title("Top 12 Correlated Features Heatmap")
    plt.tight_layout()
    plt.savefig("results/feature_correlation_heatmap.png", dpi=300)
    plt.close()


def win_corr_plot(df, fcols, topn):
    os.makedirs("results", exist_ok=True)

    corr_vals = df[fcols].corrwith(df['Target_Wins']).abs().sort_values(ascending=False)
    top_corrs = corr_vals.head(topn)
    
    print(top_corrs)

    plt.figure(figsize=(14,10))
    sns.barplot(x=top_corrs.values, y=top_corrs.index)
    plt.xlabel("Abs Corr with Wins")
    plt.ylabel("Features")
    plt.title(f"Top {topn} Features Correlated with Wins")
    plt.tight_layout()
    plt.savefig("results/wins_feature_correlation.png", dpi=300)
    plt.close()


def main():
    df = load_data("nba_team_stats_2020_2025.csv")
    df = df[df['Team'] != 'League Average'].copy()
    df = create_rows(df)

    try:
        vegas = pd.read_csv("data/nba_vegas_odds_clean.csv")
        vegas = vegas[vegas['Season'].isin(['2022-23','2023-24','2024-25','2025-26'])]
        df = df.merge(vegas, on=['Season','Team'], how='left')
    except Exception:
        print("No vegas file")

    df = lagged_features(df)
    df = engineer_features(df)

    if 'Vegas_OU' in df.columns:
        df['Vegas_Error_Target'] = df['Target_Wins'] - df['Vegas_OU']
    else:
        df['Vegas_Error_Target'] = None

    hist, pred, feat_cols = train_data(df, prediction_season='2025-26')

    heat_corr(df, feat_cols)
    win_corr_plot(df, feat_cols, topn=10)

    x_train = hist[feat_cols]
    y_train = hist['Vegas_Error_Target']
    season = hist['Season'].values

    model = fitModel(x_train, y_train, season_train=season)
    model_io(model, "saved_models/xgb_model.pkl", mode='save')

    cv = run_cv(df, feat_cols)

    eval_2024(df, feat_cols, model_path="saved_models/xgb_model.pkl")

    preds, _ = predict(df, feat_cols)
    if preds is not None:
        print(preds)

    return df, preds, model, cv


if __name__ == "__main__":
    main()
