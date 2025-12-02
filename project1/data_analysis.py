import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df_raw = pd.read_csv("data/Combined P&R Pull Up Shooting.csv")
fa = pd.read_csv("data/2026 Free Agent List.csv")
cap = pd.read_csv("data/NBA Salary Cap.csv")
pred = pd.read_csv("2026 PG Salary Predictions.csv")
adv_raw = pd.read_csv("data/Player Stat Advanced.csv")

df_raw['pid'] = df_raw['PLAYER'].str.lower().str.strip()
fa['pid'] = fa['Player'].str.lower().str.strip()
pred['pid'] = pred['Player'].str.lower().str.strip()
adv_raw['pid'] = adv_raw['player'].str.lower().str.strip()

rename_map = {
    'FGM_PU': 'PU_FGM',
    'FGA_PU': 'PU_FGA',
    'FG%_PU': 'PU_FG',
    '3PM': 'PU_3PM',
    '3PA': 'PU_3PA',
    '3P%': 'PU_3P',
    'eFG%_PU': 'PU_eFG',
    'Freq%_PU': 'PU_freq',
    'PTS_PU': 'PU_pts',
    'PPP': 'PR_ppp',
    'eFG%_PnR': 'PR_eFG',
    'TOVFreq%': 'PR_tov',
    'Freq%': 'PR_freq'
}

df_clean = df_raw.rename(columns=rename_map).fillna(0)

df_clean['vol_fg'] = df_clean['PU_FG'] * (df_clean['PU_FGA'] / df_clean['MIN'].replace(0, np.nan))
df_clean['mix2'] = 0.6*(df_clean['PU_FGM'] - df_clean['PU_3PM']) + 0.4*(df_clean['PU_FGA'] - df_clean['PU_3PA'])
df_clean['mix3'] = 1.5*(0.6*df_clean['PU_3P']) + 0.4*df_clean['PU_3PA']
df_clean['shot_mix'] = df_clean['mix2'] + df_clean['mix3']
df_clean['shot_pts'] = 2*(df_clean['PU_FGM'] - df_clean['PU_3PM']) + 3*df_clean['PU_3PM']

shoot_feats = ['vol_fg', 'shot_mix', 'shot_pts']

df_clean['r_ppp'] = df_clean['PR_ppp'] * df_clean['PR_freq']
df_clean['r_eff'] = df_clean['PR_eFG']
df_clean['r_tov'] = df_clean['PR_tov']

pnr_feats = ['r_ppp', 'r_eff', 'r_tov']

for c in shoot_feats:
    df_clean[c + '_z'] = zscore(df_clean[c])

for c in pnr_feats:
    df_clean[c + '_z'] = -zscore(df_clean[c]) if c == 'r_tov' else zscore(df_clean[c])

z_shoot = [c + '_z' for c in shoot_feats]
z_pnr = [c + '_z' for c in pnr_feats]

df_clean['shot_score'] = df_clean[z_shoot].mean(axis=1)
df_clean['pnr_score'] = df_clean[z_pnr].mean(axis=1)

adv_raw = adv_raw[adv_raw['season'].isin(['2022-23','2023-24','2024-25'])]
w_map = {'2022-23':1, '2023-24':2, '2024-25':3}
adv_raw['w'] = adv_raw['season'].map(w_map)

base_stats = ['offrtg','defrtg','netrtg','ts%','ast%','ast/to','to ratio','bpm','vorp','ws','pts','reb','ast','tov','stl','blk','+/-']
adv_raw[base_stats] = adv_raw[base_stats].apply(pd.to_numeric, errors='coerce')

adv_grp = (
    adv_raw.groupby('pid')
    .apply(lambda g: pd.Series({c: np.average(g[c], weights=g['w']) for c in base_stats}))
    .reset_index()
)

for c in base_stats:
    adv_grp[c + '_z'] = zscore(adv_grp[c])

stat_z_cols = [c + '_z' for c in base_stats]
adv_grp['adv_score'] = adv_grp[stat_z_cols].mean(axis=1)

df_clean = pd.merge(df_clean, adv_grp[['pid','adv_score']], on='pid', how='left')
df_clean['overall'] = (3*df_clean['shot_score'] + 3*df_clean['pnr_score'] + df_clean['adv_score']) / 7



fa['Prev AAV'] = (
    fa['Prev AAV']
    .astype(str)
    .str.replace('[\$,]', '', regex=True)
)

fa['Prev AAV'] = pd.to_numeric(fa['Prev AAV'], errors='coerce')




df_clean = pd.merge(df_clean, fa[['pid','Prev AAV','Age']], on='pid', how='left')
df_clean['Age'] = df_clean['Age'].astype(str).str.extract(r'(\d+)')
df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce').astype('Int64')
df_clean = pd.merge(df_clean, pred[['pid','2026 Salary Prediction']], on='pid', how='left')

df_clean['salary'] = df_clean['2026 Salary Prediction']
df_clean.loc[df_clean['salary'] > 18_000_000, 'salary'] = np.nan

plot1 = df_clean[['PLAYER','shot_score','pnr_score']].drop_duplicates()





plt.figure(figsize=(14,10))
plt.scatter(plot1['pnr_score'], plot1['shot_score'])
for _, r in plot1.iterrows():
    plt.text(r['pnr_score']+0.01, r['shot_score']+0.01, r['PLAYER'], fontsize=8)
plt.xlabel("PnR Score (z)")
plt.ylabel("Pull-Up Score (z)")
plt.title("2026 FA PGs — Shooting vs PnR")
plt.grid(True)
plt.show()

plot2 = df_clean[['PLAYER','salary','overall','Age']].drop_duplicates()




plt.figure(figsize=(14,10))
plt.scatter(plot2['overall'], plot2['salary'])
for _, r in plot2.iterrows():
    plt.text(r['overall']+0.01, r['salary'], f"{r['PLAYER']} ({r['Age']})", fontsize=8)
plt.xlabel("Total Score (z)")
plt.ylabel("Projected Salary")
plt.title("2026 FA PGs and Salary vs Score")
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
plt.show()
