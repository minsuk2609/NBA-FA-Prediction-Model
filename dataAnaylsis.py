import pandas as pd
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#DATA LOADING SECTION
#Load the csv files created from csvMaker.py
shooting = pd.read_csv("2024-2025 Guard Pull Up Shooting FA2026.csv")
pr_eff = pd.read_csv("2024-2025 P&R Eff FA2026.csv")

#NBA Salary Cap data is from: https://www.basketball-reference.com/contracts/salary-cap-history.html, converted to csv
salary_cap = pd.read_csv("NBA Salary Cap.csv")

#MERGING DATAFRAMES SECTION
#Free agent list to get previous AAVs
fa_list_df = pd.read_csv("2026 Free Agent List.csv")

#Merge the two dataframes on PLAYER, adding suffixes to distinguish columns such as FG% and eFG%
df = pd.merge(shooting, pr_eff, on="PLAYER", suffixes=('_shooting', '_pr_eff'))

#Shouldn't have any NaNs, but safety call just in case
df.fillna(0, inplace=True)


#DATA ANALYSIS SECTION
#Calculate per volume metrics for shooting, so that low volume shooters aren't overly rewarded
df['FG_volume'] = df['Pull Up FG%'] * df['Freq%_shooting']
df['Pull_up_2P_eff'] = (0.6 * (df['Pull Up FGM'] - df['Pull Up 3PM'])) + (0.4 * (df['Pull Up FGA'] - df['Pull Up 3PA']))
df['Pull_up_3P_eff'] = 1.5 * (0.6 * df['Pull Up 3P%']) + (0.4 * df['Pull Up 3PA'])
df['Pull_up_eff_combined'] = df['Pull_up_2P_eff'] + df['Pull_up_3P_eff']
df['Pull_up_points'] = (2 * (df['Pull Up FGM'] - df['Pull Up 3PM'])) + (3 * df['Pull Up 3PM'])


#List of metrics to be z-scored for pull up shooting
shooting_metrics = ['FG_volume', 'Pull_up_eff_combined', 'Pull_up_points']

#Calculate per volume metrics for pick and roll efficiency, account for frequency so low frequency aren't overly rewarded
df['PPP_pr_eff'] = df['PPP'] * df['Freq%_pr_eff']
df['eFG_pr_eff'] = df['eFG%']
df['TOV_pr_eff'] = df['TOV Freq%']

#List of metrics to be z-scored for pick and roll efficiency
pr_metrics = ['PPP_pr_eff', 'eFG_pr_eff', 'TOV_pr_eff']

#For shooting metrics, just calculate z-scores
for metrics in shooting_metrics:
    df[metrics + '_zscore'] = zscore(df[metrics])

#For pick and roll metrics, invert TOV z-score since lower is better
for metrics in pr_metrics:
    if metrics == 'TOV_pr_eff':
        df[metrics + '_zscore'] = -zscore(df[metrics])
    else:
        df[metrics + '_zscore'] = zscore(df[metrics])

#Calculate total z-score as weighted average of shooting and pick and roll z-scores
shooting_zscore = [m + '_zscore' for m in shooting_metrics]
pr_zscore = [m + '_zscore' for m in pr_metrics]

#50% weight to shooting, 50% weight to pick and roll, so that it is fairly balanced
df['Total_zscore'] = (0.5 * df[shooting_zscore].sum(axis=1)) + (0.5 * df[pr_zscore].sum(axis=1))

df['Shooting_Score'] = df[shooting_zscore].mean(axis=1)
df['PR_Score'] = df[pr_zscore].mean(axis=1)



#PROJECTED SALARY CALCULATION SECTION
#Calculate average annual increase in salary cap to use to calculate projected salaries
salary_cap['Salary Cap'] = salary_cap['Salary Cap'].replace({'\$':'', ',':''}, regex=True).astype(float)

salary_cap['Annual_Increase_%'] = salary_cap['Salary Cap'].pct_change() + 1

average_increase = salary_cap['Annual_Increase_%'].mean()


fa_list_df['Prev AAV'] = fa_list_df['Prev AAV'].replace({'\$':'', ',':''}, regex=True).astype(float)

#Merge previous AAV into main dataframe
df = pd.merge(df, fa_list_df[['Player', 'Prev AAV', 'Age']], left_on='PLAYER', right_on='Player', how='left')
df['Age'] = df['Age'].astype(int)

#Drop duplicate 'Player' column after merge
df.drop(columns=['Player'], inplace=True)

#Project salaries based on average salary cap increase
df['Projected_Salary'] = df['Prev AAV'] * average_increase

#Filter out players with projected salaries above $18 million, since the team has $18 million in cap space for a 2026 free agent signing
df['Projected_Salary'] = df[df['Projected_Salary'] <= 18_000_000]['Projected_Salary']



#DATA VISUALIZATION SECTION
#Scatter plot of Pull-up Shooting Score vs Pick & Roll Score
table1 = df[['PLAYER', 'Shooting_Score', 'PR_Score']].copy()
table1.drop_duplicates(subset=['PLAYER'], keep='first', inplace=True)

plt.figure(figsize=(12, 10))
plt.scatter(table1['PR_Score'], table1['Shooting_Score'], color='blue')

for i, player in enumerate(table1['PLAYER']):
    plt.text(table1['PR_Score'].iloc[i]+0.01, table1['Shooting_Score'].iloc[i]+0.01, player, fontsize=8)

plt.xlabel('Pick & Roll Score (z-score)')
plt.ylabel('Pull-up Shooting Score (z-score)')
plt.title('2026 Free Agent Point Guards: Pull-up Shooting vs Pick & Roll Effectiveness')
plt.grid(True)
plt.show()

print(df.columns)
#Scatter plot of Total Z Score vs Projected Salary
table2 = df[['PLAYER', 'Projected_Salary', 'Total_zscore', 'Age']].copy()
table2.drop_duplicates(subset=['PLAYER'], keep='first', inplace=True)
print(table2)


plt.figure(figsize=(12, 10))
plt.scatter(table2['Total_zscore'], table2['Projected_Salary'], color='blue')

for i, row in table2.iterrows():
    label = f"{row['PLAYER']} ({row['Age']})"
    plt.text(
        row['Total_zscore'] + 0.01,
        row['Projected_Salary'],
        label,
        fontsize=8
    )

plt.xlabel('Total Score (z-score)')
plt.ylabel('Projected Salary')

plt.title('2026 Free Agent Point Guards: Total Score vs Projected Salary (Under $18M)')
plt.grid(True)

#Format y-axis as currency
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

plt.show()