import pandas as pd
import numpy as np

#This program is just to make a new csv that only contains the 2026 Free Agent Point Guards with the relevant stats such as pull up shooting and P&R efficiency

#Grab the list of 2026 Free Agent Point Guards, from: https://www.spotrac.com/nba/free-agents/available/_/year/2026/position/pg
#Since NBA.com stats provide Guards and not point guards specifically, as well as the 2026 free agents, need to filter players
fa_list_df = pd.read_csv("2026 Free Agent List.csv")

fa_2026_pg = fa_list_df['Player'].tolist()

#Grab the csv files for Pull up Shooting and P&R Efficiency, from:
#https://www.nba.com/stats/players/pullup?PlayerPosition=G&Season=2024-25 for pull up shooting
#https://www.nba.com/stats/players/ball-handler?SeasonYear=2024-25 for P&R efficiency
csv_files = [
    ("2024-2025 Guard Pull Up Shooting.csv", "2024-2025 Guard Pull Up Shooting FA2026.csv"),
    ("2024-2025 P&R Eff.csv", "2024-2025 P&R Eff FA2026.csv")
]

#Process each file
for input_file, output_file in csv_files:
    
    df = pd.read_csv(input_file)
    
    #Drop the non-FA Point Guards
    df = df[df['PLAYER'].isin(fa_2026_pg)]
    
    # Replace '-' with NaN
    df.replace('-', np.nan, inplace=True)
    
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            pass
    
    #Save to new CSV
    df.to_csv(output_file, index=False)