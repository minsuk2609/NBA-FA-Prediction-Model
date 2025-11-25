import pandas as pd

def clean_vegas_odds(input_file, output_file):
    vegas = pd.read_csv(input_file)
    vegas = vegas[['Season', 'Team', 'W-L O/U']].copy()
    vegas.rename(columns={'W-L O/U': 'Vegas_OU'}, inplace=True)
    vegas['Vegas_OU'] = pd.to_numeric(vegas['Vegas_OU'], errors='coerce')
    vegas = vegas.dropna(subset=['Vegas_OU'])
    vegas.to_csv(output_file, index=False)
    return vegas

if __name__ == "__main__":
    clean_vegas_odds(input_file='../data/nba_preseason_odds.csv', output_file='../data/nba_vegas_odds_clean.csv')
