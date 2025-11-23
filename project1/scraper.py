import pandas as pd
import unicodedata

def clean_name_for_matching(name):
    name = str(name).strip().replace('.', '')
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    name = name.lower()
    name = ' '.join(name.split())
    return name

def extract_free_agent_data():
    print("="*80)
    print("EXTRACTING 2026 FREE AGENT DATA")
    print("="*80)
    
    # Load free agent list
    fa_list = pd.read_csv("2026 Free Agent List.csv")
    fa_list.columns = fa_list.columns.str.lower().str.strip()
    fa_list['player_clean'] = fa_list['player'].apply(clean_name_for_matching)
    
    print(f"\nLoaded {len(fa_list)} free agents")
    
    # Load training data
    training_data = pd.read_csv("player_stat_2010-2024_with_advanced.csv")
    training_data.columns = training_data.columns.str.strip()
    training_data['player_clean'] = training_data['player'].apply(clean_name_for_matching)
    
    print(f"Loaded training data: {len(training_data):,} records")
    
    # Extract free agent data
    fa_data_all = training_data[training_data['player_clean'].isin(fa_list['player_clean'])].copy()
    
    # Get latest season for each player
    fa_data_sorted = fa_data_all.sort_values(['player_clean', 'season'])
    fa_latest = fa_data_sorted.groupby('player_clean').tail(1).reset_index(drop=True)
    
    # Check matches
    matched_players = set(fa_latest['player_clean'].unique())
    fa_players = set(fa_list['player_clean'].unique())
    not_found = fa_players - matched_players
    
    print(f"Matched: {len(matched_players)}/{len(fa_list)} players")
    
    if not_found:
        print(f"\nNot found ({len(not_found)}):")
        for player_clean in sorted(not_found):
            orig_name = fa_list[fa_list['player_clean'] == player_clean]['player'].iloc[0]
            print(f"  - {orig_name}")
    
    # Save
    fa_latest.to_csv("2026 Free Agent Prediction Data.csv", index=False)
    print(f"\n✓ Saved: 2026 Free Agent Prediction Data.csv ({len(fa_latest)} players)")
    
    return fa_latest

if __name__ == "__main__":
    fa_data = extract_free_agent_data()
    print("\n" + "="*80)
    print("READY FOR PREDICTION")
    print("="*80)



