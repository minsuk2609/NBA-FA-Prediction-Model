import pandas as pd
import numpy as np

# ============================================
# DIAGNOSE TRAINING DATA ISSUES
# ============================================

def diagnose_training_data():
    print("="*80)
    print("TRAINING DATA DIAGNOSTIC")
    print("="*80)
    
    # Load training data
    stats_df = pd.read_csv("player_stat_2010-2024_with_advanced.csv")
    stats_df.columns = stats_df.columns.str.lower().str.strip()
    
    salaries_df = pd.read_csv("salaries_2020_2025_adjusted.csv")
    salaries_df.columns = salaries_df.columns.str.lower().str.strip()
    
    print(f"\nStats data: {stats_df.shape}")
    print(f"Salary data: {salaries_df.shape}")
    
    # Check salary data
    print("\n" + "="*80)
    print("SALARY DATA CHECK")
    print("="*80)
    print(f"Columns: {salaries_df.columns.tolist()}")
    print(f"\nSample salaries:")
    print(salaries_df[['player', 'year', 'adjusted_salary']].head(10))
    
    print(f"\nSalary statistics:")
    print(f"  Min: ${salaries_df['adjusted_salary'].min():,.0f}")
    print(f"  Max: ${salaries_df['adjusted_salary'].max():,.0f}")
    print(f"  Mean: ${salaries_df['adjusted_salary'].mean():,.0f}")
    print(f"  Median: ${salaries_df['adjusted_salary'].median():,.0f}")
    
    # Check stats data
    print("\n" + "="*80)
    print("STATS DATA CHECK")
    print("="*80)
    print(f"Columns: {stats_df.columns.tolist()}")
    
    # Check for key columns
    key_cols = ['player', 'season', 'pts', 'ast', 'reb', 'ts%', 'vorp']
    missing = [col for col in key_cols if col not in stats_df.columns]
    if missing:
        print(f"\n⚠️  Missing columns: {missing}")
    
    print(f"\nSample stats:")
    available_cols = [col for col in key_cols if col in stats_df.columns]
    print(stats_df[available_cols].head(10))
    
    # Merge and check
    print("\n" + "="*80)
    print("MERGE CHECK")
    print("="*80)
    
    # Normalize names
    def normalize_name(name):
        import unicodedata
        return unicodedata.normalize('NFKD', str(name)).encode('ascii','ignore').decode('utf-8').strip().lower()
    
    stats_df['player'] = stats_df['player'].apply(normalize_name)
    salaries_df['player'] = salaries_df['player'].apply(normalize_name)
    
    salaries_df['season'] = salaries_df['year'].apply(lambda x: f"{x}-{str(x+1)[-2:]}")
    
    merged = stats_df.merge(
        salaries_df[['player', 'season', 'adjusted_salary']],
        on=['player', 'season'],
        how='inner'
    )
    
    print(f"Merged records: {len(merged)}")
    print(f"Unique players: {merged['player'].nunique()}")
    
    # Check merged data quality
    print(f"\nMerged salary range:")
    print(f"  Min: ${merged['adjusted_salary'].min():,.0f}")
    print(f"  Max: ${merged['adjusted_salary'].max():,.0f}")
    print(f"  Mean: ${merged['adjusted_salary'].mean():,.0f}")
    
    # Check if stats vary
    print(f"\nStats variation:")
    print(f"  PTS range: {merged['pts'].min():.1f} - {merged['pts'].max():.1f}")
    print(f"  AST range: {merged['ast'].min():.1f} - {merged['ast'].max():.1f}")
    
    # Sample high and low salary players
    print("\n" + "="*80)
    print("HIGH SALARY PLAYERS")
    print("="*80)
    high_sal = merged.nlargest(5, 'adjusted_salary')
    print(high_sal[['player', 'season', 'pts', 'ast', 'ts%', 'adjusted_salary']])
    
    print("\n" + "="*80)
    print("LOW SALARY PLAYERS")
    print("="*80)
    low_sal = merged.nsmallest(5, 'adjusted_salary')
    print(low_sal[['player', 'season', 'pts', 'ast', 'ts%', 'adjusted_salary']])
    
    # Check your free agent data
    print("\n" + "="*80)
    print("FREE AGENT DATA CHECK")
    print("="*80)
    
    fa_data = pd.read_csv("2026 Free Agent Prediction Data.csv")
    print(f"Free agent records: {len(fa_data)}")
    print(f"\nSample free agent stats:")
    print(fa_data[['player', 'season', 'pts', 'ast', 'ts%']].head(10))
    
    # Compare Trae Young in both datasets
    print("\n" + "="*80)
    print("TRAE YOUNG COMPARISON")
    print("="*80)
    
    trae_training = merged[merged['player'].str.contains('trae', case=False, na=False)]
    if len(trae_training) > 0:
        print("In training data:")
        print(trae_training[['player', 'season', 'pts', 'ast', 'adjusted_salary']].tail(3))
    
    trae_fa = fa_data[fa_data['player'].str.contains('trae', case=False, na=False)]
    if len(trae_fa) > 0:
        print("\nIn free agent data:")
        print(trae_fa[['player', 'season', 'pts', 'ast']])

if __name__ == "__main__":
    diagnose_training_data()



