import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import numpy as np

# ============================================
# 1. ROSTER CONTINUITY SCRAPER
# ============================================

def scrape_roster_continuity(start_year=2015, end_year=2025):
    """
    Scrape roster continuity data (% of minutes returning)
    """
    print("="*80)
    print("SCRAPING ROSTER CONTINUITY")
    print("="*80)
    
    team_abbrevs = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BRK',
        'Charlotte Hornets': 'CHO', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHO',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
    continuity_data = []
    
    for year in range(start_year + 1, end_year + 1):  # Start from 2016 (need 2015 for comparison)
        season = f"{year-1}-{str(year)[-2:]}"
        print(f"\nProcessing {season}...")
        
        for team_name, abbrev in team_abbrevs.items():
            try:
                # Get previous season roster
                url_prev = f"https://www.basketball-reference.com/teams/{abbrev}/{year-1}.html"
                response_prev = requests.get(url_prev, headers={'User-Agent': 'Mozilla/5.0'})
                time.sleep(2)
                
                if response_prev.status_code == 200:
                    tables_prev = pd.read_html(response_prev.content)
                    roster_prev = None
                    
                    # Find roster table
                    for table in tables_prev:
                        if 'Player' in table.columns and 'MP' in table.columns:
                            roster_prev = table
                            break
                    
                    if roster_prev is None:
                        print(f"  ⚠️  No roster found for {team_name} {year-1}")
                        continue
                    
                    # Clean previous roster
                    roster_prev = roster_prev[roster_prev['Player'] != 'Player']  # Remove header rows
                    prev_players = {}
                    
                    for _, row in roster_prev.iterrows():
                        player = str(row['Player']).strip()
                        try:
                            minutes = float(row['MP'])
                            prev_players[player] = minutes
                        except:
                            continue
                    
                    total_prev_minutes = sum(prev_players.values())
                    
                    # Get current season roster
                    url_curr = f"https://www.basketball-reference.com/teams/{abbrev}/{year}.html"
                    response_curr = requests.get(url_curr, headers={'User-Agent': 'Mozilla/5.0'})
                    time.sleep(2)
                    
                    if response_curr.status_code == 200:
                        tables_curr = pd.read_html(response_curr.content)
                        roster_curr = None
                        
                        for table in tables_curr:
                            if 'Player' in table.columns:
                                roster_curr = table
                                break
                        
                        if roster_curr is None:
                            continue
                        
                        # Calculate returning minutes
                        returning_minutes = 0
                        new_players = 0
                        
                        for _, row in roster_curr.iterrows():
                            player = str(row['Player']).strip()
                            if player in prev_players:
                                returning_minutes += prev_players[player]
                            else:
                                new_players += 1
                        
                        continuity_pct = (returning_minutes / total_prev_minutes * 100) if total_prev_minutes > 0 else 0
                        
                        continuity_data.append({
                            'Season': season,
                            'Team': team_name,
                            'Returning_Minutes_Pct': round(continuity_pct, 2),
                            'New_Players_Count': new_players,
                            'Total_Prev_Minutes': round(total_prev_minutes, 0)
                        })
                        
                        print(f"  ✓ {team_name}: {continuity_pct:.1f}% minutes returning")
                
            except Exception as e:
                print(f"  ✗ Error with {team_name}: {str(e)}")
                continue
    
    df = pd.DataFrame(continuity_data)
    df.to_csv('nba_roster_continuity.csv', index=False)
    print(f"\n✓ Saved: nba_roster_continuity.csv ({len(df)} records)")
    return df


# ============================================
# 2. COACHING CHANGES SCRAPER
# ============================================

def scrape_coaching_changes(start_year=2015, end_year=2025):
    """
    Scrape head coaching changes
    """
    print("\n" + "="*80)
    print("SCRAPING COACHING CHANGES")
    print("="*80)
    
    # This requires manual collection as coaching data isn't in a standard table
    # I'll provide a template and you can fill it in
    
    # Manual data collection from Wikipedia or Basketball Reference
    coaching_changes = [
        # 2015-16
        {'Season': '2015-16', 'Team': 'Los Angeles Lakers', 'New_Coach': 'Byron Scott', 'Coaching_Change': 0},
        # ... (you'll need to fill this manually or scrape from Wikipedia)
        
        # 2024-25 examples
        {'Season': '2024-25', 'Team': 'Los Angeles Lakers', 'New_Coach': 'JJ Redick', 'Coaching_Change': 1},
        {'Season': '2024-25', 'Team': 'Detroit Pistons', 'New_Coach': 'JB Bickerstaff', 'Coaching_Change': 1},
        {'Season': '2024-25', 'Team': 'Charlotte Hornets', 'New_Coach': 'Charles Lee', 'Coaching_Change': 1},
        {'Season': '2024-25', 'Team': 'Sacramento Kings', 'New_Coach': 'Mike Brown', 'Coaching_Change': 0},
        # Add all teams for all seasons
    ]
    
    # For now, let's create a template
    print("\n⚠️  COACHING CHANGES REQUIRE MANUAL COLLECTION")
    print("Creating template CSV...")
    
    # Create template with all teams/seasons
    teams = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
             'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets',
             'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers',
             'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
             'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks',
             'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns',
             'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors',
             'Utah Jazz', 'Washington Wizards']
    
    template_data = []
    for year in range(start_year, end_year + 1):
        season = f"{year-1}-{str(year)[-2:]}"
        for team in teams:
            template_data.append({
                'Season': season,
                'Team': team,
                'Head_Coach': '',  # Fill manually
                'Coaching_Change': 0  # 1 if new coach, 0 if same
            })
    
    df = pd.DataFrame(template_data)
    df.to_csv('nba_coaching_changes_TEMPLATE.csv', index=False)
    print("✓ Saved: nba_coaching_changes_TEMPLATE.csv")
    print("  → Fill in 'Head_Coach' and 'Coaching_Change' columns manually")
    print("  → Use: https://www.basketball-reference.com/coaches/")
    
    return df


# ============================================
# 3. INJURY DATA SCRAPER (Up to Oct 20, 2025)
# ============================================

def scrape_injury_data(start_year=2015, end_year=2025):
    """
    Scrape injury data from previous seasons
    NOTE: Current season (2024-25) data only up to Oct 20, 2024
    """
    print("\n" + "="*80)
    print("SCRAPING INJURY DATA")
    print("="*80)
    
    injury_data = []
    
    team_abbrevs = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
    for year in range(start_year, end_year + 1):
        season = f"{year-1}-{str(year)[-2:]}"
        print(f"\nProcessing {season}...")
        
        for team_name, abbrev in team_abbrevs.items():
            try:
                # Get team roster and games played
                url = f"https://www.basketball-reference.com/teams/{abbrev}/{year}.html"
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                time.sleep(2)
                
                if response.status_code == 200:
                    tables = pd.read_html(response.content)
                    roster = None
                    
                    for table in tables:
                        if 'Player' in table.columns and 'G' in table.columns:
                            roster = table
                            break
                    
                    if roster is None:
                        continue
                    
                    roster = roster[roster['Player'] != 'Player']
                    
                    # Calculate injury metrics
                    total_games = 82 if year < 2020 or year > 2021 else 72  # COVID seasons
                    
                    games_missed_top3 = 0
                    injury_prone_count = 0
                    
                    # Sort by minutes played to get top players
                    roster['MP'] = pd.to_numeric(roster['MP'], errors='coerce')
                    roster = roster.sort_values('MP', ascending=False)
                    
                    top_3_players = roster.head(3)
                    
                    for _, player in top_3_players.iterrows():
                        try:
                            games_played = int(player['G'])
                            games_missed = total_games - games_played
                            games_missed_top3 += max(0, games_missed)
                        except:
                            continue
                    
                    # Count injury-prone players (missed 20+ games)
                    for _, player in roster.iterrows():
                        try:
                            games_played = int(player['G'])
                            if (total_games - games_played) >= 20:
                                injury_prone_count += 1
                        except:
                            continue
                    
                    injury_data.append({
                        'Season': season,
                        'Team': team_name,
                        'Games_Missed_Top3': games_missed_top3,
                        'Injury_Prone_Count': injury_prone_count,
                        'Top_Player_Games': int(top_3_players.iloc[0]['G']) if len(top_3_players) > 0 else 0
                    })
                    
                    print(f"  ✓ {team_name}: Top 3 missed {games_missed_top3} games")
                    
            except Exception as e:
                print(f"  ✗ Error with {team_name}: {str(e)}")
                continue
    
    df = pd.DataFrame(injury_data)
    df.to_csv('nba_injury_data.csv', index=False)
    print(f"\n✓ Saved: nba_injury_data.csv ({len(df)} records)")
    return df


# ============================================
# 4. VEGAS ODDS SCRAPER/TEMPLATE
# ============================================

def create_vegas_odds_template(start_year=2015, end_year=2026):
    """
    Create template for Vegas over/under lines
    NOTE: This requires manual collection from sportsbooks
    """
    print("\n" + "="*80)
    print("CREATING VEGAS ODDS TEMPLATE")
    print("="*80)
    
    teams = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
             'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets',
             'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers',
             'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
             'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks',
             'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns',
             'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors',
             'Utah Jazz', 'Washington Wizards']
    
    # Known Vegas lines (you'll need to fill in the rest)
    known_vegas_lines = {
        # 2024-25 Season (from various sportsbooks)
        ('2024-25', 'Boston Celtics'): 58.5,
        ('2024-25', 'Oklahoma City Thunder'): 56.5,
        ('2024-25', 'Denver Nuggets'): 54.5,
        ('2024-25', 'Milwaukee Bucks'): 54.5,
        ('2024-25', 'Philadelphia 76ers'): 50.5,
        ('2024-25', 'New York Knicks'): 53.5,
        ('2024-25', 'Cleveland Cavaliers'): 48.5,
        ('2024-25', 'Phoenix Suns'): 49.5,
        ('2024-25', 'Los Angeles Lakers'): 48.5,
        ('2024-25', 'Dallas Mavericks'): 50.5,
        ('2024-25', 'Minnesota Timberwolves'): 51.5,
        ('2024-25', 'Memphis Grizzlies'): 50.5,
        ('2024-25', 'Golden State Warriors'): 44.5,
        ('2024-25', 'Los Angeles Clippers'): 44.5,
        ('2024-25', 'Miami Heat'): 45.5,
        ('2024-25', 'Sacramento Kings'): 46.5,
        ('2024-25', 'Orlando Magic'): 47.5,
        ('2024-25', 'Indiana Pacers'): 47.5,
        ('2024-25', 'New Orleans Pelicans'): 47.5,
        ('2024-25', 'Houston Rockets'): 42.5,
        ('2024-25', 'Atlanta Hawks'): 36.5,
        ('2024-25', 'Chicago Bulls'): 28.5,
        ('2024-25', 'Brooklyn Nets'): 19.5,
        ('2024-25', 'Toronto Raptors'): 28.5,
        ('2024-25', 'Charlotte Hornets'): 29.5,
        ('2024-25', 'Detroit Pistons'): 25.5,
        ('2024-25', 'Washington Wizards'): 20.5,
        ('2024-25', 'Portland Trail Blazers'): 21.5,
        ('2024-25', 'San Antonio Spurs'): 35.5,
        ('2024-25', 'Utah Jazz'): 28.5,
        
        # 2025-26 Season (PRESEASON - you'll need to update these when available)
        ('2025-26', 'Boston Celtics'): None,  # Fill when lines are released
        ('2025-26', 'Oklahoma City Thunder'): None,
        # ... add all teams
    }
    
    template_data = []
    for year in range(start_year, end_year + 1):
        season = f"{year-1}-{str(year)[-2:]}"
        for team in teams:
            vegas_line = known_vegas_lines.get((season, team), None)
            
            template_data.append({
                'Season': season,
                'Team': team,
                'Vegas_Over_Under': vegas_line,
                'Championship_Odds': None,  # Optional: Add title odds
                'Source': 'DraftKings/Bovada'  # Track where you got the data
            })
    
    df = pd.DataFrame(template_data)
    df.to_csv('nba_vegas_odds.csv', index=False)
    print(f"✓ Saved: nba_vegas_odds.csv ({len(df)} records)")
    print("\n📝 TO COMPLETE THIS FILE:")
    print("  1. Visit: https://www.sportsoddshistory.com/nba-win/")
    print("  2. Visit: https://www.oddsshark.com/nba/odds")
    print("  3. Fill in missing Vegas_Over_Under values")
    print("  4. For 2025-26, wait until preseason lines are released (usually July)")
    
    return df


# ============================================
# 5. MERGE ALL DATA SOURCES
# ============================================

def merge_all_data_sources(base_stats_file='nba_team_stats_2015_2025.csv'):
    """
    Merge all scraped data with base stats
    """
    print("\n" + "="*80)
    print("MERGING ALL DATA SOURCES")
    print("="*80)
    
    # Load base stats
    df_base = pd.read_csv(base_stats_file)
    print(f"✓ Loaded base stats: {df_base.shape}")
    
    # Load additional data
    try:
        df_roster = pd.read_csv('nba_roster_continuity.csv')
        df_base = df_base.merge(df_roster, on=['Season', 'Team'], how='left')
        print(f"✓ Merged roster continuity: {df_roster.shape}")
    except FileNotFoundError:
        print("⚠️  nba_roster_continuity.csv not found")
    
    try:
        df_coaching = pd.read_csv('nba_coaching_changes.csv')
        df_base = df_base.merge(df_coaching, on=['Season', 'Team'], how='left')
        print(f"✓ Merged coaching changes: {df_coaching.shape}")
    except FileNotFoundError:
        print("⚠️  nba_coaching_changes.csv not found")
    
    try:
        df_injury = pd.read_csv('nba_injury_data.csv')
        df_base = df_base.merge(df_injury, on=['Season', 'Team'], how='left')
        print(f"✓ Merged injury data: {df_injury.shape}")
    except FileNotFoundError:
        print("⚠️  nba_injury_data.csv not found")
    
    try:
        df_vegas = pd.read_csv('nba_vegas_odds.csv')
        df_base = df_base.merge(df_vegas, on=['Season', 'Team'], how='left')
        print(f"✓ Merged Vegas odds: {df_vegas.shape}")
    except FileNotFoundError:
        print("⚠️  nba_vegas_odds.csv not found")
    
    # Save merged file
    df_base.to_csv('nba_complete_dataset.csv', index=False)
    print(f"\n✓ Saved complete dataset: nba_complete_dataset.csv")
    print(f"  Final shape: {df_base.shape}")
    
    # Show what's missing
    print("\n📊 DATA COMPLETENESS:")
    for col in df_base.columns:
        missing_pct = df_base[col].isnull().sum() / len(df_base) * 100
        if missing_pct > 0:
            print(f"  {col}: {missing_pct:.1f}% missing")
    
    return df_base


# ============================================
# 6. MAIN EXECUTION
# ============================================

def main():
    """
    Run all scrapers
    """
    print("\n" + "="*80)
    print("NBA ADDITIONAL DATA SCRAPER")
    print("="*80)
    print("\nThis will scrape:")
    print("  1. Roster Continuity (automated)")
    print("  2. Coaching Changes (template - manual fill)")
    print("  3. Injury Data (automated)")
    print("  4. Vegas Odds (template - manual fill)")
    print("\n⚠️  This will take 30-60 minutes due to rate limiting")
    print("="*80)
    
    proceed = input("\nProceed? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Cancelled.")
        return
    
    # 1. Scrape roster continuity (automated)
    print("\n" + "="*80)
    print("STEP 1/4: ROSTER CONTINUITY")
    print("="*80)
    scrape_roster_continuity(start_year=2020, end_year=2025)
    
    # 2. Create coaching template (manual)
    print("\n" + "="*80)
    print("STEP 2/4: COACHING CHANGES")
    print("="*80)
    scrape_coaching_changes(start_year=2020, end_year=2025)
    
    # 3. Scrape injury data (automated)
    print("\n" + "="*80)
    print("STEP 3/4: INJURY DATA")
    print("="*80)
    scrape_injury_data(start_year=2020, end_year=2025)
    
    # 4. Create Vegas template (manual)
    print("\n" + "="*80)
    print("STEP 4/4: VEGAS ODDS")
    print("="*80)
    create_vegas_odds_template(start_year=2020, end_year=2026)
    
    print("\n" + "="*80)
    print("SCRAPING COMPLETE!")
    print("="*80)
    print("\n📝 NEXT STEPS:")
    print("  1. Fill in nba_coaching_changes_TEMPLATE.csv manually")
    print("     → Use: https://www.basketball-reference.com/coaches/")
    print("     → Rename to: nba_coaching_changes.csv when done")
    print("\n  2. Fill in nba_vegas_odds.csv manually")
    print("     → Use: https://www.sportsoddshistory.com/nba-win/")
    print("     → Fill in missing Over/Under values")
    print("\n  3. Run merge_all_data_sources() to combine everything")
    print("="*80)


if __name__ == "__main__":
    main()
    
    # After manual data entry, run this:
    # merge_all_data_sources('nba_team_stats_2015_2025.csv')




