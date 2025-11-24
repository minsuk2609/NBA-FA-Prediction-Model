import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import time
from io import StringIO

def scrape_preseason_odds(start_year=2018, end_year=2025):
    """
    Scrape NBA preseason odds from Basketball Reference
    
    Parameters:
    start_year: First season to scrape
    end_year: Last season to scrape
    
    Returns:
    Combined DataFrame with all preseason odds
    """
    
    all_odds_data = []
    
    for year in range(start_year, end_year + 1):
        print(f"Scraping {year-1}-{str(year)[-2:]} preseason odds...")
        
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_preseason_odds.html"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the preseason odds table
            table = soup.find('table', {'id': 'NBA_preseason_odds'})
    
            if table is None:
                # Check in comments
                comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                for comment in comments:
                    if 'NBA_preseason_odds' in comment:
                        comment_soup = BeautifulSoup(comment, 'html.parser')
                        table = comment_soup.find('table', {'id': 'NBA_preseason_odds'})
                        if table:
                            break
            
            if table is None:
                print(f"  ✗ Preseason odds table not found for {year}")
                time.sleep(3)
                continue
            
            # Parse table
            df = pd.read_html(StringIO(str(table)))[0]
            
            # Clean dataframe
            df = clean_odds_dataframe(df, year)
            
            if df is not None and len(df) > 0:
                all_odds_data.append(df)
                print(f"  ✓ Successfully scraped {year} ({len(df)} teams)")
            else:
                print(f"  ✗ No data extracted for {year}")
            
            time.sleep(3)
            
        except Exception as e:
            print(f"  ✗ Error scraping {year}: {str(e)}")
            continue
    
    # Combine all seasons
    if all_odds_data:
        final_df = pd.concat(all_odds_data, ignore_index=True)
        print(f"\n✓ Total records: {len(final_df)}")
        return final_df
    else:
        print("\n✗ No data collected")
        return None

def clean_odds_dataframe(df, year):
    """
    Clean the preseason odds dataframe
    """
    try:
        # Handle multi-level headers
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns:
                parts = [str(x) for x in col if str(x) != 'nan' and 'Unnamed' not in str(x)]
                if parts:
                    new_cols.append('_'.join(parts))
                else:
                    new_cols.append('')
            df.columns = new_cols
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Find team column
        team_col = None
        for col in df.columns:
            if col in ['Team', 'Tm', 'team', 'tm']:
                team_col = col
                break
        
        if team_col and team_col != 'Team':
            df.rename(columns={team_col: 'Team'}, inplace=True)
        
        # If no Team column, assume first column
        if 'Team' not in df.columns and len(df.columns) > 0:
            df.rename(columns={df.columns[0]: 'Team'}, inplace=True)
        
        # Remove header rows
        if 'Team' in df.columns:
            df['Team'] = df['Team'].astype(str)
            df = df[df['Team'] != 'Team']
            df = df[df['Team'] != 'Tm']
            df = df[df['Team'] != 'nan']
            df = df[df['Team'] != '']
            
            # Remove asterisks and other markers
            df['Team'] = df['Team'].str.replace('*', '', regex=False)
            df['Team'] = df['Team'].str.replace('†', '', regex=False)
            df['Team'] = df['Team'].str.strip()
        
        # Remove Rk column
        if 'Rk' in df.columns:
            df = df.drop('Rk', axis=1)
        
        # Convert numeric columns
        for col in df.columns:
            if col != 'Team':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass
        
        # Add season
        df.insert(0, 'Season', f"{year-1}-{str(year)[-2:]}")
        
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"    Error cleaning odds dataframe: {str(e)}")
        return None

def save_to_csv(df, filename='nba_preseason_odds.csv'):
    if df is not None:
        df.to_csv(filename, index=False)
        print(f"\n✓ Data saved to {filename}")
        print(f"  Shape: {df.shape}")
        return True
    else:
        print("\n✗ No data to save")
        return False

def display_sample_data(df, n=10):
    if df is not None:
        print("\n" + "="*80)
        print(f"SAMPLE DATA (First {n} rows)")
        print("="*80)
        print(df.head(n).to_string(index=False))
        
        print("\n" + "="*80)
        print(f"COLUMNS ({len(df.columns)})")
        print("="*80)
        for i, col in enumerate(df.columns, 1):
            print(f"{i:3d}. {col}")

if __name__ == "__main__":
    
    print("="*80)
    print("NBA PRESEASON ODDS SCRAPER")
    print("="*80)
    print()
    
    df = scrape_preseason_odds(start_year=2018, end_year=2026)
    
    if df is not None:
        display_sample_data(df, n=10)
        save_to_csv(df, 'nba_preseason_odds.csv')
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total Teams: {df['Team'].nunique()}")
        print(f"Seasons: {', '.join(sorted(df['Season'].unique()))}")
        print(f"Total Records: {len(df)}")
    
    print("\n" + "="*80)
    print("SCRAPING COMPLETE")
    print("="*80)



