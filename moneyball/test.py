import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import time
from io import StringIO

def scrape_basketball_reference_seasons(start_year=2015, end_year=2025):
    """
    Scrape NBA team statistics from Basketball Reference for multiple seasons
    
    Parameters:
    start_year: First season to scrape (e.g., 2020 for 2019-20 season)
    end_year: Last season to scrape (e.g., 2025 for 2024-25 season)
    
    Returns:
    Combined DataFrame with all statistics
    """
    
    all_seasons_data = []
    
    for year in range(start_year, end_year + 1):
        print(f"Scraping {year-1}-{str(year)[-2:]} season...")
        
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}.html"
        
        try:
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Dictionary to store all tables for this season
            season_tables = {}
            
            # Define all table IDs we want to scrape
            table_configs = [
                ('per_game-team', 'PG_Team_'),
                ('per_game-opponent', 'PG_Opp_'),
                ('totals-team', 'Tot_Team_'),
                ('totals-opponent', 'Tot_Opp_'),
                ('advanced-team', 'Adv_'),
                ('shooting-team', 'Shoot_Team_'),
                ('shooting-opponent', 'Shoot_Opp_'),
            ]
            
            # Scrape each table
            for table_id, prefix in table_configs:
                print(f"  - Scraping {table_id}...")
                df = extract_table(soup, table_id)
                
                if df is not None:
                    # Check if Team column exists before adding prefix
                    if 'Team' not in df.columns:
                        print(f"No 'Team' column found. Columns: {df.columns.tolist()}")
                        continue
                    
                    df = add_prefix_to_columns(df, prefix, exclude=['Team'])
                    season_tables[table_id] = df
                else:
                    print(f"Table not found")
            
            # Merge all tables for this season
            if season_tables:
                season_df = merge_season_tables(season_tables, year)
                if season_df is not None:
                    all_seasons_data.append(season_df)
                else:
                    print(f"Failed to merge tables for {year}")
            else:
                print(f"No data found for {year}")
                
            time.sleep(3)
            
        except Exception as e:
            print(f"Error scraping {year}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all seasons
    if all_seasons_data:
        final_df = pd.concat(all_seasons_data, ignore_index=True)
        return final_df
    else:
        print("Error combining season stats")
        return None


def extract_table(soup, table_id):
    try:
        # First, try to find the table directly in the HTML
        table = soup.find('table', {'id': table_id})
        
        # If not found, check in HTML comments (Basketball Reference often hides tables there)
        if table is None:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                if table_id in comment:
                    # Parse the comment as HTML
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    table = comment_soup.find('table', {'id': table_id})
                    if table:
                        break
        
        if table is None:
            return None
        
        # Parse table with pandas using StringIO to avoid FutureWarning
        df = pd.read_html(StringIO(str(table)))[0]
        
        # Clean up the dataframe
        df = clean_dataframe(df)
        
        return df
        
    except Exception as e:
        print(f"    Error extracting {table_id}: {str(e)}")
        import traceback
        traceback.print_exc()  # This will show the full error
        return None


def clean_dataframe(df):
    """
    Clean the scraped dataframe
    """
    try:
        # Handle multi-level column headers
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-index columns properly
            new_cols = []
            for col in df.columns:
                # Join non-empty parts of the multi-index
                parts = [str(x) for x in col if str(x) != 'nan' and 'Unnamed' not in str(x)]
                if parts:
                    new_cols.append('_'.join(parts))
                else:
                    new_cols.append('')
            df.columns = new_cols
        
        # Clean column names - convert everything to string first
        df.columns = [str(col).strip() for col in df.columns]
        
        # Remove unnamed columns but keep track of position
        cleaned_cols = []
        for i, col in enumerate(df.columns):
            if 'Unnamed' in col or col == '':
                # Check if this might be the team column (usually first column)
                if i == 0:
                    cleaned_cols.append('Team')
                else:
                    cleaned_cols.append(col)
            else:
                cleaned_cols.append(col)
        df.columns = cleaned_cols
        
        # Find the team column - it could be named differently
        team_col = None
        possible_team_cols = ['Team', 'Tm', 'team', 'tm']
        
        for col in df.columns:
            if col in possible_team_cols:
                team_col = col
                break
        
        # If we found a team column but it's not named 'Team', rename it
        if team_col and team_col != 'Team':
            df.rename(columns={team_col: 'Team'}, inplace=True)
        
        # If still no Team column, the first column is likely the team column
        if 'Team' not in df.columns and len(df.columns) > 0:
            first_col = df.columns[0]
            # Check if first column looks like team names
            try:
                if len(df) > 0:
                    # Get first few non-null values and convert to string
                    sample_values = []
                    for val in df[first_col].head(10):
                        if pd.notna(val):
                            sample_values.append(str(val))
                        if len(sample_values) >= 3:
                            break
                    
                    # Common NBA team name patterns
                    nba_keywords = ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Bulls', 'Nets', 
                                   'Knicks', 'Clippers', 'Suns', 'Mavericks', 'Nuggets', 'Spurs',
                                   'Rockets', 'Thunder', 'Jazz', 'Blazers', 'Kings', 'Pelicans',
                                   'Hawks', 'Hornets', 'Cavaliers', 'Pistons', 'Pacers', 'Bucks',
                                   'Timberwolves', 'Grizzlies', '76ers', 'Raptors', 'Wizards', 'Magic']
                    
                    # Check if any sample contains NBA team keywords
                    has_team_names = any(any(keyword in val for keyword in nba_keywords) for val in sample_values)
                    
                    # Also check if it's just the word "Team" repeated (header row)
                    if has_team_names or (len(sample_values) > 0 and sample_values[0] not in ['Team', 'Tm']):
                        df.rename(columns={first_col: 'Team'}, inplace=True)
            except Exception as e:
                # If detection fails, assume first column is Team
                print(f"    Warning: Could not detect team column, assuming first column. Error: {e}")
                df.rename(columns={first_col: 'Team'}, inplace=True)
        
        # Remove rows that are actually headers (Basketball Reference repeats headers)
        if 'Team' in df.columns:
            # Convert Team column to string for comparison
            df['Team'] = df['Team'].astype(str)
            
            df = df[df['Team'] != 'Team']
            df = df[df['Team'] != 'Tm']
            
            # Remove rows with NaN team names
            df = df[df['Team'] != 'nan']
            df = df[df['Team'] != '']
            
            # Remove asterisks from team names (playoff indicators)
            df['Team'] = df['Team'].str.replace('*', '', regex=False)
            df['Team'] = df['Team'].str.strip()
        
        # Remove 'Rk' (rank) column and empty columns
        cols_to_drop = [col for col in df.columns if col in ['Rk', 'Unnamed', ''] or (col.startswith('Unnamed') and col != 'Team')]
        if cols_to_drop:
            df = df.drop(cols_to_drop, axis=1, errors='ignore')
        
        # Convert numeric columns (without FutureWarning)
        for col in df.columns:
            if col != 'Team':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass  # Keep as is if conversion fails
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    except Exception as e:
        print(f"    Error in clean_dataframe: {str(e)}")
        import traceback
        traceback.print_exc()
        return df  # Return the dataframe as-is if cleaning fails




def add_prefix_to_columns(df, prefix, exclude=None):
    """
    Add prefix to column names except those in exclude list
    """
    if exclude is None:
        exclude = []
    
    new_columns = {}
    for col in df.columns:
        if col not in exclude:
            new_columns[col] = prefix + col
        else:
            new_columns[col] = col
    
    df = df.rename(columns=new_columns)
    return df


def merge_season_tables(tables_dict, year):
    """
    Merge all tables for a single season
    """
    try:
        # Start with the first table
        merged_df = None
        
        for table_name, df in tables_dict.items():
            if df is not None and not df.empty:
                # Verify Team column exists
                if 'Team' not in df.columns:
                    print(f"    Warning: 'Team' column missing in {table_name}")
                    continue
                
                if merged_df is None:
                    merged_df = df.copy()
                else:
                    # Merge on Team
                    merged_df = pd.merge(merged_df, df, on='Team', how='outer')
        
        # Add season column
        if merged_df is not None:
            merged_df.insert(0, 'Season', f"{year-1}-{str(year)[-2:]}")
            # Reorder to have Team as second column
            cols = merged_df.columns.tolist()
            if 'Team' in cols:
                cols.remove('Team')
                cols.insert(1, 'Team')
                merged_df = merged_df[cols]
        
        return merged_df
    
    except Exception as e:
        print(f"    Error merging tables: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def save_to_csv(df, filename='nba_team_stats_2020_2025.csv'):
    """
    Save the dataframe to CSV
    """
    if df is not None:
        df.to_csv(filename, index=False)
        print(f"\n✓ Data saved to {filename}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {len(df.columns)}")
        return True
    else:
        print("\n✗ No data to save")
        return False


def display_sample_data(df, n=3):
    """
    Display sample of the data
    """
    if df is not None:
        print("\n" + "="*80)
        print(f"SAMPLE DATA (First {n} rows)")
        print("="*80)
        
        # Show just a few key columns for readability
        key_cols = ['Season', 'Team', 'PG_Team_PTS', 'PG_Opp_PTS', 'Adv_W', 'Adv_L']
        available_cols = [col for col in key_cols if col in df.columns]
        
        if available_cols:
            print(df[available_cols].head(n).to_string())
        else:
            print(df.iloc[:n, :10])  # Show first 10 columns
        
        print("\n" + "="*80)
        print(f"TOTAL COLUMNS: {len(df.columns)}")
        print("="*80)
        print("\nColumn categories:")
        print(f"  - PG_Team_* : {len([c for c in df.columns if c.startswith('PG_Team_')])} columns")
        print(f"  - PG_Opp_*  : {len([c for c in df.columns if c.startswith('PG_Opp_')])} columns")
        print(f"  - Tot_Team_*: {len([c for c in df.columns if c.startswith('Tot_Team_')])} columns")
        print(f"  - Tot_Opp_* : {len([c for c in df.columns if c.startswith('Tot_Opp_')])} columns")
        print(f"  - Adv_*     : {len([c for c in df.columns if c.startswith('Adv_')])} columns")


def display_all_columns(df):
    """
    Display all column names
    """
    if df is not None:
        print("\n" + "="*80)
        print("ALL COLUMNS")
        print("="*80)
        for i, col in enumerate(df.columns, 1):
            print(f"{i:3d}. {col}")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    print("="*80)
    print("NBA TEAM STATISTICS SCRAPER - Basketball Reference")
    print("="*80)
    print()
    
    # Scrape data from 2020-2025 seasons
    df = scrape_basketball_reference_seasons(start_year=2018, end_year=2025)
    
    if df is not None:
        # Display sample
        display_sample_data(df, n=3)
        
        # Save to CSV
        save_to_csv(df, 'nba_team_stats_2020_2025.csv')
        
        # Print summary statistics
        print("\n" + "="*80)
        print("DATA SUMMARY")
        print("="*80)
        print(f"Total Teams: {df['Team'].nunique()}")
        print(f"Seasons: {', '.join(sorted(df['Season'].unique()))}")
        print(f"Total Records: {len(df)}")
        print(f"Total Columns: {len(df.columns)}")
        
        # Check for missing data
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        if missing_pct.max() > 0:
            print("\nColumns with missing data (top 10):")
            print(missing_pct[missing_pct > 0].head(10))
        
        # Show all columns
        display_all_columns(df)
    
    print("\n" + "="*80)
    print("SCRAPING COMPLETE")
    print("="*80)



