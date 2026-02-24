import pandas as pd
import sys
import os
import time
from datetime import datetime
from tqdm import tqdm
import waybackpy
from newspaper import Article, Config


# --- CONFIGURATION ---
if len(sys.argv) < 2:
    print("ERROR: Missing YEAR argument.")
    print("Usage: python rescue_by_year.py <YEAR>")
    sys.exit(1)

YEAR = int(sys.argv[1])
INPUT_FILE = f'ven_usa_{YEAR}.csv' # The file from Scrape Attempt #2
OUTPUT_FILE = f'ven_usa_{YEAR}_rescued.csv' # The new file with rescued data
BATCH_SIZE = 50 # Save more often (Wayback is slow/unstable)
LOG_FILE = f'duration_log_{YEAR}.txt'

# Browser Config
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 20  # Wayback needs more time

def get_wayback_url(original_url, date_int):
    """
    Asks the Internet Archive for the closest snapshot to the article date.
    date_int is YYYYMMDD (integer).
    """
    if not original_url or pd.isna(original_url):
        return None

    try:
        # Convert YYYYMMDD to datetime
        date_str = str(date_int)
        # Handle edge cases where date might be malformed
        if len(date_str) != 8:
            dt = datetime(year=YEAR, month=1, day=1)
        else:
            dt = datetime(year=int(date_str[:4]), month=int(date_str[4:6]), day=int(date_str[6:8]))

        target = waybackpy.Url(original_url, user_agent)
        # Search for a snapshot near the publication date
        snapshot = target.near(year=dt.year, month=dt.month, day=dt.day)
        return snapshot.archive_url
    except Exception:
        return None

def scrape_archived_article(archive_url):
    result = {'Title': None, 'Text': None, 'Status': 'Failed'}
    if not archive_url:
        return result

    try:
        article = Article(archive_url, config=config)
        article.download()
        article.parse()
        
        # Stricter check for archives: must have real length
        if len(article.text) > 100: 
            result['Title'] = article.title
            result['Text'] = article.text
            result['Status'] = 'Success (Archived)'
    except Exception:
        pass
        
    return result

def prepare_dataframe():
    """
    Resumes from OUTPUT_FILE if it exists; otherwise starts from INPUT_FILE.
    """
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from partial rescue file: {OUTPUT_FILE}")
        df = pd.read_csv(OUTPUT_FILE)
    else:
        print(f"Loading original scrape file: {INPUT_FILE}")
        if not os.path.exists(INPUT_FILE):
            print(f"ERROR: Input file {INPUT_FILE} not found!")
            sys.exit(1)
        df = pd.read_csv(INPUT_FILE)
    
    # Ensure SQLDATE exists (renamed to Date sometimes) or Year
    # We need the full date for Wayback targeting
    if 'SQLDATE' not in df.columns and 'Date' in df.columns:
        df['SQLDATE'] = df['Date'] # Standardize
        
    return df

def main():
    print(f"--- LAUNCHING RESCUE MISSION FOR YEAR {YEAR} ---")
    
    df = prepare_dataframe()
    
    # 1. IDENTIFY TARGETS
    # We want rows that are NOT success, and NOT already marked as "Rescue_Failed"
    # This logic prevents infinite loops on restart.
    target_mask = (
        (df['Scrape_Status'] != 'Success') & 
        (df['Scrape_Status'] != 'Success (Archived)') & 
        (df['Scrape_Status'] != 'Rescue_Failed')
    )
    
    targets = df[target_mask]
    total_targets = len(targets)
    
    print(f"Found {total_targets} failed URLs eligible for rescue.")
    
    if total_targets == 0:
        print("Nothing to rescue! Exiting.")
        return

    # Stats counters
    rescued_count = 0
    processed_count = 0
    
    start_time = time.time()
    
    with open(LOG_FILE, 'w') as f:
        f.write("Index,Duration_Seconds\n")
    
    # 2. ITERATE
    # We iterate over the *index* of the targets so we can update the main 'df'
    for index in tqdm(targets.index):
        loop_start = time.time()
        success = False
        row = df.loc[index]
        original_url = row['SourceURL']
        date_int = row['SQLDATE'] # YYYYMMDD
        
        # A. Get Archive Link
        archive_url = get_wayback_url(original_url, date_int)
        
        if archive_url:
            # B. Scrape Archive
            data = scrape_archived_article(archive_url)
            
            if data['Status'] == 'Success (Archived)':
                # Update DataFrame with SUCCESS
                df.at[index, 'Title'] = data['Title']
                df.at[index, 'Text'] = data['Text']
                df.at[index, 'Scrape_Status'] = 'Success (Archived)'
                df.at[index, 'Error_Details'] = f"Rescued via {archive_url}"
                rescued_count += 1
                success = True
            else:
                # Mark as RESCUE FAILED so we don't try again
                df.at[index, 'Scrape_Status'] = 'Rescue_Failed'
        else:
            # No snapshot found
            df.at[index, 'Scrape_Status'] = 'Rescue_Failed'
            df.at[index, 'Error_Details'] = 'Wayback: No snapshot found'
            
        processed_count += 1
        
        # C. BATCH SAVE
        if processed_count % BATCH_SIZE == 0:
            df.to_csv(OUTPUT_FILE, index=False)
            
            elapsed_seconds = time.time() - start_time
            elapsed_hours = elapsed_seconds / 3600
            rate = (rescued_count / processed_count) * 100
            
            # Print stats to console (tqdm handles the bar, we print log)
            tqdm.write(f"Batch Saved. Rescued: {rescued_count}/{processed_count} ({rate:.1f}%) | Hours: {elapsed_hours:.2f}")

        # SMART SLEEP
        duration = time.time() - loop_start
        with open(LOG_FILE, 'a') as f:
            f.write(f"{index},{duration:.1f},{success},{original_url}\n")
        
        if duration < 2.0:
            time.sleep(1)
        else:
            time.sleep(0.1)

    # 3. FINAL SAVE
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print(f"RESCUE COMPLETE FOR YEAR {YEAR}")
    print("="*40)
    print(f"Total Attempted: {processed_count}")
    print(f"Rescued:         {rescued_count}")
    print(f"Success Rate:    {(rescued_count/processed_count)*100:.2f}%")
    print(f"Saved to:        {OUTPUT_FILE}")

if __name__ == "__main__":
    main()