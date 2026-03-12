import pandas as pd
import numpy as np
import time
import os
from typing import Any

from newspaper import Article, Config
from tqdm import tqdm
from urllib.parse import urlparse
import sys

# Notes:
# Make sure to look over CONFIGURATION variables
# Include YEAR as argument
# Interupted runs can be resumed.

if len(sys.argv) < 2:
    print("ERROR: Missing YEAR argument.")
    print("Usage: python scrape_by_year.py <YEAR>")
    sys.exit(1)

YEAR = int(sys.argv[1])

print(f"\n--- LAUNCHING SCRAPER FOR YEAR {YEAR} ---\n")

# --- CONFIGURATION ---
INPUT_FILE = 'bq-results-20260128-004024-1769560909144.csv' 
OUTPUT_FILE = f'ven_usa_{YEAR}.csv'
BATCH_SIZE = 200

# Newspaper Config
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10

def prepare_dataframe() -> pd.DataFrame:
    """
    Handles logic to either resume from an existing file OR 
    load raw data and apply the 'Interleave Sort' strategy.
    """
    # 1. Check if we can resume work
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from existing file: {OUTPUT_FILE}")
        df = pd.read_csv(OUTPUT_FILE)
        # Ensure 'Scrape_Status' exists, just in case
        if 'Scrape_Status' not in df.columns:
            df['Scrape_Status'] = 'Pending'
        return df

    # 2. If no existing file, load raw data
    print(f"No partial file found. Loading raw dataset for YEAR: {YEAR}...")
    df_all = pd.read_csv(INPUT_FILE)
    
    # Ensure Year column
    if 'Year' not in df_all.columns:
        df_all['Year'] = df_all['SQLDATE'].astype(str).str[:4].astype(int)

    # Filter Year & Clean URLs
    df = df_all[df_all['Year'] == YEAR].copy()
    df = df[
        (df['SourceURL'].notna()) & 
        (df['SourceURL'] != '') & 
        (df['SourceURL'].str.lower() != 'unspecified')
    ].copy()

    # 3. Apply the "Interleave Sort" (Domain Shuffling)
    print("Optimizing scrape order to minimize domain collisions...")
    df['domain'] = df['SourceURL'].apply(lambda x: urlparse(x).netloc)
    
    # Sort by domain + random tie-breaker
    df['random_key'] = np.random.rand(len(df))
    df = df.sort_values(by=['domain', 'random_key']).reset_index(drop=True)
    
    # Assign Group IDs to interleave
    df['group_id'] = df.groupby('domain').cumcount()
    df = df.sort_values(by='group_id').reset_index(drop=True)
    
    # Clean up (Keep 'domain' this time! It helps the loop speed)
    df = df.drop(columns=['random_key', 'group_id']) 
    
    # Initialize Status Columns
    df['Title'] = None
    df['Text'] = None
    df['Scrape_Status'] = 'Pending'
    df['Error_Details'] = None
    
    print(f"Prepared {len(df)} rows. Saving initial structure...")
    df.to_csv(OUTPUT_FILE, index=False)
    return df

def scrape_article(url: object) -> dict[str, Any]:
    """Execute scrape_article."""
    result = {'Title': None, 'Text': None, 'Scrape_Status': 'Failed', 'Error_Details': ''}
    if not url: return result

    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        if len(article.text) > 0:
            result['Title'] = article.title
            result['Text'] = article.text
            result['Scrape_Status'] = 'Success'
        else:
            result['Scrape_Status'] = 'Empty_Content'
            result['Error_Details'] = 'Parsed successfully but no text found'
    except Exception as e:
        result['Scrape_Status'] = 'Error'
        result['Error_Details'] = str(e)
    return result

def main() -> None:
    # Load Data (either fresh or resumed)
    """Run the script entry point."""
    df_year = prepare_dataframe()
    
    total_rows = len(df_year)
    # Count how many are already done (for resume stats)
    completed_mask = (df_year['Scrape_Status'] != 'Pending') & (df_year['Scrape_Status'].notna())
    processed_count = completed_mask.sum()
    success_count = len(df_year[df_year['Scrape_Status'] == 'Success'])
    
    print(f"\nStarting scrape. Already processed: {processed_count}/{total_rows}")
    
    previous_domain = None
    start_time = time.time()

    # Iterate through the DataFrame
    pbar = tqdm(total=total_rows, initial=processed_count)

    for index, row in df_year.iterrows():
        pbar.update(1)

        # SKIP LOGIC: If we already did this one, move on
        if row['Scrape_Status'] != 'Pending' and pd.notna(row['Scrape_Status']):
            continue
            
        # --- SMART SLEEP (Using the pre-calculated domain column) ---
        # Since we kept 'domain' in prepare_dataframe, we don't need to parse URL again
        current_domain = row['domain'] if 'domain' in row else "unknown"
        
        if previous_domain == current_domain:
            time.sleep(2.0) # Polite sleep for collisions
        
        previous_domain = current_domain
        # ------------------------------------------------------------

        # SCRAPE
        data = scrape_article(row['SourceURL'])
        
        # UPDATE DATAFRAME
        df_year.at[index, 'Title'] = data['Title']
        df_year.at[index, 'Text'] = data['Text']
        df_year.at[index, 'Scrape_Status'] = data['Scrape_Status']
        df_year.at[index, 'Error_Details'] = data['Error_Details']
        
        processed_count += 1
        if data['Scrape_Status'] == 'Success':
            success_count += 1
            
        # BATCH SAVE
        if processed_count % BATCH_SIZE == 0:
            df_year.to_csv(OUTPUT_FILE, index=False)
            elapsed_seconds = time.time() - start_time
            elapsed_hours = elapsed_seconds / 3600
            
            rate = (success_count / processed_count) * 100 if processed_count > 0 else 0
            pbar.set_description(f"Success Rate: {rate:.1f}% | Hours: {elapsed_hours:.2f}")

    pbar.close()
    
    # FINAL SAVE
    df_year.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFINISHED YEAR {YEAR}. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
