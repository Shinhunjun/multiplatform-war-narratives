"""
Data loader for GDELT discourse analysis pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

from .config import AnalysisConfig, EVENT_CATEGORY_MAP


def _extract_domain(url: object) -> Optional[str]:
    """Extract normalized domain from URL.
    
    Args:
        url (object): Source URL value.
    
    Returns:
        Optional[str]: Result when available; otherwise None.
    """
    if url is None or pd.isna(url):
        return None

    value = str(url).strip()
    if not value:
        return None

    try:
        parsed = urlparse(value)
        netloc = parsed.netloc or parsed.path
        if not netloc:
            return None
        netloc = netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return None


def load_url_lookup(config: AnalysisConfig) -> pd.DataFrame:
    """Load preprocessing URL lookup table used for relevance and duplicate flags.
    
    Args:
        config (AnalysisConfig): Analysis configuration object containing paths and runtime options.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    path = config.url_lookup_path
    if not path.exists():
        print(f"URL lookup not found at {path}, skipping preprocessing merge.")
        return pd.DataFrame()

    usecols = [
        "url_id",
        "SourceURL_Canonical",
        "doc_relevance_sum",
        "doc_relevance_matches",
        "doc_token_count",
        "doc_relevance_score",
        "text_len_norm",
        "text_hash",
        "domain",
        "text_hash_url_count",
        "text_hash_domain_count",
        "suspect_redirect_content",
        "row_count",
    ]

    print(f"Loading preprocessing lookup from {path}...")
    lookup = pd.read_csv(path, usecols=lambda c: c in set(usecols), low_memory=False)
    print(f"Loaded {len(lookup):,} lookup rows")
    return lookup


def load_relevant_terms(config: AnalysisConfig, top_k: int = 200) -> pd.DataFrame:
    """Load relevant terms table for optional downstream interpretation.
    
    Args:
        config (AnalysisConfig): Analysis configuration object containing paths and runtime options.
        top_k (int): Value for `top_k`. Defaults to 200.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    path = config.relevant_terms_path
    if not path.exists():
        return pd.DataFrame(columns=["token", "relevance_score"])

    terms = pd.read_csv(path, low_memory=False)
    if "relevance_score" in terms.columns:
        terms = terms.sort_values("relevance_score", ascending=False)
    return terms.head(top_k).reset_index(drop=True)


def load_relevance_tokens(config: AnalysisConfig, top_k: int = 2000) -> pd.DataFrame:
    """Load token relevance scores from preprocessing output.
    
    Args:
        config (AnalysisConfig): Analysis configuration object containing paths and runtime options.
        top_k (int): Value for `top_k`. Defaults to 2000.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    path = config.relevance_tokens_path
    if not path.exists():
        return pd.DataFrame(columns=["token", "relevance_score"])

    tokens = pd.read_csv(path, low_memory=False)
    if "relevance_score" in tokens.columns:
        tokens = tokens.sort_values("relevance_score", ascending=False)
    return tokens.head(top_k).reset_index(drop=True)


def load_gdelt_events(
    config: AnalysisConfig,
    merge_lookup: bool = True,
) -> pd.DataFrame:
    """Load and normalize scraped GDELT events.
    
    Args:
        config (AnalysisConfig): Analysis configuration object containing paths and runtime options.
        merge_lookup (bool): Whether to merge URL lookup metadata into loaded event rows. Defaults to True.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    data_path = config.gdelt_csv_path
    if not data_path.exists():
        raise FileNotFoundError(f"GDELT CSV not found: {data_path}")

    print(f"Loading GDELT scraped data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)

    required_columns = [
        "Date",
        "Actor1Name",
        "Actor1CountryCode",
        "Actor2Name",
        "Actor2CountryCode",
        "EventCode",
        "QuadClass",
        "GoldsteinScale",
        "AvgTone",
        "SourceURL",
        "Title",
        "Text",
        "Scrape_Status",
        "url_id",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"GDELT CSV missing required columns: {missing}")

    # Parse date and build temporal fields.
    date_str = df["Date"].astype(str).str.extract(r"(\d{8})")[0]
    df["created_datetime"] = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    df["year"] = df["created_datetime"].dt.year
    df["month"] = df["created_datetime"].dt.month
    df["year_month"] = df["created_datetime"].dt.to_period("M").astype(str)
    df["date"] = df["created_datetime"].dt.date

    # Normalize event fields.
    df["event_code"] = pd.to_numeric(df["EventCode"], errors="coerce")
    df["quad_class"] = pd.to_numeric(df["QuadClass"], errors="coerce").astype("Int64")
    df["event_category"] = df["quad_class"].map(EVENT_CATEGORY_MAP)
    df["goldstein_scale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["avg_tone"] = pd.to_numeric(df["AvgTone"], errors="coerce")

    # Normalize actor fields.
    df["actor1_name"] = df["Actor1Name"].fillna("").astype(str)
    df["actor2_name"] = df["Actor2Name"].fillna("").astype(str)
    df["actor1_country_code"] = df["Actor1CountryCode"].fillna("").astype(str).str.upper()
    df["actor2_country_code"] = df["Actor2CountryCode"].fillna("").astype(str).str.upper()
    df["actor_pair"] = df["actor1_country_code"] + "->" + df["actor2_country_code"]

    # Normalize source fields.
    df["source_url"] = df["SourceURL"].fillna("").astype(str)
    df["source_domain"] = df["SourceURL"].apply(_extract_domain)
    df["title"] = df["Title"].fillna("").astype(str)
    df["body"] = df["Text"].fillna("").astype(str)

    # Analysis text: title + scraped body.
    df["text"] = (df["title"] + " " + df["body"]).str.replace(r"\s+", " ", regex=True).str.strip()
    df["text_length"] = df["text"].str.len()

    # Keep successful scrape variants if configured.
    if config.require_successful_scrape and "Scrape_Status" in df.columns:
        status_mask = df["Scrape_Status"].fillna("").str.contains("success", case=False)
        before = len(df)
        df = df[status_mask].copy()
        print(f"Filtered to successful scrape rows: {len(df):,} / {before:,}")

    # Remove empty text rows.
    before = len(df)
    df = df[df["text"].str.strip() != ""].copy()
    print(f"Filtered to rows with non-empty text: {len(df):,} / {before:,}")

    # Merge preprocessing lookup outputs.
    if merge_lookup:
        lookup = load_url_lookup(config)
        if not lookup.empty:
            df = df.merge(lookup, on="url_id", how="left", validate="many_to_one")
            # Prefer domain from lookup when URL parsing fails.
            if "domain" in df.columns:
                df["source_domain"] = df["source_domain"].fillna(df["domain"])

    # Optional relevance threshold.
    if config.min_doc_relevance_score is not None and "doc_relevance_score" in df.columns:
        before = len(df)
        threshold = float(config.min_doc_relevance_score)
        df = df[df["doc_relevance_score"].fillna(0.0) >= threshold].copy()
        print(f"Applied relevance threshold >= {threshold}: {len(df):,} / {before:,}")

    # Optional duplicate-content filter.
    if config.exclude_suspect_redirect_content and "suspect_redirect_content" in df.columns:
        before = len(df)
        df = df[~df["suspect_redirect_content"].fillna(False)].copy()
        print(f"Removed suspect redirect content rows: {len(df):,} / {before:,}")

    # Sort and add canonical IDs used by downstream stages.
    df = df.sort_values("created_datetime").reset_index(drop=True)
    df["id"] = [f"gdelt_{i}" for i in range(1, len(df) + 1)]
    df["type"] = "event"
    df["scrape_status"] = df["Scrape_Status"].fillna("").astype(str)
    df["error_details"] = df["Error_Details"] if "Error_Details" in df.columns else None

    print(f"\nTotal modeled rows: {len(df):,}")
    print(f"Date range: {df['created_datetime'].min()} to {df['created_datetime'].max()}")
    print(f"Distinct source domains: {df['source_domain'].nunique(dropna=True):,}")
    print(f"Distinct event categories: {df['event_category'].nunique(dropna=True):,}")

    return df


def load_all_data(config: AnalysisConfig, merge_lookup: bool = True) -> pd.DataFrame:
    """Primary entrypoint for loading GDELT analysis data.
    
    Args:
        config (AnalysisConfig): Analysis configuration object containing paths and runtime options.
        merge_lookup (bool): Whether to merge URL lookup metadata into loaded event rows. Defaults to True.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    return load_gdelt_events(config, merge_lookup=merge_lookup)


def get_time_periods(df: pd.DataFrame, granularity: str = "month") -> List[str]:
    """Get sorted list of time periods in data.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
        granularity (str): Value for `granularity`. Defaults to 'month'.
    
    Returns:
        List[str]: List result produced by this function.
    """
    if granularity == "month":
        return sorted(df["year_month"].dropna().unique())
    if granularity == "quarter":
        tmp = df.copy()
        tmp["quarter"] = tmp["created_datetime"].dt.to_period("Q").astype(str)
        return sorted(tmp["quarter"].dropna().unique())
    if granularity == "year":
        return sorted(df["year"].dropna().astype(int).astype(str).unique())
    return sorted(df["year_month"].dropna().unique())


def filter_by_period(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Filter DataFrame to a specific date range.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
        start_date (str): Inclusive start date bound for filtering.
        end_date (str): Inclusive end date bound for filtering.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (df["created_datetime"] >= start) & (df["created_datetime"] <= end)
    return df[mask].copy()


def sample_from_ids(
    df: pd.DataFrame,
    ids: List[str],
    n: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample n rows from a list of IDs.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
        ids (List[str]): Collection of document IDs.
        n (int): Value for `n`. Defaults to 20.
        random_state (int): Random seed for deterministic sampling. Defaults to 42.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    subset = df[df["id"].isin(ids)]
    if len(subset) <= n:
        return subset
    return subset.sample(n=n, random_state=random_state)
