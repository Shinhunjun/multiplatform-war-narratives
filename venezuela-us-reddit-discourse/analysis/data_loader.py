"""
Data loader with ID tracking for analysis pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .config import AnalysisConfig


def load_json_file(filepath: Path) -> Dict:
    """Load a single JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def load_submissions(config: AnalysisConfig) -> pd.DataFrame:
    """
    Load all submissions with ID tracking.

    Returns DataFrame with columns:
    - id: unique post ID
    - type: 'submission'
    - subreddit, title, selftext, author, score, num_comments
    - created_utc, created_datetime, year_month
    - text: combined title + selftext for analysis
    """
    records = []
    files = list(config.submissions_dir.glob("*_filtered.json"))

    print(f"Loading {len(files)} submission files...")

    for filepath in tqdm(files, desc="Loading submissions"):
        if "comments" in filepath.name or "stats" in filepath.name:
            continue

        data = load_json_file(filepath)
        for post_id, post_data in data.items():
            if post_id.startswith("_"):
                continue

            # Combine title and selftext
            title = post_data.get("title", "") or ""
            selftext = post_data.get("selftext", "") or ""
            text = f"{title} {selftext}".strip()

            if not text or text in ["[deleted]", "[removed]"]:
                continue

            records.append({
                "id": post_id,
                "type": "submission",
                "subreddit": str(post_data.get("subreddit", "")).lower(),
                "title": title,
                "selftext": selftext,
                "text": text,
                "author": post_data.get("author"),
                "score": post_data.get("score", 0),
                "num_comments": post_data.get("num_comments", 0),
                "created_utc": post_data.get("created_utc"),
                "url": post_data.get("url"),
            })

    df = pd.DataFrame(records)

    if len(df) > 0 and "created_utc" in df.columns:
        df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        df["year"] = df["created_datetime"].dt.year
        df["month"] = df["created_datetime"].dt.month
        df["year_month"] = df["created_datetime"].dt.to_period("M").astype(str)
        df["date"] = df["created_datetime"].dt.date

    print(f"Loaded {len(df):,} submissions")
    return df


def load_comments(config: AnalysisConfig) -> pd.DataFrame:
    """
    Load all comments with ID tracking.

    Returns DataFrame with columns:
    - id: unique comment ID
    - type: 'comment'
    - submission_id: parent submission ID
    - subreddit, body, author, score
    - created_utc, created_datetime, year_month
    - text: body text for analysis
    """
    records = []
    files = list(config.comments_dir.glob("comments_*.json"))

    print(f"Loading {len(files)} comment files...")

    for filepath in tqdm(files, desc="Loading comments"):
        data = load_json_file(filepath)
        for comment_id, comment_data in data.items():
            if comment_id.startswith("_"):
                continue

            body = comment_data.get("body", "") or ""
            if not body or body in ["[deleted]", "[removed]"]:
                continue

            # Extract submission ID from link_id
            link_id = comment_data.get("link_id", "") or comment_data.get("_submission_id", "")
            if link_id.startswith("t3_"):
                submission_id = link_id[3:]
            else:
                submission_id = link_id

            records.append({
                "id": comment_id,
                "type": "comment",
                "submission_id": submission_id,
                "subreddit": str(comment_data.get("subreddit", "")).lower(),
                "body": body,
                "text": body,
                "author": comment_data.get("author"),
                "score": comment_data.get("score", 0),
                "created_utc": comment_data.get("created_utc"),
                "parent_id": comment_data.get("parent_id"),
                "is_top_level": comment_data.get("_is_top_level", False),
            })

    df = pd.DataFrame(records)

    if len(df) > 0 and "created_utc" in df.columns:
        df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        df["year"] = df["created_datetime"].dt.year
        df["month"] = df["created_datetime"].dt.month
        df["year_month"] = df["created_datetime"].dt.to_period("M").astype(str)
        df["date"] = df["created_datetime"].dt.date

    print(f"Loaded {len(df):,} comments")
    return df


def load_preprocessed_data(config: AnalysisConfig) -> pd.DataFrame:
    """
    Load preprocessed data from parquet files (recommended).

    This uses the cleaned data from the preprocessing pipeline.
    """
    preprocessed_dir = config.data_dir / "preprocessed"
    sub_path = preprocessed_dir / "submissions_clean.parquet"
    com_path = preprocessed_dir / "comments_clean.parquet"

    if not sub_path.exists() or not com_path.exists():
        print("Preprocessed data not found, falling back to raw data...")
        return load_all_data(config)

    print("Loading preprocessed data...")

    # Load submissions
    submissions_df = pd.read_parquet(sub_path)
    submissions_df["type"] = "submission"
    submissions_df["text"] = submissions_df["full_text"]
    print(f"Loaded {len(submissions_df):,} preprocessed submissions")

    # Load comments
    comments_df = pd.read_parquet(com_path)
    comments_df["type"] = "comment"
    comments_df["text"] = comments_df["body_clean"]
    print(f"Loaded {len(comments_df):,} preprocessed comments")

    # Add time columns if missing
    for df in [submissions_df, comments_df]:
        if "created_datetime" not in df.columns and "created_utc" in df.columns:
            df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        if "year_month" not in df.columns:
            df["year_month"] = df["created_datetime"].dt.to_period("M").astype(str)
        if "year" not in df.columns:
            df["year"] = df["created_datetime"].dt.year
        if "month" not in df.columns:
            df["month"] = df["created_datetime"].dt.month

    # Combine
    combined = pd.concat([submissions_df, comments_df], ignore_index=True)
    combined = combined.sort_values("created_utc").reset_index(drop=True)

    print(f"\nTotal records: {len(combined):,}")
    print(f"  Submissions: {len(submissions_df):,}")
    print(f"  Comments: {len(comments_df):,}")
    print(f"  Date range: {combined['created_datetime'].min()} to {combined['created_datetime'].max()}")
    print(f"  Subreddits: {combined['subreddit'].nunique()}")

    return combined


def load_all_data(config: AnalysisConfig, use_preprocessed: bool = True) -> pd.DataFrame:
    """
    Load and combine all submissions and comments.

    Args:
        config: Analysis configuration
        use_preprocessed: If True, try to load preprocessed data first

    Returns unified DataFrame with consistent schema for analysis.
    """
    # Try preprocessed data first
    if use_preprocessed:
        preprocessed_dir = config.data_dir / "preprocessed"
        if (preprocessed_dir / "submissions_clean.parquet").exists():
            return load_preprocessed_data(config)

    submissions_df = load_submissions(config)
    comments_df = load_comments(config)

    # Rename body to text for comments (already done above)
    # Combine
    combined = pd.concat([submissions_df, comments_df], ignore_index=True)

    # Sort by time
    combined = combined.sort_values("created_utc").reset_index(drop=True)

    print(f"\nTotal records: {len(combined):,}")
    print(f"  Submissions: {len(submissions_df):,}")
    print(f"  Comments: {len(comments_df):,}")
    print(f"  Date range: {combined['created_datetime'].min()} to {combined['created_datetime'].max()}")
    print(f"  Subreddits: {combined['subreddit'].nunique()}")

    return combined


def get_time_periods(
    df: pd.DataFrame,
    granularity: str = "month"
) -> List[str]:
    """Get sorted list of time periods in data."""
    if granularity == "month":
        return sorted(df["year_month"].dropna().unique())
    elif granularity == "quarter":
        df["quarter"] = df["created_datetime"].dt.to_period("Q").astype(str)
        return sorted(df["quarter"].dropna().unique())
    elif granularity == "year":
        return sorted(df["year"].dropna().unique().astype(str))
    else:
        return sorted(df["year_month"].dropna().unique())


def filter_by_period(
    df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Filter DataFrame to a specific time period."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (df["created_datetime"] >= start) & (df["created_datetime"] <= end)
    return df[mask].copy()


def sample_from_ids(
    df: pd.DataFrame,
    ids: List[str],
    n: int = 20,
    random_state: int = 42
) -> pd.DataFrame:
    """Sample n records from a list of IDs."""
    subset = df[df["id"].isin(ids)]
    if len(subset) <= n:
        return subset
    return subset.sample(n=n, random_state=random_state)
