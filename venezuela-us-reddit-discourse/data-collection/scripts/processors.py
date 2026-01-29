"""
Data processing functions for loading, preprocessing, and exporting data.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def save_json(data: Dict, filepath: Path) -> Path:
    """Save data to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return filepath


def load_json(filepath: Path) -> Dict:
    """Load data from JSON file with error recovery."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"JSON error in {filepath.name}, attempting recovery...")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        data = {}
        pattern = r'"([a-z0-9]+)":\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        for item_id, item_json in re.findall(pattern, content, re.DOTALL):
            try:
                data[item_id] = json.loads(item_json)
            except Exception:
                continue
        return data


def load_all_submissions(data_dir: Path) -> pd.DataFrame:
    """Load all submissions into DataFrame with enhanced preprocessing."""
    records = []
    for f in data_dir.glob("*.json"):
        if "stats" in f.name or "comments" in f.name:
            continue
        try:
            data = load_json(f)
            for item_id, item_data in data.items():
                item_data["_id"] = item_id
                item_data["_source"] = f.name
                records.append(item_data)
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
            continue

    df = pd.DataFrame(records)

    if "created_utc" in df.columns:
        df["created_datetime"] = pd.to_datetime(
            df["created_utc"], unit="s", errors="coerce"
        )
        df["year"] = df["created_datetime"].dt.year
        df["month"] = df["created_datetime"].dt.month
        df["day"] = df["created_datetime"].dt.day
        df["hour"] = df["created_datetime"].dt.hour
        df["dayofweek"] = df["created_datetime"].dt.dayofweek
        df["date"] = df["created_datetime"].dt.date
        df["year_month"] = df["created_datetime"].dt.to_period("M")

    if "title" in df.columns:
        df["title_length"] = df["title"].fillna("").str.len()
        df["title_word_count"] = df["title"].fillna("").str.split().str.len()

    if "score" in df.columns and "num_comments" in df.columns:
        df["engagement_ratio"] = df["num_comments"] / (df["score"] + 1)
        df["total_engagement"] = df["score"] + df["num_comments"]

    if "subreddit" in df.columns:
        df["subreddit"] = df["subreddit"].str.lower()

    return df


def load_all_comments(data_dir: Path) -> pd.DataFrame:
    """Load all comments into DataFrame."""
    records = []
    for f in data_dir.glob("*comments*.json"):
        try:
            data = load_json(f)
            for item_id, item_data in data.items():
                item_data["_id"] = item_id
                item_data["_source"] = f.name
                records.append(item_data)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Handle mixed-type columns that cause parquet export issues
    # The 'edited' column can be False (boolean) or a timestamp (int)
    if "edited" in df.columns:
        # Convert to float: False becomes 0.0, timestamps remain as floats
        df["edited"] = pd.to_numeric(df["edited"], errors="coerce").fillna(0.0)

    # Handle other potentially problematic columns
    mixed_type_cols = ["distinguished", "author_flair_text", "link_flair_text"]
    for col in mixed_type_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("None", "").replace("nan", "")

    if "created_utc" in df.columns:
        df["created_datetime"] = pd.to_datetime(
            df["created_utc"], unit="s", errors="coerce"
        )
        df["year"] = df["created_datetime"].dt.year
        df["date"] = df["created_datetime"].dt.date
        df["hour"] = df["created_datetime"].dt.hour

    if "body" in df.columns:
        df["comment_length"] = df["body"].fillna("").str.len()
        df["word_count"] = df["body"].fillna("").str.split().str.len()

    return df


def export_sub_to_parquet(data_dir: Path, output_file: Path) -> pd.DataFrame:
    """Export all JSON submissions to Parquet."""
    df = load_all_submissions(data_dir)
    if len(df) == 0:
        print("No submissions found to export.")
        return df
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, compression="snappy", index=False)
    print(f"Exported {len(df):,} submissions to {output_file}")
    return df

def export_comments_to_parquet(data_dir: Path, output_file: Path) -> pd.DataFrame:
    """Export all JSON comments to Parquet."""
    df = load_all_comments(data_dir)
    if len(df) == 0:
        print("No comments found to export.")
        return df
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, compression="snappy", index=False)
    print(f"Exported {len(df):,} comments to {output_file}")
    return df

def preprocess_text(text: Any) -> str:
    """Clean text for NLP analysis."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_stopwords() -> set:
    """Comprehensive stopwords for conflict discourse."""
    try:
        from nltk.corpus import stopwords
        base = set(stopwords.words("english"))
    except Exception:
        base = set()

    custom = {
        "said", "says", "would", "could", "also", "one", "two", "new", "first",
        "last", "many", "much", "even", "still", "well", "back", "now", "get",
        "got", "way", "take", "make", "made", "come", "going", "go", "see",
        "like", "know", "think", "want", "say", "look", "time", "year", "years",
        "day", "days", "week", "today", "news", "report", "reddit", "post",
        "comments", "amp", "people", "thing", "something", "really", "actually",
        "just", "right", "need", "let", "us", "im", "dont", "thats", "cant",
        "wont", "didnt", "doesnt", "isnt", "arent", "wasnt", "werent", "youre",
        "hes", "shes", "theyre", "weve", "youve", "theyve", "id", "youd",
        "hed", "wed", "theyd", "ill", "youll", "hell", "shell", "well", "theyll",
    }
    return base | custom


def identify_crisis_posts(
    df: pd.DataFrame, crisis_periods: Dict[str, tuple]
) -> pd.DataFrame:
    """Tag posts that fall within crisis periods."""
    df = df.copy()
    df["crisis_period"] = "Normal"
    for crisis_name, (start, end, color) in crisis_periods.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        mask = (df["created_datetime"] >= start_dt) & (
            df["created_datetime"] <= end_dt)
        df.loc[mask, "crisis_period"] = crisis_name
    return df


def deduplicate_submissions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate submissions, keeping the most complete record."""
    if "_id" in df.columns:
        df = df.drop_duplicates(subset=["_id"], keep="first")
    elif "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
    return df


def normalize_subreddit_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize subreddit names to lowercase."""
    if "subreddit" in df.columns:
        df["subreddit"] = df["subreddit"].str.lower()
    return df


def compute_subreddit_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute subreddit statistics."""
    sub_stats = df.groupby("subreddit").agg({
        "_id": "count",
        "score": ["mean", "median", "sum"],
        "num_comments": ["mean", "sum"],
        "author": "nunique"
    }).round(1)
    sub_stats.columns = [
        "posts", "avg_score", "med_score", "total_score",
        "avg_comments", "total_comments", "unique_authors"
    ]
    sub_stats = sub_stats.sort_values("posts", ascending=False)
    return sub_stats
