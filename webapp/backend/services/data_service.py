"""
Data service: loads and caches all Reddit analysis outputs.
"""

import functools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Base paths
ANALYSIS_DIR = Path(__file__).parent.parent.parent.parent / "venezuela-us-reddit-discourse" / "analysis" / "outputs"
SENTIMENT_DIR = ANALYSIS_DIR / "sentiment"
TOPICS_DIR = ANALYSIS_DIR / "topics"
CLUSTERS_DIR = ANALYSIS_DIR / "clusters"


@functools.lru_cache(maxsize=1)
def get_sentiment_by_month() -> pd.DataFrame:
    return pd.read_csv(SENTIMENT_DIR / "sentiment_by_month.csv")


@functools.lru_cache(maxsize=1)
def get_sentiment_by_subreddit() -> pd.DataFrame:
    return pd.read_csv(SENTIMENT_DIR / "sentiment_by_subreddit.csv")


@functools.lru_cache(maxsize=1)
def get_sentiment_by_subreddit_month() -> pd.DataFrame:
    return pd.read_csv(SENTIMENT_DIR / "sentiment_by_subreddit_month.csv")


@functools.lru_cache(maxsize=1)
def get_topic_info() -> pd.DataFrame:
    df = pd.read_csv(TOPICS_DIR / "topic_info.csv")
    # Drop outlier topic -1 for display
    return df[df["Topic"] >= 0].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def get_topics_by_subreddit() -> pd.DataFrame:
    df = pd.read_csv(TOPICS_DIR / "topics_by_subreddit.csv")
    return df[df["topic_id"] >= 0].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def get_topics_over_time() -> pd.DataFrame:
    df = pd.read_csv(TOPICS_DIR / "topics_over_time.csv")
    return df[df["Topic"] >= 0].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def get_cluster_summaries() -> pd.DataFrame:
    return pd.read_csv(CLUSTERS_DIR / "cluster_summaries.csv")


@functools.lru_cache(maxsize=1)
def get_cluster_keywords() -> pd.DataFrame:
    return pd.read_csv(CLUSTERS_DIR / "cluster_keywords.csv")


@functools.lru_cache(maxsize=1)
def get_temporal_clusters() -> pd.DataFrame:
    return pd.read_csv(CLUSTERS_DIR / "temporal_clusters.csv")


@functools.lru_cache(maxsize=1)
def get_cluster_summary_table() -> Optional[pd.DataFrame]:
    path = CLUSTERS_DIR / ".." / "visualizations" / "cluster_summary_table.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@functools.lru_cache(maxsize=1)
def get_embeddings_2d() -> Optional[np.ndarray]:
    """Load 2D UMAP embeddings for scatter plot (360K x 2)."""
    path = CLUSTERS_DIR / "embeddings_2d.npy"
    if path.exists():
        return np.load(path)
    return None


@functools.lru_cache(maxsize=1)
def get_cluster_assignments() -> Optional[pd.DataFrame]:
    """Load cluster assignments parquet."""
    path = CLUSTERS_DIR / "cluster_assignments.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def get_overview_stats() -> dict:
    """Get summary statistics across all analyses."""
    sentiment_month = get_sentiment_by_month()
    sentiment_sub = get_sentiment_by_subreddit()
    topics = get_topic_info()
    clusters = get_cluster_summaries()

    total_documents = int(sentiment_sub["total_count"].sum())
    date_range_start = sentiment_month["year_month"].min()
    date_range_end = sentiment_month["year_month"].max()

    return {
        "platform": "reddit",
        "total_documents": total_documents,
        "subreddits": len(sentiment_sub),
        "date_range": {"start": date_range_start, "end": date_range_end},
        "num_topics": len(topics),
        "num_clusters": len(clusters),
        "avg_sentiment": round(float(sentiment_sub["mean_sentiment"].mean()), 4),
        "subreddit_list": sorted(sentiment_sub["subreddit"].tolist()),
    }
