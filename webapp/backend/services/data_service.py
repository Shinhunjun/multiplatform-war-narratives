"""
Data service: loads and caches all Reddit and News analysis outputs.
Supports local files and GCS download for Cloud Run deployment.
"""

import functools
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Base paths — use DATA_DIR env var if set, otherwise default to local relative path
_data_dir_env = os.environ.get("DATA_DIR")
if _data_dir_env:
    ANALYSIS_DIR = Path(_data_dir_env)
else:
    ANALYSIS_DIR = Path(__file__).parent.parent.parent.parent / "reddit" / "analysis" / "outputs"

NEWS_ANALYSIS_DIR = ANALYSIS_DIR.parent / "outputs_news"

SENTIMENT_DIR = ANALYSIS_DIR / "sentiment"
TOPICS_DIR = ANALYSIS_DIR / "topics"
CLUSTERS_DIR = ANALYSIS_DIR / "clusters"
VISUALIZATIONS_DIR = ANALYSIS_DIR / "visualizations"

NEWS_SENTIMENT_DIR = NEWS_ANALYSIS_DIR / "sentiment"
NEWS_TOPICS_DIR = NEWS_ANALYSIS_DIR / "topics"


def download_from_gcs() -> None:
    """Download analysis data from GCS bucket to local DATA_DIR.
    Only runs if GCS_BUCKET env var is set and data hasn't been downloaded yet.
    """
    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        return

    # Skip if data already downloaded
    if SENTIMENT_DIR.exists() and any(SENTIMENT_DIR.iterdir()):
        logger.info("Data already present, skipping GCS download")
        return

    from google.cloud import storage

    logger.info(f"Downloading data from gs://{bucket_name}/ to {ANALYSIS_DIR}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs())
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        # Route outputs_news/ blobs to NEWS_ANALYSIS_DIR, rest to ANALYSIS_DIR
        if blob.name.startswith("outputs_news/"):
            rel = blob.name[len("outputs_news/"):]
            local_path = NEWS_ANALYSIS_DIR / rel
        else:
            local_path = ANALYSIS_DIR / blob.name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        logger.info(f"  Downloaded {blob.name} ({blob.size:,} bytes)")

    logger.info(f"Download complete: {len(blobs)} objects")


# ---------------------------------------------------------------------------
# Reddit data loaders
# ---------------------------------------------------------------------------

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
    path = VISUALIZATIONS_DIR / "cluster_summary_table.csv"
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


# ---------------------------------------------------------------------------
# News (GDELT) data loaders
# ---------------------------------------------------------------------------

def _news_data_available() -> bool:
    return NEWS_SENTIMENT_DIR.exists() and (NEWS_SENTIMENT_DIR / "sentiment_by_month.csv").exists()


@functools.lru_cache(maxsize=1)
def get_news_sentiment_by_month() -> pd.DataFrame:
    return pd.read_csv(NEWS_SENTIMENT_DIR / "sentiment_by_month.csv")


@functools.lru_cache(maxsize=1)
def get_news_sentiment_by_source() -> pd.DataFrame:
    df = pd.read_csv(NEWS_SENTIMENT_DIR / "sentiment_by_source.csv")
    df = df.fillna(0)
    df["source"] = df["source"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def get_news_sentiment_by_source_month() -> pd.DataFrame:
    df = pd.read_csv(NEWS_SENTIMENT_DIR / "sentiment_by_source_month.csv")
    df = df.fillna(0)
    df["source"] = df["source"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def get_news_topic_info() -> pd.DataFrame:
    path = NEWS_TOPICS_DIR / "topic_info.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df[df["Topic"] >= 1].reset_index(drop=True)
    return pd.DataFrame(columns=["Topic", "Count", "Name", "Representation"])


@functools.lru_cache(maxsize=1)
def get_news_topics_over_time() -> pd.DataFrame:
    path = NEWS_TOPICS_DIR / "topics_over_time.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df[df["Topic"] >= 1].reset_index(drop=True)
    return pd.DataFrame(columns=["Topic", "Timestamp", "Frequency"])


@functools.lru_cache(maxsize=1)
def get_news_topics_by_source() -> pd.DataFrame:
    path = NEWS_TOPICS_DIR / "topics_by_source.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df[df["topic_id"] >= 1].reset_index(drop=True)
    return pd.DataFrame(columns=["source", "topic_id", "count"])


def get_news_overview_stats() -> Optional[dict]:
    """Get summary statistics for news data."""
    if not _news_data_available():
        return None

    sentiment_month = get_news_sentiment_by_month()
    sentiment_src = get_news_sentiment_by_source()
    topics = get_news_topic_info()

    total_documents = int(sentiment_src["total_count"].sum())
    date_range_start = sentiment_month["year_month"].min()
    date_range_end = sentiment_month["year_month"].max()

    return {
        "platform": "news",
        "total_documents": total_documents,
        "sources": len(sentiment_src),
        "date_range": {"start": date_range_start, "end": date_range_end},
        "num_topics": len(topics),
        "num_clusters": 0,
        "avg_sentiment": round(float(sentiment_src["mean_sentiment"].mean()), 4),
        "source_list": sorted(sentiment_src["source"].dropna().astype(str).tolist()),
    }
