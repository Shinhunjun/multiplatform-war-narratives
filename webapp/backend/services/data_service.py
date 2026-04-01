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
TIKTOK_ANALYSIS_DIR = ANALYSIS_DIR.parent / "outputs_tiktok"

SENTIMENT_DIR = ANALYSIS_DIR / "sentiment"
TOPICS_DIR = ANALYSIS_DIR / "topics"
CLUSTERS_DIR = ANALYSIS_DIR / "clusters"
VISUALIZATIONS_DIR = ANALYSIS_DIR / "visualizations"

NEWS_SENTIMENT_DIR = NEWS_ANALYSIS_DIR / "sentiment"
NEWS_TOPICS_DIR = NEWS_ANALYSIS_DIR / "topics"
NEWS_CLUSTERS_DIR = NEWS_ANALYSIS_DIR / "clusters"

TIKTOK_SENTIMENT_DIR = TIKTOK_ANALYSIS_DIR / "sentiment"
TIKTOK_TOPICS_DIR = TIKTOK_ANALYSIS_DIR / "topics"
TIKTOK_CLUSTERS_DIR = TIKTOK_ANALYSIS_DIR / "clusters"
TIKTOK_SPECIFIC_DIR = TIKTOK_ANALYSIS_DIR / "tiktok_specific"


def download_from_gcs() -> None:
    """Download analysis data from GCS bucket to local DATA_DIR.
    Only runs if GCS_BUCKET env var is set and data hasn't been downloaded yet.
    """
    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        return

    # Skip if cross-platform scatter already downloaded (latest addition)
    cross_scatter = CLUSTERS_DIR / "cross_platform_scatter.parquet"
    has_all = cross_scatter.exists() and SENTIMENT_DIR.exists() and any(SENTIMENT_DIR.iterdir())
    if has_all:
        logger.info("Data already present, skipping GCS download")
        return

    from google.cloud import storage

    # Skip large files not needed by the API server
    _SKIP_PATTERNS = {
        "embeddings.npy", "embeddings_2d.npy", "document_embeddings.npy",
        "sentiment_full.parquet", "bertopic_model",
    }

    def _should_skip(name: str) -> bool:
        basename = name.rsplit("/", 1)[-1] if "/" in name else name
        return any(pat in basename or pat in name for pat in _SKIP_PATTERNS)

    try:
        logger.info(f"Downloading data from gs://{bucket_name}/ to {ANALYSIS_DIR}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        blobs = list(bucket.list_blobs())
        downloaded = 0
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            if _should_skip(blob.name):
                logger.info(f"  Skipped {blob.name} (not needed by API)")
                continue
            # Route blobs to appropriate directories
            if blob.name.startswith("outputs_news/"):
                rel = blob.name[len("outputs_news/"):]
                local_path = NEWS_ANALYSIS_DIR / rel
            elif blob.name.startswith("outputs_tiktok/"):
                rel = blob.name[len("outputs_tiktok/"):]
                local_path = TIKTOK_ANALYSIS_DIR / rel
            else:
                local_path = ANALYSIS_DIR / blob.name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            logger.info(f"  Downloaded {blob.name} ({blob.size:,} bytes)")
            downloaded += 1

        logger.info(f"Download complete: {downloaded}/{len(blobs)} objects")
    except Exception as e:
        logger.error(f"Failed to download from GCS: {e}")


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


def get_entity_network(platform: str = "reddit") -> dict:
    path = ANALYSIS_DIR / "entities" / f"network_{platform}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def get_entity_relationships(platform: str = "reddit") -> list:
    path = ANALYSIS_DIR / "entities" / f"top_relationships_{platform}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


# --- Dynamic entity network building from parquet data ---

@functools.lru_cache(maxsize=3)
def _load_entities_parquet(platform: str) -> Optional[pd.DataFrame]:
    path = ANALYSIS_DIR / "entities" / f"entities_{platform}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@functools.lru_cache(maxsize=3)
def _load_relationships_parquet(platform: str) -> Optional[pd.DataFrame]:
    path = ANALYSIS_DIR / "entities" / f"relationships_{platform}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def get_entity_months(platform: str = "reddit") -> list[str]:
    """Return all available months for entity data."""
    df = _load_entities_parquet(platform)
    if df is None:
        return []
    return sorted(df["year_month"].unique().tolist())


def build_entity_network(platform: str, start: str | None = None, end: str | None = None) -> dict:
    """Build entity co-occurrence network dynamically from parquet, filtered by period."""
    import networkx as nx

    ent_df = _load_entities_parquet(platform)
    rel_df = _load_relationships_parquet(platform)
    if ent_df is None:
        return {}

    # Filter by period
    if start:
        ent_df = ent_df[ent_df["year_month"] >= start]
        if rel_df is not None:
            rel_df = rel_df[rel_df["year_month"] >= start]
    if end:
        ent_df = ent_df[ent_df["year_month"] <= end]
        if rel_df is not None:
            rel_df = rel_df[rel_df["year_month"] <= end]

    if ent_df.empty:
        return {"nodes": [], "edges": [], "communities": [], "platform": platform}

    # Aggregate entities across months
    agg = ent_df.groupby(["name", "type"]).agg({"count": "sum"}).reset_index()
    agg = agg.sort_values("count", ascending=False)

    # Top 150 entities for graph
    top_entities = agg.head(150)
    entity_set = set(top_entities["name"])
    entity_type = dict(zip(top_entities["name"], top_entities["type"]))
    entity_freq = dict(zip(top_entities["name"], top_entities["count"]))

    # Build co-occurrence from relationships
    G = nx.Graph()
    for _, row in top_entities.iterrows():
        G.add_node(row["name"], type=row["type"], frequency=int(row["count"]))

    if rel_df is not None and not rel_df.empty:
        rel_agg = rel_df.groupby(["source", "target"]).agg({"count": "sum"}).reset_index()
        for _, row in rel_agg.iterrows():
            if row["source"] in entity_set and row["target"] in entity_set:
                if G.has_edge(row["source"], row["target"]):
                    G[row["source"]][row["target"]]["weight"] += int(row["count"])
                else:
                    G.add_edge(row["source"], row["target"], weight=int(row["count"]))

    # Also build co-occurrence from same-month entity pairs
    months = ent_df[ent_df["name"].isin(entity_set)].groupby("year_month")["name"].apply(list)
    for names in months:
        for i in range(len(names)):
            for j in range(i + 1, min(len(names), i + 30)):
                a, b = names[i], names[j]
                if a != b and a in entity_set and b in entity_set:
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += 1
                    else:
                        G.add_edge(a, b, weight=1)

    # Community detection
    if len(G.nodes) > 1:
        try:
            communities_map = nx.community.louvain_communities(G, seed=42)
        except Exception:
            communities_map = [{n} for n in G.nodes]
    else:
        communities_map = [{n} for n in G.nodes]

    # Assign community IDs
    node_community = {}
    for cid, members in enumerate(communities_map):
        for m in members:
            node_community[m] = cid

    # Build output
    nodes = []
    for n in G.nodes:
        nodes.append({
            "id": n,
            "community": node_community.get(n, 0),
            "frequency": G.nodes[n].get("frequency", entity_freq.get(n, 1)),
            "type": G.nodes[n].get("type", entity_type.get(n, "UNKNOWN")),
        })

    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({"source": u, "target": v, "weight": d.get("weight", 1)})

    communities_list = []
    for cid, members in enumerate(communities_map):
        sorted_members = sorted(members, key=lambda m: entity_freq.get(m, 0), reverse=True)
        total_freq = sum(entity_freq.get(m, 0) for m in members)
        communities_list.append({
            "id": cid,
            "size": len(members),
            "total_frequency": total_freq,
            "top_members": sorted_members[:10],
            "label": ", ".join(sorted_members[:3]),
        })
    communities_list.sort(key=lambda c: c["total_frequency"], reverse=True)

    return {
        "nodes": nodes,
        "edges": edges,
        "communities": communities_list,
        "platform": platform,
    }


def get_entity_relationships_filtered(platform: str, start: str | None = None, end: str | None = None) -> list:
    """Get top relationships filtered by period."""
    rel_df = _load_relationships_parquet(platform)
    if rel_df is None:
        return []

    if start:
        rel_df = rel_df[rel_df["year_month"] >= start]
    if end:
        rel_df = rel_df[rel_df["year_month"] <= end]

    if rel_df.empty:
        return []

    agg = rel_df.groupby(["source", "target", "relation"]).agg({"count": "sum"}).reset_index()
    agg = agg.sort_values("count", ascending=False).head(50)
    return agg.to_dict("records")


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


@functools.lru_cache(maxsize=1)
def get_topics_monthly() -> pd.DataFrame:
    """Load pre-computed monthly topic counts (global model aggregation)."""
    path = TOPICS_DIR / "topics_monthly.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "topic_id", "count", "proportion", "name", "keywords"])


@functools.lru_cache(maxsize=1)
def get_topics_monthly_fitted() -> pd.DataFrame:
    """Load independently-fitted monthly topics (BERTopic per month)."""
    path = TOPICS_DIR / "monthly_topics_fitted.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "topic_id", "keywords", "count", "proportion"])


@functools.lru_cache(maxsize=1)
def get_news_topics_monthly() -> pd.DataFrame:
    """Load pre-computed monthly topic counts for news (global model aggregation)."""
    path = NEWS_TOPICS_DIR / "topics_monthly.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "topic_id", "count", "proportion", "name", "keywords"])


@functools.lru_cache(maxsize=1)
def get_news_topics_monthly_fitted() -> pd.DataFrame:
    """Load independently-fitted monthly topics for news."""
    path = NEWS_TOPICS_DIR / "monthly_topics_fitted.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "topic_id", "keywords", "count", "proportion"])


@functools.lru_cache(maxsize=1)
def get_cross_platform_scatter() -> pd.DataFrame:
    """Load unified cross-platform UMAP scatter (Reddit + News in same space)."""
    path = CLUSTERS_DIR / "cross_platform_scatter.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@functools.lru_cache(maxsize=1)
def get_clusters_monthly() -> pd.DataFrame:
    """Load pre-computed monthly cluster counts."""
    path = CLUSTERS_DIR / "clusters_monthly.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "cluster_id", "count", "proportion", "keywords"])


# --- News clusters ---
@functools.lru_cache(maxsize=1)
def get_news_cluster_summaries() -> pd.DataFrame:
    path = NEWS_CLUSTERS_DIR / "cluster_summaries.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@functools.lru_cache(maxsize=1)
def get_news_cluster_keywords() -> pd.DataFrame:
    path = NEWS_CLUSTERS_DIR / "cluster_keywords.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@functools.lru_cache(maxsize=1)
def get_news_temporal_clusters() -> pd.DataFrame:
    path = NEWS_CLUSTERS_DIR / "temporal_clusters.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@functools.lru_cache(maxsize=1)
def get_news_clusters_monthly() -> pd.DataFrame:
    path = NEWS_CLUSTERS_DIR / "monthly_clusters_fitted.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "cluster_id", "count", "proportion", "keywords"])


# --- TikTok clusters ---
@functools.lru_cache(maxsize=1)
def get_tiktok_cluster_summaries() -> pd.DataFrame:
    path = TIKTOK_CLUSTERS_DIR / "cluster_summaries.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@functools.lru_cache(maxsize=1)
def get_tiktok_cluster_keywords() -> pd.DataFrame:
    path = TIKTOK_CLUSTERS_DIR / "cluster_keywords.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@functools.lru_cache(maxsize=1)
def get_tiktok_temporal_clusters() -> pd.DataFrame:
    path = TIKTOK_CLUSTERS_DIR / "temporal_clusters.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@functools.lru_cache(maxsize=1)
def get_tiktok_clusters_monthly() -> pd.DataFrame:
    path = TIKTOK_CLUSTERS_DIR / "monthly_clusters_fitted.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "cluster_id", "count", "proportion", "keywords"])


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


# ---------------------------------------------------------------------------
# TikTok data loaders
# ---------------------------------------------------------------------------

def _tiktok_data_available() -> bool:
    return TIKTOK_SENTIMENT_DIR.exists() and (TIKTOK_SENTIMENT_DIR / "sentiment_by_month.csv").exists()


@functools.lru_cache(maxsize=1)
def get_tiktok_sentiment_by_month() -> pd.DataFrame:
    return pd.read_csv(TIKTOK_SENTIMENT_DIR / "sentiment_by_month.csv")


@functools.lru_cache(maxsize=1)
def get_tiktok_sentiment_by_source() -> pd.DataFrame:
    df = pd.read_csv(TIKTOK_SENTIMENT_DIR / "sentiment_by_source.csv")
    df = df.fillna(0)
    df["source"] = df["source"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def get_tiktok_sentiment_by_source_month() -> pd.DataFrame:
    df = pd.read_csv(TIKTOK_SENTIMENT_DIR / "sentiment_by_source_month.csv")
    df = df.fillna(0)
    df["source"] = df["source"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def get_tiktok_topic_info() -> pd.DataFrame:
    path = TIKTOK_TOPICS_DIR / "topic_info.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df[df["Topic"] >= 0].reset_index(drop=True)
    return pd.DataFrame(columns=["Topic", "Count", "Name", "Representation"])


@functools.lru_cache(maxsize=1)
def get_tiktok_topics_over_time() -> pd.DataFrame:
    path = TIKTOK_TOPICS_DIR / "topics_over_time.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df[df["Topic"] >= 0].reset_index(drop=True)
    return pd.DataFrame(columns=["Topic", "Timestamp", "Frequency"])


@functools.lru_cache(maxsize=1)
def get_tiktok_topics_monthly_fitted() -> pd.DataFrame:
    path = TIKTOK_TOPICS_DIR / "monthly_topics_fitted.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "topic_id", "keywords", "count", "proportion"])


@functools.lru_cache(maxsize=1)
def get_tiktok_hashtag_trends() -> pd.DataFrame:
    path = TIKTOK_SPECIFIC_DIR / "hashtag_trends.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "hashtag", "count", "mean_sentiment"])


@functools.lru_cache(maxsize=1)
def get_tiktok_engagement_metrics() -> pd.DataFrame:
    path = TIKTOK_SPECIFIC_DIR / "engagement_metrics.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "video_count", "total_views", "total_likes"])


@functools.lru_cache(maxsize=1)
def get_tiktok_region_distribution() -> pd.DataFrame:
    path = TIKTOK_SPECIFIC_DIR / "region_distribution.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["year_month", "region_code", "count"])


def get_tiktok_overview_stats() -> Optional[dict]:
    """Get summary statistics for TikTok data."""
    if not _tiktok_data_available():
        return None

    # Try loading overview.json first
    overview_path = TIKTOK_ANALYSIS_DIR / "overview.json"
    if overview_path.exists():
        with open(overview_path) as f:
            return json.load(f)

    sentiment_month = get_tiktok_sentiment_by_month()
    sentiment_src = get_tiktok_sentiment_by_source()
    topics = get_tiktok_topic_info()

    total_documents = int(sentiment_src["total_count"].sum())
    date_range_start = sentiment_month["year_month"].min()
    date_range_end = sentiment_month["year_month"].max()

    return {
        "platform": "tiktok",
        "total_documents": total_documents,
        "sources": len(sentiment_src),
        "date_range": {"start": date_range_start, "end": date_range_end},
        "num_topics": len(topics),
        "num_clusters": 0,
        "avg_sentiment": round(float(sentiment_src["mean_sentiment"].mean()), 4),
        "source_list": sorted(sentiment_src["source"].dropna().astype(str).head(50).tolist()),
    }
