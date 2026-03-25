"""Clustering endpoints."""

import functools
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from ..services.data_service import (
    get_cluster_assignments,
    get_cluster_keywords,
    get_cluster_summaries,
    get_clusters_monthly,
    get_cross_platform_scatter,
    get_temporal_clusters,
    get_news_cluster_summaries,
    get_news_cluster_keywords,
    get_news_temporal_clusters,
    get_news_clusters_monthly,
    get_tiktok_cluster_summaries,
    get_tiktok_cluster_keywords,
    get_tiktok_temporal_clusters,
    get_tiktok_clusters_monthly,
)

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


# ---------------------------------------------------------------------------
# Shared helpers — cached by (start, end) so repeated queries are free
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _filtered_counts(start: Optional[str], end: Optional[str]) -> Optional[dict]:
    """Cluster document counts filtered by time range. Cached per (start, end)."""
    assignments = get_cluster_assignments()
    if assignments is None:
        return None
    adf = assignments[assignments["cluster_id"] != -1]
    if start:
        adf = adf[adf["year_month"] >= start]
    if end:
        adf = adf[adf["year_month"] <= end]
    return adf["cluster_id"].value_counts().to_dict()


@functools.lru_cache(maxsize=64)
def _filtered_stats(start: Optional[str], end: Optional[str]) -> Optional[pd.DataFrame]:
    """Cluster stats (count, top_subreddit) filtered by time range."""
    assignments = get_cluster_assignments()
    if assignments is None:
        return None
    adf = assignments[assignments["cluster_id"] != -1]
    if start:
        adf = adf[adf["year_month"] >= start]
    if end:
        adf = adf[adf["year_month"] <= end]
    if adf.empty:
        return None

    counts = adf.groupby("cluster_id").agg(
        count=("cluster_id", "size"),
        top_subreddit=("subreddit", lambda s: s.value_counts().index[0] if len(s) > 0 else ""),
    ).reset_index()
    return counts


@functools.lru_cache(maxsize=64)
def _scatter_data(
    top_n: int, max_points: int,
    start: Optional[str], end: Optional[str],
) -> list[dict]:
    """Scatter plot data, cached per parameter combo."""
    assignments = get_cluster_assignments()
    if assignments is None:
        return []

    df = assignments[assignments["cluster_id"] != -1]

    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]

    if df.empty:
        return []

    # Keep only top N clusters by size
    top_clusters = df["cluster_id"].value_counts().nlargest(top_n).index.tolist()
    df = df[df["cluster_id"].isin(top_clusters)]

    # Stratified sampling to cap total points
    if len(df) > max_points:
        total = len(df)
        sampled = []
        for cid, g in df.groupby("cluster_id"):
            n = min(len(g), max(1, int(max_points * len(g) / total)))
            sampled.append(g.sample(n=n, random_state=42))
        df = pd.concat(sampled, ignore_index=True)

    # Join keywords (top 3)
    kw = get_cluster_keywords()
    kw_map = {
        row["cluster_id"]: ", ".join(row["keywords"].split(", ")[:3])
        for _, row in kw.iterrows()
        if isinstance(row["keywords"], str)
    }
    df = df.copy()
    df["keywords"] = df["cluster_id"].map(kw_map).fillna("")

    result = df[["umap_1", "umap_2", "cluster_id", "subreddit", "keywords"]].rename(
        columns={"umap_1": "x", "umap_2": "y"}
    )
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.where(result.notna(), None).to_dict(orient="records")


@functools.lru_cache(maxsize=64)
def _keywords_map() -> dict:
    """cluster_id → top-3 keywords string. Cached."""
    kw = get_cluster_keywords()
    return {
        cid: ", ".join(kws.split(", ")[:3])
        for cid, kws in zip(kw["cluster_id"], kw["keywords"])
        if isinstance(kws, str)
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/cross-platform-scatter")
def cross_platform_scatter(
    max_points: int = Query(30000),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Unified UMAP scatter: Reddit + News in the same embedding space."""
    df = get_cross_platform_scatter()
    if df.empty:
        return []
    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
    return df.to_dict(orient="records")


@router.get("/summaries")
def cluster_summaries(
    limit: int = Query(50, description="Max clusters to return"),
    min_count: int = Query(10, description="Minimum document count"),
    start: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end: Optional[str] = Query(None, description="End month YYYY-MM"),
    platform: Optional[str] = Query(None),
):
    """Get cluster summaries sorted by document count, with keywords."""
    if platform == "news":
        df = get_news_cluster_summaries()
    elif platform == "tiktok":
        df = get_tiktok_cluster_summaries()
    else:
        df = get_cluster_summaries()
    if df.empty:
        return []

    # If time range specified, recompute counts from assignments
    if start or end:
        if platform == "news":
            adf_raw = _load_platform_assignments("news")
        elif platform == "tiktok":
            adf_raw = _load_platform_assignments("tiktok")
        else:
            adf_raw = get_cluster_assignments()

        if adf_raw is not None and not adf_raw.empty:
            adf = adf_raw[adf_raw["cluster_id"] != -1]
            if start:
                adf = adf[adf["year_month"] >= start]
            if end:
                adf = adf[adf["year_month"] <= end]
            if not adf.empty:
                counts = adf["cluster_id"].value_counts().to_dict()
                df = df.copy()
                df["count"] = df["cluster_id"].map(counts).fillna(0).astype(int)
                # Update top source if available
                src_col = "subreddit" if "subreddit" in adf.columns else "source" if "source" in adf.columns else None
                if src_col:
                    top_src = adf.groupby("cluster_id")[src_col].agg(lambda s: s.value_counts().index[0] if len(s) > 0 else "")
                    if "top_subreddit" in df.columns:
                        df["top_subreddit"] = df["cluster_id"].map(top_src).fillna(df["top_subreddit"])
                    elif "top_source" in df.columns:
                        df["top_source"] = df["cluster_id"].map(top_src).fillna(df["top_source"])

    df = df[df["count"] >= min_count]
    df = df.sort_values("count", ascending=False).head(limit)
    df = df.copy()

    # Ensure keywords column exists
    if "keywords" not in df.columns:
        # Try platform-specific keywords
        if platform == "news":
            kw = get_news_cluster_keywords()
        elif platform == "tiktok":
            kw = get_tiktok_cluster_keywords()
        else:
            kw = get_cluster_keywords()
        if not kw.empty:
            kw_map = {row["cluster_id"]: str(row["keywords"])[:60] for _, row in kw.iterrows() if isinstance(row.get("keywords"), str)}
            df["keywords"] = df["cluster_id"].map(kw_map).fillna("")
        else:
            df["keywords"] = ""

    df["keywords_short"] = df["keywords"].apply(
        lambda k: ", ".join(str(k).split(", ")[:3]) if k else ""
    )

    # theme column may not exist for news/tiktok
    if "theme" not in df.columns:
        df["theme"] = df["keywords_short"]
    else:
        df["theme"] = df.apply(
            lambda r: r["keywords_short"] if r.get("theme") in (None, "Error", "") else r["theme"],
            axis=1,
        )

    # Ensure expected columns exist
    for col in ["top_subreddit", "sentiment_mean", "time_start", "time_end"]:
        if col not in df.columns:
            if col == "top_subreddit":
                df[col] = df.get("top_source", "")
            elif col == "sentiment_mean":
                df[col] = None
            else:
                df[col] = ""

    return df.to_dict(orient="records")


@router.get("/keywords")
def cluster_keywords(
    cluster_id: Optional[int] = Query(None),
    limit: int = Query(50),
):
    df = get_cluster_keywords()
    if cluster_id is not None:
        df = df[df["cluster_id"] == cluster_id]
    else:
        df = df.head(limit)
    return df.to_dict(orient="records")


@router.get("/temporal")
def temporal_clusters(
    cluster_id: Optional[int] = Query(None),
    start: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end: Optional[str] = Query(None, description="End month YYYY-MM"),
    limit: int = Query(20, description="Top N clusters by total count"),
    platform: Optional[str] = Query(None),
):
    """Get temporal cluster activity."""
    if platform == "news":
        df = get_news_temporal_clusters()
    elif platform == "tiktok":
        df = get_tiktok_temporal_clusters()
    else:
        df = get_temporal_clusters()

    if cluster_id is not None:
        df = df[df["cluster_id"] == cluster_id]
    else:
        top_clusters = (
            df.groupby("cluster_id")["count"]
            .sum()
            .nlargest(limit)
            .index.tolist()
        )
        df = df[df["cluster_id"].isin(top_clusters)]

    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]

    return df.to_dict(orient="records")


@router.get("/monthly")
def clusters_monthly(
    month: str = Query(..., description="Month YYYY-MM"),
    top_n: int = Query(15, ge=1, le=50),
    platform: Optional[str] = Query(None),
):
    """Get top N clusters for a specific month."""
    if platform == "news":
        df = get_news_clusters_monthly()
    elif platform == "tiktok":
        df = get_tiktok_clusters_monthly()
    else:
        df = get_clusters_monthly()
    if df.empty:
        return []
    filtered = df[df["year_month"] == month].nlargest(top_n, "count")
    return filtered.to_dict(orient="records")


@router.get("/monthly/months")
def clusters_monthly_months(platform: Optional[str] = Query(None)):
    """Get list of all available months for cluster slider."""
    if platform == "news":
        df = get_news_clusters_monthly()
    elif platform == "tiktok":
        df = get_tiktok_clusters_monthly()
    else:
        df = get_clusters_monthly()
    if df.empty:
        return []
    return sorted(df["year_month"].unique().tolist())


@router.get("/scatter")
def cluster_scatter(
    top_n: int = Query(50, description="Number of top clusters to include"),
    max_points: int = Query(30000, description="Maximum number of points"),
    start: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end: Optional[str] = Query(None, description="End month YYYY-MM"),
    platform: Optional[str] = Query(None),
):
    """Get scatter plot data: stratified sample of top clusters with UMAP coords."""
    if platform == "news":
        from ..services.data_service import get_news_cluster_keywords
        assignments_fn = lambda: _load_platform_assignments("news")
        kw_fn = get_news_cluster_keywords
    elif platform == "tiktok":
        from ..services.data_service import get_tiktok_cluster_keywords
        assignments_fn = lambda: _load_platform_assignments("tiktok")
        kw_fn = get_tiktok_cluster_keywords
    else:
        return _scatter_data(top_n, max_points, start, end)

    assignments = assignments_fn()
    if assignments is None or assignments.empty:
        return []
    df = assignments[assignments["cluster_id"] != -1]
    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]
    if df.empty:
        return []
    top_clusters = df["cluster_id"].value_counts().nlargest(top_n).index.tolist()
    df = df[df["cluster_id"].isin(top_clusters)]
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
    kw = kw_fn()
    kw_map = {row["cluster_id"]: ", ".join(str(row["keywords"]).split(", ")[:3]) for _, row in kw.iterrows() if isinstance(row.get("keywords"), str)} if not kw.empty else {}
    df = df.copy()
    df["keywords"] = df["cluster_id"].map(kw_map).fillna("")
    src_col = "source" if "source" in df.columns else "subreddit" if "subreddit" in df.columns else None
    result = df[["umap_1", "umap_2", "cluster_id", "keywords"]].copy()
    if src_col:
        result["subreddit"] = df[src_col]
    else:
        result["subreddit"] = ""
    result = result.rename(columns={"umap_1": "x", "umap_2": "y"})
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.where(result.notna(), None).to_dict(orient="records")


def _load_platform_assignments(platform: str):
    """Load cluster assignments for news/tiktok."""
    from ..services.data_service import NEWS_CLUSTERS_DIR, TIKTOK_CLUSTERS_DIR
    if platform == "news":
        path = NEWS_CLUSTERS_DIR / "cluster_assignments.parquet"
    elif platform == "tiktok":
        path = TIKTOK_CLUSTERS_DIR / "cluster_assignments.parquet"
    else:
        return None
    if path.exists():
        return pd.read_parquet(path)
    return None
