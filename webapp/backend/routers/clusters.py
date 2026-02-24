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
    get_temporal_clusters,
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

@router.get("/summaries")
def cluster_summaries(
    limit: int = Query(50, description="Max clusters to return"),
    min_count: int = Query(10, description="Minimum document count"),
    start: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end: Optional[str] = Query(None, description="End month YYYY-MM"),
):
    """Get cluster summaries sorted by document count, with keywords."""
    df = get_cluster_summaries()

    # If time range specified, recompute counts from assignments (cached)
    if start or end:
        counts = _filtered_counts(start, end)
        if counts is not None:
            df = df.copy()
            df["count"] = df["cluster_id"].map(counts).fillna(0).astype(int)

    df = df[df["count"] >= min_count]
    df = df.sort_values("count", ascending=False).head(limit)

    # Join keywords and use top-3 keywords as label when theme is "Error"
    kw_map = _keywords_map()
    df = df.copy()
    df["keywords"] = df["cluster_id"].map(kw_map).fillna("")
    df["keywords_short"] = df["keywords"].apply(
        lambda k: ", ".join(k.split(", ")[:3]) if isinstance(k, str) and k else ""
    )
    df["theme"] = df.apply(
        lambda r: r["keywords_short"] if r["theme"] == "Error" and r["keywords_short"] else r["theme"],
        axis=1,
    )
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
):
    """Get temporal cluster activity."""
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


@router.get("/scatter")
def cluster_scatter(
    top_n: int = Query(50, description="Number of top clusters to include"),
    max_points: int = Query(30000, description="Maximum number of points"),
    start: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end: Optional[str] = Query(None, description="End month YYYY-MM"),
):
    """Get scatter plot data: stratified sample of top clusters with UMAP coords."""
    return _scatter_data(top_n, max_points, start, end)
