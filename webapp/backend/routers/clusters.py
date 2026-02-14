"""Clustering endpoints."""

from typing import Optional

from fastapi import APIRouter, Query

from ..services.data_service import (
    get_cluster_keywords,
    get_cluster_summaries,
    get_temporal_clusters,
)

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


@router.get("/summaries")
def cluster_summaries(
    limit: int = Query(50, description="Max clusters to return"),
    min_count: int = Query(10, description="Minimum document count"),
):
    """Get cluster summaries sorted by document count."""
    df = get_cluster_summaries()
    df = df[df["count"] >= min_count]
    df = df.sort_values("count", ascending=False).head(limit)
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
        # Return only top N clusters by total count
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
