"""TikTok-specific endpoints: hashtags, engagement, regions."""

from typing import Optional

from fastapi import APIRouter, Query

from ..services.data_service import (
    get_tiktok_hashtag_trends,
    get_tiktok_engagement_metrics,
    get_tiktok_region_distribution,
)

router = APIRouter(prefix="/api/tiktok", tags=["tiktok"])


@router.get("/hashtags")
def hashtag_trends(
    start: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end: Optional[str] = Query(None, description="End month YYYY-MM"),
    top_n: int = Query(20, ge=1, le=100),
):
    """Top hashtags with frequency and sentiment."""
    df = get_tiktok_hashtag_trends()
    if df.empty:
        return []

    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]

    # Aggregate across time for overall ranking
    agg = df.groupby("hashtag").agg(
        total_count=("count", "sum"),
        mean_sentiment=("mean_sentiment", "mean"),
    ).reset_index().nlargest(top_n, "total_count")

    return agg.to_dict(orient="records")


@router.get("/hashtags/over-time")
def hashtag_over_time(
    hashtags: Optional[str] = Query(None, description="Comma-separated hashtag list"),
    top_n: int = Query(10, ge=1, le=30),
):
    """Hashtag frequency over time."""
    df = get_tiktok_hashtag_trends()
    if df.empty:
        return []

    if hashtags:
        ht_list = [h.strip().lower() for h in hashtags.split(",")]
        df = df[df["hashtag"].isin(ht_list)]
    else:
        # Use top N hashtags by total count
        top = df.groupby("hashtag")["count"].sum().nlargest(top_n).index
        df = df[df["hashtag"].isin(top)]

    return df.to_dict(orient="records")


@router.get("/engagement")
def engagement_metrics(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Monthly engagement metrics (views, likes, shares, comments)."""
    df = get_tiktok_engagement_metrics()
    if df.empty:
        return []

    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]

    return df.to_dict(orient="records")


@router.get("/regions")
def region_distribution(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    top_n: int = Query(15, ge=1, le=50),
):
    """Video count by region."""
    df = get_tiktok_region_distribution()
    if df.empty:
        return []

    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]

    agg = df.groupby("region_code").agg(
        total_count=("count", "sum"),
    ).reset_index().nlargest(top_n, "total_count")

    return agg.to_dict(orient="records")
