"""Topic modeling endpoints."""

import functools
from typing import Optional

from fastapi import APIRouter, Query

from ..services.data_service import (
    get_topic_info,
    get_topics_by_subreddit,
    get_topics_over_time,
    get_topics_monthly,
    get_topics_monthly_fitted,
    get_news_topic_info,
    get_news_topics_over_time,
    get_news_topics_by_source,
    get_news_topics_monthly,
    get_news_topics_monthly_fitted,
)

router = APIRouter(prefix="/api/topics", tags=["topics"])


@functools.lru_cache(maxsize=64)
def _topic_freq(platform: Optional[str], start: Optional[str], end: Optional[str]) -> dict:
    """Topic frequency sums filtered by time range. Cached per combo."""
    tot = get_news_topics_over_time() if platform == "news" else get_topics_over_time()
    if tot.empty:
        return {}
    filtered = tot.copy()
    filtered["_month"] = filtered["Timestamp"].str[:7]
    if start:
        filtered = filtered[filtered["_month"] >= start]
    if end:
        filtered = filtered[filtered["_month"] <= end]
    return filtered.groupby("Topic")["Frequency"].sum().to_dict()


@router.get("/info")
def topic_info(
    platform: Optional[str] = Query(None),
    start: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end: Optional[str] = Query(None, description="End month YYYY-MM"),
):
    """Get all topics with their keywords and counts."""
    if platform == "news":
        df = get_news_topic_info()
    else:
        df = get_topic_info()

    # Recompute counts from over-time data when time range is given (cached)
    if start or end:
        freq_by_topic = _topic_freq(platform, start, end)
        if freq_by_topic:
            df = df.copy()
            df["Count"] = df["Topic"].map(freq_by_topic).fillna(0).astype(int)
            df = df[df["Count"] > 0]

    return df.to_dict(orient="records")


@router.get("/by-subreddit")
def topics_by_subreddit(
    platform: Optional[str] = Query(None),
    subreddit: Optional[str] = Query(None),
):
    if platform == "news":
        df = get_news_topics_by_source()
        if subreddit:
            df = df[df["source"] == subreddit]
    else:
        df = get_topics_by_subreddit()
        if subreddit:
            df = df[df["subreddit"] == subreddit]
    return df.to_dict(orient="records")


@router.get("/monthly")
def topics_monthly(
    month: str = Query(..., description="Month YYYY-MM"),
    top_n: int = Query(15, ge=1, le=50),
    platform: Optional[str] = Query(None),
):
    """Get top N topics for a specific month."""
    if platform == "news":
        df = get_news_topics_monthly()
    else:
        df = get_topics_monthly()

    if df.empty:
        return []

    filtered = df[df["year_month"] == month].nlargest(top_n, "count")
    return filtered.to_dict(orient="records")


@router.get("/monthly/months")
def topics_monthly_months(platform: Optional[str] = Query(None)):
    """Get list of all available months."""
    if platform == "news":
        df = get_news_topics_monthly()
    else:
        df = get_topics_monthly()

    if df.empty:
        return []

    return sorted(df["year_month"].unique().tolist())


@router.get("/monthly-fitted")
def topics_monthly_fitted(
    month: str = Query(..., description="Month YYYY-MM"),
    top_n: int = Query(15, ge=1, le=50),
    platform: Optional[str] = Query(None),
):
    """Get top N independently-fitted topics for a specific month."""
    if platform == "news":
        df = get_news_topics_monthly_fitted()
    else:
        df = get_topics_monthly_fitted()

    if df.empty:
        return []

    filtered = df[df["year_month"] == month].nlargest(top_n, "count")
    return filtered.to_dict(orient="records")


@router.get("/monthly-fitted/months")
def topics_monthly_fitted_months(platform: Optional[str] = Query(None)):
    """Get list of available months for independently-fitted topics."""
    if platform == "news":
        df = get_news_topics_monthly_fitted()
    else:
        df = get_topics_monthly_fitted()

    if df.empty:
        return []

    return sorted(df["year_month"].unique().tolist())


@router.get("/over-time")
def topics_over_time(
    platform: Optional[str] = Query(None),
    topic_id: Optional[int] = Query(None),
):
    if platform == "news":
        df = get_news_topics_over_time()
    else:
        df = get_topics_over_time()
    if topic_id is not None:
        df = df[df["Topic"] == topic_id]
    return df.to_dict(orient="records")
