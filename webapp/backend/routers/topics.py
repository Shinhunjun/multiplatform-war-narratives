"""Topic modeling endpoints."""

from typing import Optional

from fastapi import APIRouter, Query

from ..services.data_service import (
    get_topic_info,
    get_topics_by_subreddit,
    get_topics_over_time,
    get_news_topic_info,
    get_news_topics_over_time,
    get_news_topics_by_source,
)

router = APIRouter(prefix="/api/topics", tags=["topics"])


@router.get("/info")
def topic_info(platform: Optional[str] = Query(None)):
    """Get all topics with their keywords and counts."""
    if platform == "news":
        df = get_news_topic_info()
    else:
        df = get_topic_info()
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
