"""Sentiment analysis endpoints."""

from typing import Optional

from fastapi import APIRouter, Query

from ..services.data_service import (
    get_sentiment_by_month,
    get_sentiment_by_subreddit,
    get_sentiment_by_subreddit_month,
)

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])


@router.get("/by-month")
def sentiment_by_month(
    start: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end: Optional[str] = Query(None, description="End month YYYY-MM"),
):
    df = get_sentiment_by_month()
    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]
    return df.to_dict(orient="records")


@router.get("/by-subreddit")
def sentiment_by_subreddit():
    df = get_sentiment_by_subreddit()
    return df.to_dict(orient="records")


@router.get("/by-subreddit-month")
def sentiment_by_subreddit_month(
    subreddit: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    df = get_sentiment_by_subreddit_month()
    if subreddit:
        df = df[df["subreddit"] == subreddit]
    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]
    return df.to_dict(orient="records")
