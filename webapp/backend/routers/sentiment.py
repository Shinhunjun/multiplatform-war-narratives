"""Sentiment analysis endpoints."""

from typing import Optional

import numpy as np
from fastapi import APIRouter, Query

from ..services.data_service import (
    get_sentiment_by_month,
    get_sentiment_by_subreddit,
    get_sentiment_by_subreddit_month,
    get_news_sentiment_by_month,
    get_news_sentiment_by_source,
    get_news_sentiment_by_source_month,
)

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])


@router.get("/by-month")
def sentiment_by_month(
    platform: Optional[str] = Query(None, description="Platform: reddit or news"),
    start: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end: Optional[str] = Query(None, description="End month YYYY-MM"),
):
    if platform == "news":
        df = get_news_sentiment_by_month()
    else:
        df = get_sentiment_by_month()
    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]
    return df.to_dict(orient="records")


@router.get("/by-subreddit")
def sentiment_by_subreddit(
    platform: Optional[str] = Query(None, description="Platform: reddit or news"),
):
    if platform == "news":
        df = get_news_sentiment_by_source()
        return df.to_dict(orient="records")
    df = get_sentiment_by_subreddit()
    return df.to_dict(orient="records")


@router.get("/by-subreddit-month")
def sentiment_by_subreddit_month(
    platform: Optional[str] = Query(None, description="Platform: reddit or news"),
    subreddit: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    if platform == "news":
        df = get_news_sentiment_by_source_month()
        if subreddit:
            df = df[df["source"] == subreddit]
    else:
        df = get_sentiment_by_subreddit_month()
        if subreddit:
            df = df[df["subreddit"] == subreddit]
    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]
    return df.to_dict(orient="records")


@router.get("/boxplot")
def sentiment_boxplot(
    start: Optional[str] = Query(None, description="Start month YYYY-MM"),
    end: Optional[str] = Query(None, description="End month YYYY-MM"),
):
    """Box plot statistics per subreddit from monthly sentiment values. Reddit only."""
    df = get_sentiment_by_subreddit_month()
    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]

    results = []
    for subreddit, group in df.groupby("subreddit"):
        vals = group["mean_sentiment"].dropna()
        if len(vals) < 3:
            continue
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        results.append({
            "subreddit": subreddit,
            "min": round(float(vals.min()), 4),
            "q1": round(float(q1), 4),
            "median": round(float(median), 4),
            "q3": round(float(q3), 4),
            "max": round(float(vals.max()), 4),
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "count": int(group["total_count"].sum()),
        })

    results.sort(key=lambda x: x["mean"])
    return results
