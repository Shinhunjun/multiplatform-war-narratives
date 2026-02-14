"""
Preprocessing for newly collected data.
Mirrors the existing preprocessing pipeline but operates incrementally.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import PipelineConfig

logger = logging.getLogger(__name__)

# Bot accounts to filter out
BOT_AUTHORS = {
    "AutoModerator", "autotldr", "empleadoEstatalBot",
    "RemindMeBot", "bot", "WikiTextBot", "TotesMessenger",
    "RepostSleuthBot", "SaveVideo", "VisualMod",
}

# Minimum word count
MIN_WORDS = 5


def clean_text(text: str) -> str:
    """Clean text for analysis."""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    # Remove markdown formatting
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Remove Reddit quote markers
    text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)
    # Remove code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_submission(record: dict) -> bool:
    """Check if a submission should be included."""
    author = record.get("author", "")
    if author in BOT_AUTHORS or author == "[deleted]":
        return False

    title = record.get("title", "")
    selftext = record.get("selftext", "")
    combined = f"{title} {selftext}".strip()

    if combined in ("[deleted]", "[removed]", ""):
        return False
    if len(combined.split()) < MIN_WORDS:
        return False

    return True


def is_valid_comment(record: dict) -> bool:
    """Check if a comment should be included."""
    author = record.get("author", "")
    if author in BOT_AUTHORS or author == "[deleted]":
        return False

    body = record.get("body", "")
    if body in ("[deleted]", "[removed]", ""):
        return False
    if len(body.split()) < MIN_WORDS:
        return False

    return True


def preprocess_reddit(
    submissions: List[dict], comments: List[dict]
) -> pd.DataFrame:
    """
    Preprocess Reddit data into a unified DataFrame.
    Same structure as the existing analysis pipeline expects.
    """
    rows = []

    for sub in submissions:
        if not is_valid_submission(sub):
            continue
        text = clean_text(f"{sub['title']} {sub.get('selftext', '')}".strip())
        if len(text.split()) < MIN_WORDS:
            continue

        from datetime import datetime, timezone
        created_dt = datetime.fromtimestamp(sub["created_utc"], tz=timezone.utc)

        rows.append({
            "id": sub["id"],
            "type": "submission",
            "subreddit": sub["subreddit"],
            "author": sub["author"],
            "text": text,
            "score": sub.get("score", 0),
            "created_utc": sub["created_utc"],
            "created_datetime": created_dt,
            "year": created_dt.year,
            "month": created_dt.month,
            "year_month": created_dt.strftime("%Y-%m"),
        })

    for com in comments:
        if not is_valid_comment(com):
            continue
        text = clean_text(com["body"])
        if len(text.split()) < MIN_WORDS:
            continue

        from datetime import datetime, timezone
        created_dt = datetime.fromtimestamp(com["created_utc"], tz=timezone.utc)

        rows.append({
            "id": com["id"],
            "type": "comment",
            "subreddit": com["subreddit"],
            "author": com["author"],
            "text": text,
            "score": com.get("score", 0),
            "created_utc": com["created_utc"],
            "created_datetime": created_dt,
            "year": created_dt.year,
            "month": created_dt.month,
            "year_month": created_dt.strftime("%Y-%m"),
        })

    df = pd.DataFrame(rows)
    logger.info(
        f"Preprocessed: {len(df)} records "
        f"({len([r for r in rows if r['type'] == 'submission'])} submissions, "
        f"{len([r for r in rows if r['type'] == 'comment'])} comments)"
    )
    return df


def preprocess_gdelt(events_df: pd.DataFrame, articles: List[dict]) -> pd.DataFrame:
    """
    Preprocess GDELT events and scraped articles into a unified DataFrame.
    """
    rows = []

    # Process scraped articles
    for article in articles:
        text = clean_text(article.get("text", ""))
        if len(text.split()) < 20:  # Higher threshold for news articles
            continue

        rows.append({
            "id": hash(article["url"]) & 0xFFFFFFFF,  # numeric ID from URL hash
            "type": "news_article",
            "source": "gdelt",
            "title": article.get("title", ""),
            "text": text,
            "url": article["url"],
            "scraped_at": article.get("scraped_at", ""),
        })

    # Process GDELT events metadata (tone, actors, etc.)
    if not events_df.empty:
        for _, row in events_df.iterrows():
            rows.append({
                "id": str(row.get("GLOBALEVENTID", "")),
                "type": "gdelt_event",
                "source": "gdelt",
                "actor1": row.get("Actor1Name", ""),
                "actor2": row.get("Actor2Name", ""),
                "event_code": row.get("EventCode", ""),
                "goldstein_scale": row.get("GoldsteinScale", 0),
                "avg_tone": row.get("AvgTone", 0),
                "num_mentions": row.get("NumMentions", 0),
                "num_articles": row.get("NumArticles", 0),
                "source_url": row.get("SOURCEURL", ""),
                "date": str(row.get("SQLDATE", "")),
            })

    df = pd.DataFrame(rows)
    logger.info(f"Preprocessed GDELT: {len(df)} records")
    return df


class Preprocessor:
    """Orchestrates preprocessing for all data sources."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_raw_reddit(self, run_date: str) -> tuple[List[dict], List[dict]]:
        """Load raw Reddit JSON files for a run date."""
        sub_path = self.config.raw_dir / "reddit" / "submissions" / f"submissions_{run_date}.json"
        com_path = self.config.raw_dir / "reddit" / "comments" / f"comments_{run_date}.json"

        submissions = []
        comments = []

        if sub_path.exists():
            with open(sub_path) as f:
                submissions = json.load(f)

        if com_path.exists():
            with open(com_path) as f:
                comments = json.load(f)

        return submissions, comments

    def load_raw_gdelt(self, run_date: str) -> tuple[pd.DataFrame, List[dict]]:
        """Load raw GDELT data for a run date."""
        events_path = self.config.raw_dir / "gdelt" / f"events_{run_date}.parquet"
        articles_path = self.config.raw_dir / "news" / f"articles_{run_date}.json"

        events_df = pd.DataFrame()
        articles = []

        if events_path.exists():
            events_df = pd.read_parquet(events_path)

        if articles_path.exists():
            with open(articles_path) as f:
                articles = json.load(f)

        return events_df, articles

    def run(self, run_date: str) -> dict:
        """Run preprocessing for all sources."""
        results = {}

        # Reddit
        submissions, comments = self.load_raw_reddit(run_date)
        if submissions or comments:
            reddit_df = preprocess_reddit(submissions, comments)
            if not reddit_df.empty:
                out_path = self.config.processed_dir / "reddit" / f"reddit_{run_date}.parquet"
                reddit_df.to_parquet(out_path, index=False)
                results["reddit"] = {
                    "count": len(reddit_df),
                    "path": str(out_path),
                }

        # GDELT
        events_df, articles = self.load_raw_gdelt(run_date)
        if not events_df.empty or articles:
            gdelt_df = preprocess_gdelt(events_df, articles)
            if not gdelt_df.empty:
                out_path = self.config.processed_dir / "gdelt" / f"gdelt_{run_date}.parquet"
                gdelt_df.to_parquet(out_path, index=False)
                results["gdelt"] = {
                    "count": len(gdelt_df),
                    "path": str(out_path),
                }

        return results
