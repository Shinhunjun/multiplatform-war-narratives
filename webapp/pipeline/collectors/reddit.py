"""
Reddit data collector using Arctic Shift API directly.
Fetches submissions and comments via https://arctic-shift.photon-reddit.com/api
No authentication required.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

from ..config import PipelineConfig

logger = logging.getLogger(__name__)

BASE_URL = "https://arctic-shift.photon-reddit.com/api"

SUBMISSION_FIELDS = "id,title,selftext,author,subreddit,created_utc,score,num_comments,url"
COMMENT_FIELDS = "id,body,author,subreddit,created_utc,score,link_id,parent_id"


class RedditCollector:
    """Collects Reddit submissions and comments via Arctic Shift API."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def _request(
        self, session: aiohttp.ClientSession, endpoint: str, params: dict
    ) -> list:
        """Make a single API request with retry logic."""
        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(self.config.arctic_max_retries):
            try:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.arctic_timeout),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("data", [])
                    elif resp.status == 429:
                        wait = self.config.arctic_backoff_sec * (attempt + 1)
                        logger.warning(f"Rate limited. Waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logger.error(f"API {resp.status}: {await resp.text()}")
                        return []
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt+1}): {e}")
                await asyncio.sleep(self.config.arctic_backoff_sec)

        return []

    async def _search_submissions(
        self,
        session: aiohttp.ClientSession,
        subreddit: str,
        after: str,
        before: str,
        query: Optional[str] = None,
    ) -> List[dict]:
        """Search submissions in a subreddit."""
        params = {
            "subreddit": subreddit,
            "after": after,
            "before": before,
            "limit": 100,
            "sort": "desc",
            "fields": SUBMISSION_FIELDS,
        }
        if query:
            params["title"] = query

        await asyncio.sleep(self.config.arctic_sleep_sec)
        return await self._request(session, "posts/search", params)

    async def _search_comments(
        self,
        session: aiohttp.ClientSession,
        link_id: str,
    ) -> List[dict]:
        """Fetch comments for a specific submission."""
        params = {
            "link_id": f"t3_{link_id}",
            "limit": 100,
            "sort": "desc",
            "fields": COMMENT_FIELDS,
        }
        await asyncio.sleep(self.config.arctic_sleep_sec)
        return await self._request(session, "comments/search", params)

    async def collect_submissions(
        self, session: aiohttp.ClientSession, start_date: str, end_date: str
    ) -> Dict[str, dict]:
        """Collect submissions from all subreddits for a date range."""
        all_submissions: Dict[str, dict] = {}

        for subreddit in self.config.subreddits:
            is_venezuela_sub = subreddit.lower() in [
                s.lower() for s in self.config.venezuela_subreddits
            ]

            if is_venezuela_sub:
                # Fetch all posts from Venezuela subs (no keyword filter)
                results = await self._search_submissions(
                    session, subreddit, start_date, end_date
                )
                for post in results:
                    all_submissions[post["id"]] = post

            else:
                # Use keyword queries for general subs, deduplicate by ID
                for query in self.config.reddit_queries:
                    results = await self._search_submissions(
                        session, subreddit, start_date, end_date, query=query
                    )
                    for post in results:
                        pid = post["id"]
                        if pid not in all_submissions:
                            post["_matched_queries"] = [query]
                            all_submissions[pid] = post
                        else:
                            existing = all_submissions[pid].get("_matched_queries", [])
                            if query not in existing:
                                existing.append(query)
                            all_submissions[pid]["_matched_queries"] = existing

            count = sum(
                1 for p in all_submissions.values()
                if p.get("subreddit", "").lower() == subreddit.lower()
            )
            logger.info(f"r/{subreddit}: {count} submissions")

        logger.info(f"Total submissions: {len(all_submissions)}")
        return all_submissions

    async def collect_comments(
        self,
        session: aiohttp.ClientSession,
        submissions: Dict[str, dict],
        min_comments: int = 1,
    ) -> Dict[str, dict]:
        """Collect comments for submissions with enough comments."""
        all_comments: Dict[str, dict] = {}

        # Filter and sort by num_comments
        posts = [
            (pid, pdata) for pid, pdata in submissions.items()
            if pdata.get("num_comments", 0) >= min_comments
        ]
        posts.sort(key=lambda x: x[1].get("num_comments", 0), reverse=True)

        logger.info(f"Fetching comments for {len(posts)} submissions...")

        for post_id, post_data in posts:
            try:
                results = await self._search_comments(session, post_id)
                for comment in results:
                    cid = comment.get("id", "")
                    if cid:
                        comment["_submission_id"] = post_id
                        all_comments[cid] = comment
            except Exception as e:
                logger.debug(f"Comment fetch failed for {post_id}: {e}")

        logger.info(f"Total comments: {len(all_comments)}")
        return all_comments

    def save_raw(
        self, submissions: Dict, comments: Dict, run_date: str
    ) -> tuple[Path, Path]:
        """Save raw collected data as JSON."""
        sub_dir = self.config.raw_dir / "reddit" / "submissions"
        com_dir = self.config.raw_dir / "reddit" / "comments"

        sub_path = sub_dir / f"submissions_{run_date}.json"
        com_path = com_dir / f"comments_{run_date}.json"

        with open(sub_path, "w") as f:
            json.dump(submissions, f, ensure_ascii=False, indent=2)

        with open(com_path, "w") as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(submissions)} submissions, {len(comments)} comments")
        return sub_path, com_path

    async def _run_async(self, run_date: str) -> dict:
        """Async execution of the full collection pipeline."""
        end = datetime.strptime(run_date, "%Y-%m-%d")
        start = end - timedelta(days=self.config.lookback_days)

        start_str = f"{start.strftime('%Y-%m-%d')}T00:00:00"
        end_str = f"{end.strftime('%Y-%m-%d')}T23:59:59"

        logger.info(f"Collecting Reddit data: {start_str} to {end_str}")

        async with aiohttp.ClientSession() as session:
            submissions = await self.collect_submissions(session, start_str, end_str)
            comments = await self.collect_comments(session, submissions)

        sub_path, com_path = self.save_raw(submissions, comments, run_date)

        return {
            "submissions_count": len(submissions),
            "comments_count": len(comments),
            "submissions_path": str(sub_path),
            "comments_path": str(com_path),
        }

    def run(self, run_date: str) -> dict:
        """Execute full Reddit collection pipeline."""
        return asyncio.run(self._run_async(run_date))
