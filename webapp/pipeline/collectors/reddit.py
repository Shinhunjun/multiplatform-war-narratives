"""
Reddit data collector using PRAW (Python Reddit API Wrapper).
Fetches new submissions and comments from monitored subreddits.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import praw

from ..config import PipelineConfig

logger = logging.getLogger(__name__)


class RedditCollector:
    """Collects new Reddit submissions and comments about Venezuela."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.reddit = praw.Reddit(
            client_id=config.reddit_client_id,
            client_secret=config.reddit_client_secret,
            user_agent=config.reddit_user_agent,
        )
        self.reddit.read_only = True

    def _matches_keywords(self, text: str) -> bool:
        """Check if text contains any Venezuela-related keywords."""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.config.reddit_keywords)

    def collect_submissions(
        self, lookback_hours: int = 24
    ) -> List[dict]:
        """Fetch new submissions from all monitored subreddits."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        all_submissions = []

        for sub_name in self.config.subreddits:
            try:
                subreddit = self.reddit.subreddit(sub_name)
                count = 0

                for submission in subreddit.new(limit=self.config.reddit_posts_per_sub):
                    created = datetime.fromtimestamp(
                        submission.created_utc, tz=timezone.utc
                    )
                    if created < cutoff:
                        break

                    # For Venezuela-specific subs, take everything
                    # For general subs, filter by keywords
                    is_venezuela_sub = sub_name.lower() in ("venezuela", "vzla")
                    title_body = f"{submission.title} {submission.selftext or ''}"

                    if is_venezuela_sub or self._matches_keywords(title_body):
                        record = {
                            "id": submission.id,
                            "subreddit": sub_name,
                            "author": str(submission.author) if submission.author else "[deleted]",
                            "title": submission.title,
                            "selftext": submission.selftext or "",
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "created_utc": submission.created_utc,
                            "created_datetime": created.isoformat(),
                            "url": submission.url,
                            "permalink": submission.permalink,
                            "upvote_ratio": submission.upvote_ratio,
                        }
                        all_submissions.append(record)
                        count += 1

                logger.info(f"r/{sub_name}: {count} new submissions")

            except Exception as e:
                logger.error(f"Error fetching r/{sub_name}: {e}")

        logger.info(f"Total new submissions: {len(all_submissions)}")
        return all_submissions

    def collect_comments(
        self,
        submission_ids: Optional[List[str]] = None,
        lookback_hours: int = 24,
    ) -> List[dict]:
        """Fetch comments for recent submissions."""
        all_comments = []

        if submission_ids is None:
            # Collect from subreddits' new comments
            for sub_name in self.config.subreddits:
                try:
                    subreddit = self.reddit.subreddit(sub_name)
                    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
                    count = 0

                    for comment in subreddit.comments(limit=self.config.reddit_comments_per_post * 5):
                        created = datetime.fromtimestamp(
                            comment.created_utc, tz=timezone.utc
                        )
                        if created < cutoff:
                            break

                        is_venezuela_sub = sub_name.lower() in ("venezuela", "vzla")
                        if is_venezuela_sub or self._matches_keywords(comment.body):
                            record = {
                                "id": comment.id,
                                "submission_id": comment.link_id.replace("t3_", ""),
                                "subreddit": sub_name,
                                "author": str(comment.author) if comment.author else "[deleted]",
                                "body": comment.body,
                                "score": comment.score,
                                "created_utc": comment.created_utc,
                                "created_datetime": created.isoformat(),
                                "parent_id": comment.parent_id,
                            }
                            all_comments.append(record)
                            count += 1

                    logger.info(f"r/{sub_name}: {count} new comments")

                except Exception as e:
                    logger.error(f"Error fetching comments from r/{sub_name}: {e}")
        else:
            # Fetch comments for specific submissions
            for sid in submission_ids:
                try:
                    submission = self.reddit.submission(id=sid)
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list()[
                        : self.config.reddit_comments_per_post
                    ]:
                        record = {
                            "id": comment.id,
                            "submission_id": sid,
                            "subreddit": str(submission.subreddit),
                            "author": str(comment.author) if comment.author else "[deleted]",
                            "body": comment.body,
                            "score": comment.score,
                            "created_utc": comment.created_utc,
                            "created_datetime": datetime.fromtimestamp(
                                comment.created_utc, tz=timezone.utc
                            ).isoformat(),
                            "parent_id": comment.parent_id,
                        }
                        all_comments.append(record)
                except Exception as e:
                    logger.error(f"Error fetching comments for {sid}: {e}")

        logger.info(f"Total new comments: {len(all_comments)}")
        return all_comments

    def save_raw(
        self, submissions: List[dict], comments: List[dict], run_date: str
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

        logger.info(f"Saved {len(submissions)} submissions to {sub_path}")
        logger.info(f"Saved {len(comments)} comments to {com_path}")

        return sub_path, com_path

    def run(self, run_date: str) -> dict:
        """Execute full Reddit collection pipeline."""
        lookback = self.config.lookback_days * 24

        submissions = self.collect_submissions(lookback_hours=lookback)
        submission_ids = [s["id"] for s in submissions]
        comments = self.collect_comments(
            submission_ids=submission_ids, lookback_hours=lookback
        )

        sub_path, com_path = self.save_raw(submissions, comments, run_date)

        return {
            "submissions_count": len(submissions),
            "comments_count": len(comments),
            "submissions_path": str(sub_path),
            "comments_path": str(com_path),
        }
