"""
Data collection functions for Reddit submissions and comments.
Uses Arctic Shift API via BAScraper.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from BAScraper.BAScraper_async import ArcticShiftAsync
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from .config import (
    COMMENT_FIELDS_TO_KEEP,
    COMMENTS_DEFAULT_END,
    COMMENTS_DEFAULT_START,
    CRISIS_QUERIES,
    PipelineConfig,
    PRIORITY_QUERIES,
    SUBMISSION_FIELDS,
)
from .processors import load_json, save_json


def generate_monthly_windows(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """Generate monthly time windows."""
    windows = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current < end:
        month_end = current + relativedelta(months=1) - timedelta(days=1)
        if month_end > end:
            month_end = end
        windows.append((current.strftime("%Y-%m-%d"),
                       month_end.strftime("%Y-%m-%d")))
        current = current + relativedelta(months=1)
    return windows


def generate_hourly_windows(
    start_datetime: str, end_datetime: str
) -> List[Tuple[str, str]]:
    """Generate hourly time windows."""
    windows = []
    current = datetime.fromisoformat(start_datetime.replace("Z", ""))
    end = datetime.fromisoformat(end_datetime.replace("Z", ""))
    while current < end:
        hour_end = current + timedelta(hours=1)
        if hour_end > end:
            hour_end = end
        windows.append((current.isoformat(), hour_end.isoformat()))
        current = hour_end
    return windows


def generate_daily_windows(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """Generate daily time windows."""
    windows = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        windows.append(
            (
                f"{current.strftime('%Y-%m-%d')}T00:00:00",
                f"{current.strftime('%Y-%m-%d')}T23:59:59",
            )
        )
        current += timedelta(days=1)
    return windows


def create_client(config: PipelineConfig) -> ArcticShiftAsync:
    """Create configured Arctic Shift client."""
    return ArcticShiftAsync(
        sleep_sec=config.sleep_sec,
        backoff_sec=config.backoff_sec,
        max_retries=config.max_retries,
        timeout=config.timeout,
        pace_mode="auto-hard",
        task_num=config.task_num,
        save_dir=str(config.submissions_dir),
        log_stream_level="WARNING",
        log_level="WARNING",
        duplicate_action="keep_newest",
    )


async def fetch_submissions(
    client: ArcticShiftAsync,
    subreddit: str,
    start_date: str,
    end_date: str,
    query: Optional[str] = None,
    limit: int = 100,
    fields: Optional[List[str]] = None,
) -> Dict:
    """
    Fetch submissions from a subreddit with query filtering.

    Args:
        client: ArcticShiftAsync instance
        subreddit: Target subreddit name
        start_date: Start date (ISO8601 format)
        end_date: End date (ISO8601 format)
        query: Search query for title and selftext
        limit: Results per request (0 = auto-adjust)
        fields: Specific fields to return

    Returns:
        Dictionary of submission_id -> submission_data
    """
    try:
        params = {
            "mode": "submissions_search",
            "subreddit": subreddit,
            "after": start_date,
            "before": end_date,
            "limit": limit,
            "sort": "desc",
            "fields": fields or SUBMISSION_FIELDS,
        }

        if query:
            params["query"] = query

        result = await client.fetch(**params)
        return result if result else {}

    except Exception as e:
        print(f"Error fetching r/{subreddit} with query '{query}': {e}")
        return {}


async def fetch_submissions_multi_query(
    client: ArcticShiftAsync,
    subreddit: str,
    start_date: str,
    end_date: str,
    queries: List[str],
    limit: int = 100,
    fields: Optional[List[str]] = None,
) -> Dict:
    """
    Fetch submissions using multiple search queries and deduplicate results.

    Args:
        client: ArcticShiftAsync instance
        subreddit: Target subreddit name
        start_date: Start date (ISO8601 format)
        end_date: End date (ISO8601 format)
        queries: List of search queries to use
        limit: Results per request (0 = auto-adjust)
        fields: Specific fields to return

    Returns:
        Deduplicated dictionary of submission_id -> submission_data
    """
    all_results = {}

    for query in queries:
        result = await fetch_submissions(
            client=client,
            subreddit=subreddit,
            start_date=start_date,
            end_date=end_date,
            query=query,
            limit=limit,
            fields=fields,
        )

        if result:
            for post_id, post_data in result.items():
                if post_id not in all_results:
                    post_data["_matched_queries"] = [query]
                    all_results[post_id] = post_data
                else:
                    if "_matched_queries" in all_results[post_id]:
                        all_results[post_id]["_matched_queries"].append(query)
                    else:
                        all_results[post_id]["_matched_queries"] = [query]

    return all_results


async def collect_historical(
    config: PipelineConfig,
    subreddits: List[str],
    start_date: str,
    end_date: str,
    granularity: str = "monthly",
    queries: Optional[List[str]] = None,
    use_priority_queries: bool = True,
    resume: bool = True,
) -> Dict[str, int]:
    """
    Collect historical data with query-based filtering for research relevance.

    Args:
        config: Pipeline configuration
        subreddits: List of subreddit names
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        granularity: 'monthly', 'daily', or 'hourly'
        queries: Custom search queries (defaults to PRIORITY_QUERIES)
        use_priority_queries: If True and queries is None, use PRIORITY_QUERIES
        resume: If True, skip subreddits that already have output files

    Returns:
        Dictionary of subreddit -> total submissions collected
    """
    client = create_client(config)

    if queries is None:
        queries = PRIORITY_QUERIES if use_priority_queries else []

    if granularity == "monthly":
        windows = generate_monthly_windows(start_date, end_date)
    elif granularity == "daily":
        windows = generate_daily_windows(start_date, end_date)
    else:
        windows = generate_hourly_windows(
            f"{start_date}T00:00:00", f"{end_date}T23:59:59"
        )

    # Check which subreddits already have data (resume feature)
    skipped = []
    if resume:
        remaining_subreddits = []
        for sub in subreddits:
            filename = f"{sub}_{start_date.replace('-', '')}_{end_date.replace('-', '')}_{granularity}_filtered.json"
            filepath = config.submissions_dir / filename
            if filepath.exists():
                existing_data = load_json(filepath)
                skipped.append((sub, len(existing_data)))
            else:
                remaining_subreddits.append(sub)
        subreddits = remaining_subreddits

    print(f"\n{'='*70}")
    print("HISTORICAL COLLECTION (Query-Filtered)")
    print(f"{'='*70}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Granularity: {granularity} ({len(windows)} windows)")
    print(f"Subreddits: {len(subreddits)}")
    if skipped:
        print(f"Skipped (already collected): {len(skipped)}")
        for sub, count in skipped:
            print(f"  - r/{sub}: {count:,} posts")
    print(f"Search queries: {len(queries)}")
    print(f"Queries: {queries[:5]}{'...' if len(queries) > 5 else ''}")
    print(f"{'='*70}\n")

    stats = {sub: 0 for sub in subreddits}
    # Add skipped subreddits to stats
    for sub, count in skipped:
        stats[sub] = count

    if not subreddits:
        print("All subreddits already collected. Nothing to do.")
        return stats

    for subreddit in subreddits:
        subreddit_data = {}

        for window_start, window_end in tqdm(windows, desc=f"r/{subreddit}"):
            result = await fetch_submissions_multi_query(
                client=client,
                subreddit=subreddit,
                start_date=window_start,
                end_date=window_end,
                queries=queries,
                limit=100,
                fields=SUBMISSION_FIELDS,
            )

            if result:
                for post_id, post_data in result.items():
                    post_data["_window_start"] = window_start
                    post_data["_window_end"] = window_end
                    post_data["_granularity"] = granularity
                    post_data["_collection_type"] = "historical"

                for post_id, post_data in result.items():
                    if post_id not in subreddit_data:
                        subreddit_data[post_id] = post_data
                    else:
                        existing_queries = subreddit_data[post_id].get(
                            "_matched_queries", []
                        )
                        new_queries = post_data.get("_matched_queries", [])
                        subreddit_data[post_id]["_matched_queries"] = list(
                            set(existing_queries + new_queries)
                        )

        if subreddit_data:
            filename = f"{subreddit}_{start_date.replace('-', '')}_{end_date.replace('-', '')}_{granularity}_filtered.json"
            save_json(subreddit_data, config.submissions_dir / filename)
            stats[subreddit] = len(subreddit_data)
            print(
                f"r/{subreddit}: {len(subreddit_data):,} relevant submissions saved")

    total = sum(stats.values())
    print(f"\n{'='*70}")
    print(f"Collection Complete: {total:,} total relevant submissions")
    print(f"{'='*70}")

    return stats


async def collect_crisis(
    config: PipelineConfig,
    start_datetime: str,
    end_datetime: str,
    subreddits: List[str],
    queries: Optional[List[str]] = None,
    crisis_name: Optional[str] = None,
) -> Dict:
    """
    Collect crisis period data with hourly granularity and query filtering.

    Args:
        config: Pipeline configuration
        start_datetime: Start datetime (ISO format)
        end_datetime: End datetime (ISO format)
        subreddits: List of subreddit names
        queries: Search queries (defaults to CRISIS_QUERIES)
        crisis_name: Optional name for the crisis period

    Returns:
        Dictionary with submissions, hourly_stats, and metadata
    """
    client = create_client(config)
    hourly_windows = generate_hourly_windows(start_datetime, end_datetime)

    if queries is None:
        queries = CRISIS_QUERIES

    print(f"\n{'='*70}")
    print(f"CRISIS PERIOD COLLECTION: {crisis_name or 'Custom'}")
    print(f"{'='*70}")
    print(f"Period: {start_datetime} to {end_datetime}")
    print(f"Hourly windows: {len(hourly_windows)}")
    print(f"Subreddits: {', '.join(subreddits)}")
    print(f"Search queries: {len(queries)}")
    print(f"Queries: {queries[:5]}{'...' if len(queries) > 5 else ''}")
    print(f"{'='*70}\n")

    all_submissions = {}
    hourly_stats = []
    query_stats = {q: 0 for q in queries}

    for window_start, window_end in tqdm(hourly_windows, desc="Hourly windows"):
        window_data = {}

        for subreddit in subreddits:
            try:
                result = await fetch_submissions_multi_query(
                    client=client,
                    subreddit=subreddit,
                    start_date=window_start,
                    end_date=window_end,
                    queries=queries,
                    limit=100,
                    fields=SUBMISSION_FIELDS,
                )

                if result:
                    for post_id, post_data in result.items():
                        post_data["_window"] = window_start
                        post_data["_crisis"] = True
                        post_data["_crisis_name"] = crisis_name
                        post_data["_collection_type"] = "crisis"

                        for matched_query in post_data.get("_matched_queries", []):
                            if matched_query in query_stats:
                                query_stats[matched_query] += 1

                        window_data[post_id] = post_data

            except Exception:
                continue

        hourly_stats.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "count": len(window_data),
                "subreddits": len(subreddits),
            }
        )

        for post_id, post_data in window_data.items():
            if post_id not in all_submissions:
                all_submissions[post_id] = post_data

    if all_submissions:
        start_str = start_datetime[:10].replace("-", "")
        end_str = end_datetime[:10].replace("-", "")

        save_json(
            all_submissions,
            config.submissions_dir /
            f"crisis_{start_str}_{end_str}_submissions_filtered.json",
        )

        stats_df = pd.DataFrame(hourly_stats)
        stats_df.to_csv(
            config.submissions_dir /
            f"crisis_{start_str}_{end_str}_hourly_stats.csv",
            index=False,
        )

    print(f"\n{'='*70}")
    print("Crisis Collection Complete")
    print(f"{'='*70}")
    print(f"Total relevant submissions: {len(all_submissions):,}")
    print("\nTop queries by matches:")
    for query, count in sorted(query_stats.items(), key=lambda x: -x[1])[:10]:
        if count > 0:
            print(f"  '{query}': {count:,}")
    print(f"{'='*70}")

    return {
        "submissions": all_submissions,
        "hourly_stats": hourly_stats,
        "stats_df": pd.DataFrame(hourly_stats),
        "query_stats": query_stats,
        "metadata": {
            "start": start_datetime,
            "end": end_datetime,
            "subreddits": subreddits,
            "queries": queries,
            "total": len(all_submissions),
            "crisis_name": crisis_name,
        },
    }


def filter_comment_fields(comment_data: Dict) -> Dict:
    """Filter comment to only keep essential fields."""
    return {k: v for k, v in comment_data.items() if k in COMMENT_FIELDS_TO_KEEP}


def flatten_tree(
    tree_data: Dict,
    submission_id: str,
    submission_data: Dict,
    depth: int = 0,
) -> Dict:
    """Flatten nested comment tree with metadata and field filtering."""
    flattened = {}

    if not isinstance(tree_data, dict):
        return flattened

    for cid, cinfo in tree_data.items():
        if not isinstance(cinfo, dict) or "body" not in cinfo:
            continue

        comment = filter_comment_fields(cinfo)

        comment["_submission_id"] = submission_id
        comment["_submission_title"] = submission_data.get("title", "")
        comment["_submission_score"] = submission_data.get("score", 0)
        comment["_depth"] = depth
        comment["_is_top_level"] = depth == 0

        parent = comment.get("parent_id", "")
        if parent.startswith("t1_"):
            comment["_parent_comment_id"] = parent[3:]
        else:
            comment["_parent_comment_id"] = None

        flattened[cid] = comment

        if "replies" in cinfo and cinfo["replies"]:
            nested = flatten_tree(
                cinfo["replies"], submission_id, submission_data, depth + 1
            )
            flattened.update(nested)

    return flattened


async def collect_comments(
    config: PipelineConfig,
    submissions_file: Path,
    min_comments: int = 1,
    max_submissions: Optional[int] = None,
    resume: bool = True,
) -> Dict:
    """
    Collect comments for submissions.

    Args:
        config: Pipeline configuration
        submissions_file: Path to submissions JSON file
        min_comments: Minimum comments threshold
        max_submissions: Maximum submissions to process
        resume: If True, skip posts that already have comments collected

    Returns:
        Dictionary of comment_id -> comment_data
    """
    submissions = load_json(submissions_file)
    client = create_client(config)

    posts = [
        (pid, pdata)
        for pid, pdata in submissions.items()
        if pdata.get("num_comments", 0) >= min_comments
    ]
    posts.sort(key=lambda x: x[1].get("num_comments", 0), reverse=True)

    if max_submissions:
        posts = posts[:max_submissions]

    # Resume feature: load existing comments and skip processed posts
    all_comments = {}
    processed_post_ids = set()
    comments_filename = f"comments_{submissions_file.stem}.json"
    comments_filepath = config.comments_dir / comments_filename

    if resume and comments_filepath.exists():
        existing_comments = load_json(comments_filepath)
        all_comments = existing_comments
        # Extract already processed post IDs from existing comments
        processed_post_ids = set(
            c.get("_submission_id") for c in existing_comments.values()
            if c.get("_submission_id")
        )
        # Filter out already processed posts
        original_count = len(posts)
        posts = [(pid, pdata) for pid, pdata in posts if pid not in processed_post_ids]
        skipped_count = original_count - len(posts)
        print(f"\n[Resume] Found existing comments file with {len(existing_comments):,} comments")
        print(f"[Resume] Skipping {skipped_count:,} already processed posts")
        print(f"[Resume] Remaining posts to process: {len(posts):,}")

    total_expected = sum(p[1].get("num_comments", 0) for p in posts)

    print(f"\n{'='*70}")
    print("COMMENT COLLECTION")
    print(f"{'='*70}")
    print(f"Source: {submissions_file.name}")
    print(f"Posts to process: {len(posts)}")
    print(f"Expected comments: ~{total_expected:,}")
    if processed_post_ids:
        print(f"Already collected: {len(all_comments):,} comments from {len(processed_post_ids):,} posts")
    print(f"{'='*70}\n")

    if not posts:
        print("All posts already processed. Nothing to do.")
        return all_comments

    stats = {
        "posts_processed": 0,
        "posts_with_comments": 0,
        "tree_success": 0,
        "flat_fallback": 0,
    }

    for post_id, post_data in tqdm(posts, desc="Fetching comments"):
        try:
            tree = await client.fetch(
                mode="comments_tree_search",
                link_id=f"t3_{post_id}",
                limit=25000,
                start_breadth=1000,
                start_depth=1000,
            )

            if tree:
                post_comments = flatten_tree(tree, post_id, post_data)
                all_comments.update(post_comments)
                stats["tree_success"] += 1

                if post_comments:
                    stats["posts_with_comments"] += 1

            expected = post_data.get("num_comments", 0)
            collected = len(
                [c for c in all_comments.values() if c.get(
                    "_submission_id") == post_id]
            )

            if expected > 0 and collected < expected * 0.5:
                flat_result = await client.fetch(
                    mode="comments_search",
                    link_id=f"t3_{post_id}",
                    limit=0,
                    sort="asc",
                    after=COMMENTS_DEFAULT_START,
                    before=COMMENTS_DEFAULT_END,
                )

                if flat_result:
                    stats["flat_fallback"] += 1
                    for cid, cdata in flat_result.items():
                        if cid not in all_comments:
                            filtered = filter_comment_fields(cdata)
                            filtered["_submission_id"] = post_id
                            filtered["_submission_title"] = post_data.get(
                                "title", "")
                            parent = filtered.get("parent_id", "")
                            filtered["_is_top_level"] = parent.startswith(
                                "t3_")
                            filtered["_parent_comment_id"] = (
                                parent[3:] if parent.startswith(
                                    "t1_") else None
                            )
                            all_comments[cid] = filtered

            stats["posts_processed"] += 1

            # Periodic save every 50 posts to prevent data loss
            if stats["posts_processed"] % 50 == 0:
                save_json(all_comments, comments_filepath)

        except Exception:
            continue

    # Final save
    if all_comments:
        save_json(all_comments, comments_filepath)

    print(f"\n{'='*70}")
    print("COMMENT COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Posts processed: {stats['posts_processed']}")
    print(f"Posts with comments: {stats['posts_with_comments']}")
    print(f"Total comments: {len(all_comments):,}")
    print(f"Tree search success: {stats['tree_success']}")
    print(f"Flat search fallbacks: {stats['flat_fallback']}")
    if total_expected > 0:
        rate = len(all_comments) / total_expected * 100
        print(f"Collection rate: {rate:.1f}%")
    print(f"{'='*70}")

    return all_comments


async def collect_comments_batch(
    config: PipelineConfig,
    file_pattern: str = "*_filtered.json",
    min_comments: int = 1,
    max_submissions: Optional[int] = None,
    resume: bool = True,
    skip_completed: bool = True,
) -> Dict[str, int]:
    """
    Collect comments for all matching submission files.

    Args:
        config: Pipeline configuration
        file_pattern: Glob pattern for submission files
        min_comments: Minimum comments threshold
        max_submissions: Maximum submissions per file
        resume: If True, resume from where each file left off
        skip_completed: If True, skip files that already have complete comment files

    Returns:
        Dictionary of filename -> comment count
    """
    stats = {}
    submission_files = list(config.submissions_dir.glob(file_pattern))

    if not submission_files:
        print(f"No files matching pattern: {file_pattern}")
        return stats

    # Filter out non-submission files
    submission_files = [
        f for f in submission_files
        if "comments" not in f.name and "stats" not in f.name
    ]

    # Check for already completed files (skip_completed feature)
    skipped_files = []
    if skip_completed:
        remaining_files = []
        for sub_file in submission_files:
            comments_filename = f"comments_{sub_file.stem}.json"
            comments_filepath = config.comments_dir / comments_filename
            if comments_filepath.exists():
                # Check if all posts are processed
                submissions = load_json(sub_file)
                existing_comments = load_json(comments_filepath)
                processed_ids = set(
                    c.get("_submission_id") for c in existing_comments.values()
                    if c.get("_submission_id")
                )
                posts_with_comments = [
                    pid for pid, pdata in submissions.items()
                    if pdata.get("num_comments", 0) >= min_comments
                ]
                if len(processed_ids) >= len(posts_with_comments):
                    skipped_files.append((sub_file.name, len(existing_comments)))
                    stats[sub_file.name] = len(existing_comments)
                else:
                    remaining_files.append(sub_file)
            else:
                remaining_files.append(sub_file)
        submission_files = remaining_files

    print(f"\n{'='*70}")
    print(f"BATCH COMMENT COLLECTION")
    print(f"{'='*70}")
    print(f"Files to process: {len(submission_files)}")
    if skipped_files:
        print(f"Skipped (already complete): {len(skipped_files)}")
        for fname, count in skipped_files:
            print(f"  - {fname}: {count:,} comments")
    print(f"{'='*70}\n")

    if not submission_files:
        print("All files already processed. Nothing to do.")
        return stats

    for submissions_file in submission_files:
        comments = await collect_comments(
            config=config,
            submissions_file=submissions_file,
            min_comments=min_comments,
            max_submissions=max_submissions,
            resume=resume,
        )
        stats[submissions_file.name] = len(comments)

    print(f"\n{'='*70}")
    print("BATCH COLLECTION COMPLETE")
    print(f"{'='*70}")
    for filename, count in stats.items():
        print(f"  {filename}: {count:,} comments")
    print(f"Total: {sum(stats.values()):,} comments")
    print(f"{'='*70}")

    return stats
