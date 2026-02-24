"""
Data collection functions for TikTok videos and comments.
Uses TikTok Research API via official SDK.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .config import (
    ALL_HASHTAGS,
    COMMENT_FIELDS,
    CRISIS_QUERIES,
    FLASHPOINTS,
    PRIORITY_HASHTAGS,
    PRIORITY_QUERIES,
    PipelineConfig,
    VIDEO_FIELDS,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def save_json(data: Any, filepath: Path) -> None:
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved {len(data) if isinstance(data, (list, dict)) else '?'} records to {filepath.name}")


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_30day_windows(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """
    Generate 30-day time windows (TikTok API max range per query).

    Args:
        start_date: Start date in YYYYMMDD format.
        end_date: End date in YYYYMMDD format.

    Returns:
        List of (start, end) tuples in YYYYMMDD format.
    """
    windows = []
    current = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    while current <= end:
        window_end = min(current + timedelta(days=29), end)
        windows.append((
            current.strftime("%Y%m%d"),
            window_end.strftime("%Y%m%d"),
        ))
        current = window_end + timedelta(days=1)

    return windows


# =============================================================================
# QUOTA TRACKER
# =============================================================================


class QuotaTracker:
    """Track daily API quota usage and manage checkpoints."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.daily_requests = 0
        self.daily_records = 0
        self.max_requests = config.daily_request_limit
        self.max_records = config.daily_record_limit
        self._date = datetime.utcnow().date()

    def _check_date_reset(self) -> None:
        """Reset counters if a new day (UTC)."""
        today = datetime.utcnow().date()
        if today != self._date:
            print(f"\n[Quota] New day detected. Resetting counters.")
            self.daily_requests = 0
            self.daily_records = 0
            self._date = today

    def can_request(self, expected_records: int = 100) -> bool:
        """Check if we can make another request within daily limits."""
        self._check_date_reset()
        return (
            self.daily_requests < self.max_requests
            and self.daily_records + expected_records <= self.max_records
        )

    def record_usage(self, num_records: int) -> None:
        """Record API usage."""
        self.daily_requests += 1
        self.daily_records += num_records

    def wait_for_reset(self) -> None:
        """Wait until daily quota resets (midnight UTC)."""
        now = datetime.utcnow()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=10, microsecond=0
        )
        wait_seconds = (tomorrow - now).total_seconds()
        print(f"\n[Quota] Daily limit reached. Requests: {self.daily_requests}, Records: {self.daily_records}")
        print(f"[Quota] Waiting {wait_seconds/3600:.1f} hours until UTC midnight reset...")
        time.sleep(wait_seconds)
        self.daily_requests = 0
        self.daily_records = 0
        self._date = datetime.utcnow().date()
        print("[Quota] Reset complete. Resuming collection.")

    def save_checkpoint(self, checkpoint_data: Dict, name: str) -> None:
        """Save checkpoint for resume."""
        checkpoint_path = self.config.checkpoints_dir / f"{name}.json"
        checkpoint_data["_quota"] = {
            "daily_requests": self.daily_requests,
            "daily_records": self.daily_records,
            "date": str(self._date),
            "timestamp": datetime.utcnow().isoformat(),
        }
        save_json(checkpoint_data, checkpoint_path)

    def load_checkpoint(self, name: str) -> Optional[Dict]:
        """Load checkpoint if exists."""
        checkpoint_path = self.config.checkpoints_dir / f"{name}.json"
        if checkpoint_path.exists():
            return load_json(checkpoint_path)
        return None


# =============================================================================
# VIDEO COLLECTION
# =============================================================================


def build_keyword_query(keywords: List[str]):
    """Build a TikTok API query for keyword search."""
    from tiktok_research_api import Criteria, Query

    criteria = Criteria(
        operation="IN",
        field_name="keyword",
        field_values=keywords,
    )
    return Query(and_criteria=[criteria])


def build_hashtag_query(hashtags: List[str]):
    """Build a TikTok API query for hashtag search."""
    from tiktok_research_api import Criteria, Query

    criteria = Criteria(
        operation="IN",
        field_name="hashtag_name",
        field_values=hashtags,
    )
    return Query(and_criteria=[criteria])


def fetch_videos_for_query(
    api,
    query,
    start_date: str,
    end_date: str,
    max_count: int = 100,
    max_total: int = 1000000,
) -> List[Dict]:
    """
    Fetch videos for a single query within a date range.
    The SDK automatically handles 30-day chunking and pagination.

    Returns:
        List of video dictionaries.
    """
    from tiktok_research_api import QueryVideoRequest

    request = QueryVideoRequest(
        query=query,
        start_date=start_date,
        end_date=end_date,
        max_count=max_count,
        max_total=max_total,
        fields=VIDEO_FIELDS,
    )

    try:
        # SDK returns tuple: (videos, search_id, cursor, has_more, start_str, end_str)
        result = api.query_videos(request, fetch_all_pages=True)
        videos = result[0] if isinstance(result, tuple) else result
        return videos if videos else []
    except Exception as e:
        print(f"  Error fetching videos: {e}")
        return []


def collect_videos_historical(
    config: PipelineConfig,
    api,
    start_date: str,
    end_date: str,
    keywords: Optional[List[str]] = None,
    hashtags: Optional[List[str]] = None,
    resume: bool = True,
) -> Dict[str, int]:
    """
    Collect historical video data with keyword and hashtag queries.

    The SDK handles 30-day windowing internally, but we split into
    monthly chunks ourselves for better checkpointing and progress tracking.

    Args:
        config: Pipeline configuration.
        api: Authenticated TikTok Research API client.
        start_date: Start date YYYYMMDD.
        end_date: End date YYYYMMDD.
        keywords: Search keywords (defaults to PRIORITY_QUERIES).
        hashtags: Search hashtags (defaults to PRIORITY_HASHTAGS).
        resume: If True, skip windows that already have data.

    Returns:
        Dictionary of window_key -> video count.
    """
    if keywords is None:
        keywords = PRIORITY_QUERIES
    if hashtags is None:
        hashtags = PRIORITY_HASHTAGS

    windows = generate_30day_windows(start_date, end_date)
    quota = QuotaTracker(config)

    # Load checkpoint for resume
    checkpoint_name = f"historical_{start_date}_{end_date}"
    checkpoint = quota.load_checkpoint(checkpoint_name) if resume else None
    completed_windows = set()
    if checkpoint:
        completed_windows = set(checkpoint.get("completed_windows", []))
        print(f"[Resume] Found checkpoint with {len(completed_windows)} completed windows")

    print(f"\n{'='*70}")
    print("HISTORICAL VIDEO COLLECTION")
    print(f"{'='*70}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Windows: {len(windows)} (30-day chunks)")
    print(f"Keywords: {keywords[:5]}{'...' if len(keywords) > 5 else ''}")
    print(f"Hashtags: {hashtags[:5]}{'...' if len(hashtags) > 5 else ''}")
    if completed_windows:
        print(f"Skipping: {len(completed_windows)} already completed windows")
    print(f"{'='*70}\n")

    stats = {}
    all_video_ids = set()  # Global dedup

    for window_start, window_end in tqdm(windows, desc="Collecting videos"):
        window_key = f"{window_start}_{window_end}"

        # Skip completed windows
        if window_key in completed_windows:
            # Load existing data to count
            existing_file = config.videos_dir / f"videos_{window_key}.json"
            if existing_file.exists():
                existing = load_json(existing_file)
                stats[window_key] = len(existing)
                all_video_ids.update(v.get("id") for v in existing if v.get("id"))
            continue

        # Check quota
        if not quota.can_request():
            # Save checkpoint before waiting
            quota.save_checkpoint({
                "completed_windows": list(completed_windows),
                "total_videos": sum(stats.values()),
            }, checkpoint_name)
            quota.wait_for_reset()

        window_videos = {}

        # Keyword query
        if keywords:
            keyword_query = build_keyword_query(keywords)
            videos = fetch_videos_for_query(
                api, keyword_query, window_start, window_end,
                max_count=config.max_count,
            )
            for v in videos:
                vid = str(v.get("id", ""))
                if vid and vid not in all_video_ids:
                    v["_matched_type"] = "keyword"
                    v["_window_start"] = window_start
                    v["_window_end"] = window_end
                    v["_collection_type"] = "historical"
                    window_videos[vid] = v
                    all_video_ids.add(vid)
            quota.record_usage(len(videos))

        # Hashtag query
        if hashtags and quota.can_request():
            hashtag_query = build_hashtag_query(hashtags)
            videos = fetch_videos_for_query(
                api, hashtag_query, window_start, window_end,
                max_count=config.max_count,
            )
            for v in videos:
                vid = str(v.get("id", ""))
                if vid and vid not in all_video_ids:
                    v["_matched_type"] = "hashtag"
                    v["_window_start"] = window_start
                    v["_window_end"] = window_end
                    v["_collection_type"] = "historical"
                    window_videos[vid] = v
                    all_video_ids.add(vid)
                elif vid in window_videos:
                    # Already found by keyword, mark as both
                    window_videos[vid]["_matched_type"] = "keyword+hashtag"
            quota.record_usage(len(videos))

        # Save window data
        if window_videos:
            video_list = list(window_videos.values())
            save_json(video_list, config.videos_dir / f"videos_{window_key}.json")
            stats[window_key] = len(video_list)
        else:
            stats[window_key] = 0

        completed_windows.add(window_key)

        # Periodic checkpoint save
        if len(completed_windows) % 5 == 0:
            quota.save_checkpoint({
                "completed_windows": list(completed_windows),
                "total_videos": sum(stats.values()),
            }, checkpoint_name)

    # Final checkpoint
    quota.save_checkpoint({
        "completed_windows": list(completed_windows),
        "total_videos": sum(stats.values()),
        "status": "complete",
    }, checkpoint_name)

    total = sum(stats.values())
    print(f"\n{'='*70}")
    print(f"Collection Complete: {total:,} total videos")
    print(f"Windows with data: {sum(1 for v in stats.values() if v > 0)} / {len(windows)}")
    print(f"{'='*70}")

    return stats


def collect_videos_crisis(
    config: PipelineConfig,
    api,
    crisis_key: Optional[str] = None,
    collect_all: bool = False,
    keywords: Optional[List[str]] = None,
    hashtags: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Collect videos for crisis/flashpoint periods.

    Args:
        config: Pipeline configuration.
        api: Authenticated TikTok Research API client.
        crisis_key: Specific flashpoint key, or None for all.
        collect_all: If True, collect all flashpoints.
        keywords: Search keywords (defaults to CRISIS_QUERIES).
        hashtags: Search hashtags (defaults to ALL_HASHTAGS).

    Returns:
        Dictionary of crisis_name -> video count.
    """
    if keywords is None:
        keywords = CRISIS_QUERIES
    if hashtags is None:
        hashtags = ALL_HASHTAGS

    flashpoints = FLASHPOINTS
    if crisis_key and not collect_all:
        if crisis_key not in flashpoints:
            print(f"Error: Unknown crisis '{crisis_key}'")
            print(f"Available: {', '.join(flashpoints.keys())}")
            return {}
        flashpoints = {crisis_key: flashpoints[crisis_key]}

    quota = QuotaTracker(config)
    stats = {}

    for fp_key, fp_info in flashpoints.items():
        print(f"\n{'='*70}")
        print(f"CRISIS COLLECTION: {fp_info['name']}")
        print(f"Period: {fp_info['start']} to {fp_info['end']}")
        print(f"{'='*70}")

        if not quota.can_request():
            quota.wait_for_reset()

        all_videos = {}

        # Keyword search
        if keywords:
            keyword_query = build_keyword_query(keywords)
            videos = fetch_videos_for_query(
                api, keyword_query, fp_info["start"], fp_info["end"],
                max_count=100,
            )
            for v in videos:
                vid = str(v.get("id", ""))
                if vid:
                    v["_matched_type"] = "keyword"
                    v["_crisis_name"] = fp_info["name"]
                    v["_collection_type"] = "crisis"
                    all_videos[vid] = v
            quota.record_usage(len(videos))

        # Hashtag search
        if hashtags and quota.can_request():
            hashtag_query = build_hashtag_query(hashtags)
            videos = fetch_videos_for_query(
                api, hashtag_query, fp_info["start"], fp_info["end"],
                max_count=100,
            )
            for v in videos:
                vid = str(v.get("id", ""))
                if vid and vid not in all_videos:
                    v["_matched_type"] = "hashtag"
                    v["_crisis_name"] = fp_info["name"]
                    v["_collection_type"] = "crisis"
                    all_videos[vid] = v
                elif vid in all_videos:
                    all_videos[vid]["_matched_type"] = "keyword+hashtag"
            quota.record_usage(len(videos))

        if all_videos:
            video_list = list(all_videos.values())
            filename = f"crisis_{fp_key}_{fp_info['start']}_{fp_info['end']}.json"
            save_json(video_list, config.videos_dir / filename)

        stats[fp_info["name"]] = len(all_videos)
        print(f"  Collected: {len(all_videos):,} videos")

    print(f"\n{'='*70}")
    print("CRISIS COLLECTION COMPLETE")
    print(f"{'='*70}")
    for name, count in stats.items():
        print(f"  {name}: {count:,}")
    print(f"Total: {sum(stats.values()):,}")
    print(f"{'='*70}")

    return stats


# =============================================================================
# COMMENT COLLECTION
# =============================================================================


def collect_comments(
    config: PipelineConfig,
    api,
    videos_file: Optional[Path] = None,
    min_comments: int = 5,
    max_videos: Optional[int] = None,
    resume: bool = True,
) -> Dict[str, int]:
    """
    Collect comments for videos with high engagement.

    Args:
        config: Pipeline configuration.
        api: Authenticated TikTok Research API client.
        videos_file: Specific videos JSON file, or None for all.
        min_comments: Minimum comment_count threshold.
        max_videos: Maximum videos to process.
        resume: If True, skip already processed videos.

    Returns:
        Dictionary of filename -> comment count.
    """
    from tiktok_research_api import QueryVideoCommentsRequest

    # Find video files
    if videos_file:
        video_files = [videos_file]
    else:
        video_files = sorted(config.videos_dir.glob("*.json"))
        video_files = [f for f in video_files if "comments" not in f.name]

    if not video_files:
        print("No video files found.")
        return {}

    quota = QuotaTracker(config)
    file_stats = {}

    for vf in video_files:
        videos = load_json(vf)
        if not isinstance(videos, list):
            videos = list(videos.values()) if isinstance(videos, dict) else []

        # Filter by min_comments
        eligible = [
            v for v in videos
            if v.get("comment_count", 0) >= min_comments
        ]
        eligible.sort(key=lambda x: x.get("comment_count", 0), reverse=True)

        if max_videos:
            eligible = eligible[:max_videos]

        # Resume: load existing comments
        comments_filename = f"comments_{vf.stem}.json"
        comments_filepath = config.comments_dir / comments_filename
        all_comments = []
        processed_ids = set()

        if resume and comments_filepath.exists():
            all_comments = load_json(comments_filepath)
            processed_ids = set(
                str(c.get("video_id", "")) for c in all_comments
            )
            original_count = len(eligible)
            eligible = [v for v in eligible if str(v.get("id", "")) not in processed_ids]
            print(f"[Resume] {comments_filepath.name}: {original_count - len(eligible)} already processed, {len(eligible)} remaining")

        if not eligible:
            print(f"  {vf.name}: No videos to process (all done or none eligible)")
            file_stats[vf.name] = len(all_comments)
            continue

        print(f"\n{'='*70}")
        print(f"COMMENT COLLECTION: {vf.name}")
        print(f"{'='*70}")
        print(f"Eligible videos: {len(eligible)}")
        print(f"Expected comments: ~{sum(v.get('comment_count', 0) for v in eligible):,}")
        if processed_ids:
            print(f"Already collected: {len(all_comments):,} comments")
        print(f"{'='*70}\n")

        for video in tqdm(eligible, desc=f"Comments for {vf.stem}"):
            video_id = video.get("id")
            if not video_id:
                continue

            if not quota.can_request():
                # Save progress before waiting
                save_json(all_comments, comments_filepath)
                quota.wait_for_reset()

            try:
                request = QueryVideoCommentsRequest(
                    video_id=video_id,
                    max_count=100,
                    fields=COMMENT_FIELDS,
                )
                comments, cursor, has_more = api.query_video_comments(
                    request, fetch_all_pages=True
                )
                quota.record_usage(len(comments) if comments else 0)

                if comments:
                    for c in comments:
                        c["_source_file"] = vf.name
                        c["_video_username"] = video.get("username", "")
                    all_comments.extend(comments)

            except Exception as e:
                print(f"  Error for video {video_id}: {e}")
                continue

            # Periodic save every 50 videos
            if len(processed_ids) % 50 == 0 and all_comments:
                save_json(all_comments, comments_filepath)

            processed_ids.add(str(video_id))

        # Final save
        if all_comments:
            save_json(all_comments, comments_filepath)

        file_stats[vf.name] = len(all_comments)

    print(f"\n{'='*70}")
    print("COMMENT COLLECTION COMPLETE")
    print(f"{'='*70}")
    for filename, count in file_stats.items():
        print(f"  {filename}: {count:,} comments")
    print(f"Total: {sum(file_stats.values()):,}")
    print(f"{'='*70}")

    return file_stats
