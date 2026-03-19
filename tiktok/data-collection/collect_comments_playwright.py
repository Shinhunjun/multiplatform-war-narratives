"""
Collect TikTok comments for Research API videos using Playwright browser automation.
Bypasses Research API daily quota by using browser's authenticated session.

Usage:
    python collect_comments_playwright.py
    python collect_comments_playwright.py --min-comments 5 --max-comments 200
    python collect_comments_playwright.py --resume
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright

# Import cookie config from comment_scrape
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "comment_scrape"))
from tiktok_config import COOKIE_STRING


def parse_cookies(cookie_string: str) -> list:
    """Parse cookie string into list of cookie dicts for Playwright."""
    cookies = []
    for item in cookie_string.split("; "):
        if "=" in item:
            name, value = item.split("=", 1)
            cookies.append({
                "name": name,
                "value": value,
                "domain": ".tiktok.com",
                "path": "/",
            })
    return cookies


def load_all_videos(videos_dir: Path, min_comments: int = 1) -> list:
    """Load all video records from collected JSON files, filter by comment count."""
    all_videos = []
    for f in sorted(videos_dir.glob("videos_*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for v in data:
            if v.get("comment_count", 0) >= min_comments:
                v["_source_file"] = f.name
                all_videos.append(v)

    # Deduplicate by video id
    seen = set()
    unique = []
    for v in all_videos:
        vid = str(v.get("id", ""))
        if vid and vid not in seen:
            seen.add(vid)
            unique.append(v)

    # Sort by comment_count descending (prioritize engagement)
    unique.sort(key=lambda x: x.get("comment_count", 0), reverse=True)
    return unique


def load_existing_comments(comments_file: Path) -> set:
    """Load already collected video IDs from comments file."""
    collected_video_ids = set()
    if comments_file.exists():
        with open(comments_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        collected_video_ids = set(str(c.get("video_id", "")) for c in data)
    return collected_video_ids


def parse_comments(data: dict, video_id: str, video_info: dict) -> list:
    """Parse comments from TikTok web API response."""
    comments = []
    items = data.get("comments") or []
    for item in items:
        user = item.get("user", {})
        comments.append({
            "comment_id": str(item.get("cid", "")),
            "video_id": video_id,
            "text": item.get("text", ""),
            "create_time": item.get("create_time"),
            "digg_count": item.get("digg_count", 0),
            "reply_count": item.get("reply_comment_total", 0),
            "comment_language": item.get("comment_language", ""),
            "user_id": user.get("uid"),
            "user_unique_id": user.get("unique_id"),
            "user_nickname": user.get("nickname"),
            "is_reply": False,
            "parent_comment_id": None,
            "video_username": video_info.get("username", ""),
            "video_description": video_info.get("video_description", "")[:200],
            "video_create_time": video_info.get("create_time"),
            "source_file": video_info.get("_source_file", ""),
            "collected_at": datetime.now().isoformat(),
        })
    return comments


async def collect_video_comments(page, video_id: str, max_comments: int = 200) -> list:
    """Collect comments for a single video using browser's fetch API."""
    all_comments_raw = []
    cursor = 0
    count = 20

    while len(all_comments_raw) < max_comments:
        result = await page.evaluate(f'''
            async () => {{
                const api_url = `https://www.tiktok.com/api/comment/list/?aid=1988&aweme_id={video_id}&count={count}&cursor={cursor}`;
                try {{
                    const response = await fetch(api_url, {{
                        credentials: 'include',
                        headers: {{ 'Accept': 'application/json' }}
                    }});
                    const data = await response.json();
                    return {{ success: true, data: data }};
                }} catch (e) {{
                    return {{ success: false, error: e.toString() }};
                }}
            }}
        ''')

        if not result.get("success"):
            break

        data = result.get("data", {})
        if data.get("status_code") != 0:
            break

        items = data.get("comments") or []
        all_comments_raw.append(data)

        if not data.get("has_more", False) or len(items) == 0:
            break

        cursor = data.get("cursor", cursor + count)
        await asyncio.sleep(0.5)

    return all_comments_raw


async def main():
    parser = argparse.ArgumentParser(description="Collect TikTok comments via Playwright")
    parser.add_argument("--min-comments", type=int, default=1,
                        help="Minimum comment_count to include video (default: 1)")
    parser.add_argument("--max-comments", type=int, default=200,
                        help="Max comments per video (default: 200)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Max videos to process")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already collected videos (default: True)")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    videos_dir = data_dir / "videos"
    comments_dir = data_dir / "comments"
    comments_dir.mkdir(parents=True, exist_ok=True)

    comments_file = comments_dir / "comments_playwright.json"

    # Load videos
    videos = load_all_videos(videos_dir, min_comments=args.min_comments)
    print(f"Total videos with comment_count >= {args.min_comments}: {len(videos)}")

    # Resume: skip already collected
    if args.resume:
        collected_ids = load_existing_comments(comments_file)
        videos = [v for v in videos if str(v.get("id", "")) not in collected_ids]
        print(f"Already collected: {len(collected_ids)} videos")
        print(f"Remaining: {len(videos)} videos")

    if args.max_videos:
        videos = videos[:args.max_videos]

    if not videos:
        print("No videos to process.")
        return

    # Load existing comments for appending
    all_comments = []
    if comments_file.exists():
        with open(comments_file, "r", encoding="utf-8") as f:
            all_comments = json.load(f)

    print(f"\n{'='*70}")
    print("TIKTOK COMMENT COLLECTION (Playwright)")
    print(f"{'='*70}")
    print(f"Videos to process: {len(videos)}")
    print(f"Max comments per video: {args.max_comments}")
    print(f"{'='*70}\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1512, "height": 982},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
        )
        cookies = parse_cookies(COOKIE_STRING)
        await context.add_cookies(cookies)

        page = await context.new_page()

        print("Establishing TikTok session...")
        await page.goto("https://www.tiktok.com", timeout=30000)
        await page.wait_for_timeout(3000)

        success_count = 0
        total_comments = 0
        errors = 0

        for idx, video in enumerate(videos, 1):
            video_id = str(video.get("id", ""))
            expected = video.get("comment_count", 0)

            try:
                raw_pages = await collect_video_comments(
                    page, video_id, max_comments=args.max_comments,
                )

                comments = []
                for raw in raw_pages:
                    comments.extend(parse_comments(raw, video_id, video))

                if comments:
                    # Deduplicate
                    seen_cids = set()
                    unique_comments = []
                    for c in comments:
                        cid = c["comment_id"]
                        if cid not in seen_cids:
                            seen_cids.add(cid)
                            unique_comments.append(c)

                    all_comments.extend(unique_comments)
                    total_comments += len(unique_comments)
                    success_count += 1
                    print(f"  [{idx}/{len(videos)}] {video_id}: {len(unique_comments)} comments (expected: {expected})")
                else:
                    print(f"  [{idx}/{len(videos)}] {video_id}: no comments (expected: {expected})")

            except Exception as e:
                print(f"  [{idx}/{len(videos)}] {video_id}: error - {e}")
                errors += 1

            # Save every 20 videos
            if idx % 20 == 0:
                with open(comments_file, "w", encoding="utf-8") as f:
                    json.dump(all_comments, f, ensure_ascii=False, indent=2, default=str)
                print(f"  [Checkpoint] Saved {total_comments:,} comments so far")

            await asyncio.sleep(1)

        await browser.close()

    # Final save
    with open(comments_file, "w", encoding="utf-8") as f:
        json.dump(all_comments, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*70}")
    print("COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Videos processed: {success_count + errors}")
    print(f"Videos with comments: {success_count}")
    print(f"Total comments collected: {total_comments:,}")
    print(f"Errors: {errors}")
    print(f"Output: {comments_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
