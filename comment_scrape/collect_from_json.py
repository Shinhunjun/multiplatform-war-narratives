"""
Playwright-based comment collection for TikTok videos stored in JSON files.
No API quota — uses browser session with cookies.

Usage:
    python comment_scrape/collect_from_json.py
"""

import os
import sys
import json
import asyncio
import glob
from datetime import datetime
from playwright.async_api import async_playwright

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tiktok_config import COOKIE_STRING

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "..", "tiktok", "data-collection", "data", "videos")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tiktok", "data-collection", "data", "comments_playwright")
MIN_COMMENTS = 5


def parse_cookies(cookie_string: str) -> list:
    cookies = []
    for item in cookie_string.split("; "):
        if "=" in item:
            name, value = item.split("=", 1)
            cookies.append({"name": name, "value": value, "domain": ".tiktok.com", "path": "/"})
    return cookies


def load_eligible_videos():
    """Load all videos with >= MIN_COMMENTS, skip already collected."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load already collected video IDs
    collected_ids = set()
    output_file = os.path.join(OUTPUT_DIR, "comments_all.json")
    if os.path.exists(output_file):
        existing = json.load(open(output_file))
        collected_ids = set(str(c.get("video_id", "")) for c in existing)
        print(f"Already collected comments for {len(collected_ids)} videos")

    videos = []
    for f in sorted(glob.glob(os.path.join(VIDEOS_DIR, "videos_*.json"))):
        data = json.load(open(f))
        for v in data:
            vid = str(v.get("id", ""))
            cc = v.get("comment_count", 0)
            if cc >= MIN_COMMENTS and vid not in collected_ids:
                videos.append({
                    "video_id": vid,
                    "comment_count": cc,
                    "username": v.get("username", ""),
                    "source_file": os.path.basename(f),
                })

    # Sort by comment count descending (high engagement first)
    videos.sort(key=lambda x: x["comment_count"], reverse=True)
    return videos


async def collect_comments(page, video_id: str, max_comments: int = 500):
    """Collect comments for one video using browser fetch API."""
    all_comments = []
    cursor = 0
    count = 50

    while len(all_comments) < max_comments:
        result = await page.evaluate(f'''
            async () => {{
                try {{
                    const r = await fetch(
                        `https://www.tiktok.com/api/comment/list/?aid=1988&aweme_id={video_id}&count={count}&cursor={cursor}`,
                        {{ credentials: 'include', headers: {{ 'Accept': 'application/json' }} }}
                    );
                    return {{ success: true, data: await r.json() }};
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
        for item in items:
            user = item.get("user", {})
            all_comments.append({
                "comment_id": str(item.get("cid", "")),
                "video_id": video_id,
                "text": item.get("text", ""),
                "create_time": item.get("create_time"),
                "like_count": item.get("digg_count", 0),
                "reply_count": item.get("reply_comment_total", 0),
                "user_id": user.get("uid"),
                "user_nickname": user.get("nickname"),
            })

        if not data.get("has_more", False) or len(items) == 0:
            break

        cursor = data.get("cursor", cursor + count)
        await asyncio.sleep(0.3)

    return all_comments


async def main():
    videos = load_eligible_videos()
    print(f"\nEligible videos to process: {len(videos):,}")
    if not videos:
        print("Nothing to collect!")
        return

    # Load existing comments
    output_file = os.path.join(OUTPUT_DIR, "comments_all.json")
    all_comments = []
    if os.path.exists(output_file):
        all_comments = json.load(open(output_file))

    print(f"Existing comments: {len(all_comments):,}")
    print(f"Starting Playwright collection...\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1512, "height": 982},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        await context.add_cookies(parse_cookies(COOKIE_STRING))

        page = await context.new_page()
        await page.goto("https://www.tiktok.com/", timeout=60000)
        await page.wait_for_timeout(3000)

        collected = 0
        errors = 0

        for i, v in enumerate(videos):
            vid = v["video_id"]
            expected = v["comment_count"]

            try:
                comments = await collect_comments(page, vid)
                if comments:
                    all_comments.extend(comments)
                    collected += len(comments)
                    print(f"[{i+1}/{len(videos)}] {vid}: {len(comments)} comments (expected {expected})")
                else:
                    print(f"[{i+1}/{len(videos)}] {vid}: 0 comments")
            except Exception as e:
                errors += 1
                print(f"[{i+1}/{len(videos)}] {vid}: ERROR {e}")

            # Save every 50 videos
            if (i + 1) % 50 == 0:
                with open(output_file, "w") as f:
                    json.dump(all_comments, f, ensure_ascii=False, default=str)
                print(f"  --- Saved {len(all_comments):,} total comments ---")

            await asyncio.sleep(0.5)

        await browser.close()

    # Final save
    with open(output_file, "w") as f:
        json.dump(all_comments, f, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"DONE: {collected:,} new comments from {len(videos)} videos")
    print(f"Total comments: {len(all_comments):,}")
    print(f"Errors: {errors}")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
