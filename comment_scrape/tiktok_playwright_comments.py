"""
TikTok Playwright Comments - Comment Collection with Browser Automation

Collects comments by using browser's fetch API with valid session.
Bypasses X-Bogus limitation by executing fetch from authenticated browser context.

Usage:
    python src/tiktok_playwright_comments.py [--max-videos N] [--keyword KEYWORD]
    python src/tiktok_playwright_comments.py --data-dir control_group  # For control group data
"""

import os
import sys
import json
import asyncio
import argparse
import pandas as pd
from datetime import datetime
from playwright.async_api import async_playwright

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tiktok_config import COOKIE_STRING


def parse_cookies(cookie_string: str) -> list:
    """Parse cookie string into list of cookie dicts for Playwright"""
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


class TikTokCommentCollector:
    def __init__(self, data_subdir: str = "fear_mongering"):
        """
        Args:
            data_subdir: Subdirectory name under data/ (e.g., "by_keyword" or "comparison")
        """
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.by_keyword_dir = os.path.join(self.data_dir, data_subdir)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.by_keyword_dir, exist_ok=True)

    def parse_comments(self, data: dict, video_id: str, search_keyword: str) -> list:
        """Parse comments from API response"""
        comments = []

        items = data.get("comments") or []
        for item in items:
            user = item.get("user", {})

            comment_data = {
                "comment_id": item.get("cid"),
                "video_id": video_id,
                "search_keyword": search_keyword,
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
                "collected_at": datetime.now().isoformat(),
            }

            comments.append(comment_data)

        return comments

    async def collect_video_comments(self, page, video_id: str, search_keyword: str, max_comments: int = 50):
        """
        Collect comments for a single video using browser's fetch API

        Args:
            page: Playwright page object (must be on a TikTok page)
            video_id: TikTok video ID
            search_keyword: Search keyword the video was found with
            max_comments: Maximum comments to collect per video
        """
        all_comments = []
        cursor = 0
        count = 20

        while len(all_comments) < max_comments:
            # Use browser's fetch API
            result = await page.evaluate(f'''
                async () => {{
                    const api_url = `https://www.tiktok.com/api/comment/list/?aid=1988&aweme_id={video_id}&count={count}&cursor={cursor}`;

                    try {{
                        const response = await fetch(api_url, {{
                            credentials: 'include',
                            headers: {{
                                'Accept': 'application/json'
                            }}
                        }});
                        const data = await response.json();
                        return {{
                            success: true,
                            data: data
                        }};
                    }} catch (e) {{
                        return {{ success: false, error: e.toString() }};
                    }}
                }}
            ''')

            if not result.get("success"):
                print(f"    Error: {result.get('error')}")
                break

            data = result.get("data", {})

            if data.get("status_code") != 0:
                print(f"    API error: {data.get('status_msg', 'Unknown')}")
                break

            comments = self.parse_comments(data, video_id, search_keyword)
            all_comments.extend(comments)

            # Check if more comments available
            if not data.get("has_more", False) or len(comments) == 0:
                break

            cursor = data.get("cursor", cursor + count)

            # Small delay between requests
            await asyncio.sleep(0.5)

        return all_comments

    def _get_keyword_folders(self) -> list:
        """Get list of keyword folders"""
        folders = []
        if os.path.exists(self.by_keyword_dir):
            for name in os.listdir(self.by_keyword_dir):
                folder_path = os.path.join(self.by_keyword_dir, name)
                if os.path.isdir(folder_path):
                    videos_file = os.path.join(folder_path, "videos.csv")
                    if os.path.exists(videos_file):
                        folders.append(name)
        return sorted(folders)

    def _get_collected_video_ids(self, keyword_folder: str) -> set:
        """Get video IDs already collected for a specific keyword"""
        collected = set()
        comments_file = os.path.join(self.by_keyword_dir, keyword_folder, "comments.csv")
        if os.path.exists(comments_file):
            try:
                df = pd.read_csv(comments_file)
                collected.update(df["video_id"].astype(str).unique())
            except Exception:
                pass
        return collected

    def _save_comments(self, keyword_folder: str, comments: list):
        """Save comments to keyword folder"""
        if not comments:
            return

        output_file = os.path.join(self.by_keyword_dir, keyword_folder, "comments.csv")

        df = pd.DataFrame(comments)
        df = df.drop_duplicates(subset=["comment_id"], keep="first")

        # Append to existing
        if os.path.exists(output_file):
            try:
                df_existing = pd.read_csv(output_file)
                if not df_existing.empty:
                    df = pd.concat([df_existing, df], ignore_index=True)
                    df = df.drop_duplicates(subset=["comment_id"], keep="first")
            except Exception:
                pass

        df.to_csv(output_file, index=False, encoding="utf-8")

    async def collect_keyword_comments(self, keyword_folder: str, page, max_videos: int = None, max_comments_per_video: int = 50):
        """Collect comments for all videos in a keyword folder"""
        keyword_dir = os.path.join(self.by_keyword_dir, keyword_folder)
        videos_file = os.path.join(keyword_dir, "videos.csv")

        if not os.path.exists(videos_file):
            print(f"  No videos.csv found in {keyword_folder}")
            return 0

        df_videos = pd.read_csv(videos_file)
        collected_video_ids = self._get_collected_video_ids(keyword_folder)

        # Filter out already collected
        df_videos = df_videos[~df_videos["video_id"].astype(str).isin(collected_video_ids)]

        if max_videos:
            df_videos = df_videos.head(max_videos)

        if len(df_videos) == 0:
            print(f"  All videos already collected for {keyword_folder}")
            return 0

        print(f"  Processing {len(df_videos)} videos (skipping {len(collected_video_ids)} already done)")

        all_comments = []
        success_count = 0

        for idx, row in df_videos.iterrows():
            video_id = str(row["video_id"])
            search_keyword = str(row.get("search_keyword", keyword_folder.replace("_", " ")))

            try:
                comments = await self.collect_video_comments(page, video_id, search_keyword, max_comments_per_video)

                if comments:
                    all_comments.extend(comments)
                    print(f"    [{idx + 1}] {video_id}: {len(comments)} comments")
                    success_count += 1
                else:
                    print(f"    [{idx + 1}] {video_id}: No comments")

                # Save progress periodically
                if len(all_comments) >= 500:
                    self._save_comments(keyword_folder, all_comments)
                    all_comments = []
                    print(f"    [Progress saved]")

            except Exception as e:
                print(f"    [{idx + 1}] {video_id}: Error - {e}")

            # Rate limiting
            await asyncio.sleep(1)

        # Final save
        if all_comments:
            self._save_comments(keyword_folder, all_comments)

        return success_count

    async def collect_all_comments(self, keywords: list = None, max_videos_per_keyword: int = None, max_comments_per_video: int = 50):
        """
        Collect comments for all keywords

        Args:
            keywords: List of keyword folders to process (None = all)
            max_videos_per_keyword: Max videos per keyword
            max_comments_per_video: Max comments per video
        """
        print("=" * 60)
        print("TikTok Comment Collection (by Keyword)")
        print("=" * 60)

        # Get keyword folders
        all_keywords = self._get_keyword_folders()

        if keywords:
            # Filter to specified keywords
            all_keywords = [k for k in all_keywords if k in keywords]

        print(f"Keywords to process: {len(all_keywords)}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()

            # Add cookies
            cookies = parse_cookies(COOKIE_STRING)
            await context.add_cookies(cookies)

            page = await context.new_page()

            # Navigate to TikTok to establish session
            print("Establishing session...")
            await page.goto("https://www.tiktok.com", timeout=30000)
            await page.wait_for_timeout(3000)

            total_success = 0

            for keyword_folder in all_keywords:
                print(f"\n[Keyword: {keyword_folder}]")
                success = await self.collect_keyword_comments(
                    keyword_folder, page, max_videos_per_keyword, max_comments_per_video
                )
                total_success += success

            await browser.close()

        print("\n" + "=" * 60)
        print(f"Collection complete! Total videos with comments: {total_success}")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="TikTok Comment Collection")
    parser.add_argument("--max-videos", type=int, default=None, help="Max videos per keyword")
    parser.add_argument("--max-comments", type=int, default=50, help="Max comments per video")
    parser.add_argument("--keyword", type=str, default=None, help="Specific keyword folder to process")
    parser.add_argument("--data-dir", type=str, default="fear_mongering",
                        help="Data subdirectory: 'fear_mongering' (default) or 'control_group'")
    args = parser.parse_args()

    collector = TikTokCommentCollector(data_subdir=args.data_dir)

    keywords = [args.keyword] if args.keyword else None

    await collector.collect_all_comments(
        keywords=keywords,
        max_videos_per_keyword=args.max_videos,
        max_comments_per_video=args.max_comments
    )


if __name__ == "__main__":
    asyncio.run(main())
