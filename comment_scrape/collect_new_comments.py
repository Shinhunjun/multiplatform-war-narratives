"""
TikTok New Comments Collection - Collect Comments with Deduplication
Collects comments only for videos that don't have comments yet
"""

import os
import sys
import asyncio
import pandas as pd
import glob
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


class NewCommentCollector:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.fear_dir = os.path.join(self.data_dir, "fear_mongering")
        os.makedirs(self.fear_dir, exist_ok=True)

        # Load videos and existing comment IDs (not video IDs)
        self.df_videos = pd.read_csv(os.path.join(self.data_dir, "tiktok_videos.csv"))
        self.existing_comment_ids = self._load_existing_comment_ids()

        print(f"\n{'='*80}")
        print(f"Comment Collection Status")
        print(f"{'='*80}")
        print(f"Total videos: {len(self.df_videos):,}")
        print(f"Existing comment IDs loaded: {len(self.existing_comment_ids):,}")
        print(f"{'='*80}\n")

        self.total_comments_collected = 0
        self.videos_processed = 0
        self.duplicate_comments_skipped = 0

    def _load_existing_comment_ids(self) -> set:
        """Load existing comment IDs to prevent duplicates"""
        existing_ids = set()
        comment_files = glob.glob(os.path.join(self.fear_dir, "*", "comments.csv"))

        for file in comment_files:
            try:
                df = pd.read_csv(file)
                existing_ids.update(df['comment_id'].astype(str).unique())
            except Exception:
                pass

        return existing_ids

    def _get_videos_to_process(self, batch_size: int) -> pd.DataFrame:
        """Get videos sorted by engagement for processing"""
        df = self.df_videos.copy()

        # Sort by engagement (prioritize popular videos)
        df['engagement'] = (
            df['play_count'].fillna(0) +
            df['digg_count'].fillna(0) * 2 +
            df['comment_count'].fillna(0) * 3
        )
        df = df.sort_values('engagement', ascending=False)

        return df.head(batch_size)

    def parse_comments(self, data: dict, video_id: str, search_keyword: str) -> list:
        """Parse comments from API response, skip duplicates"""
        comments = []

        items = data.get("comments") or []
        for item in items:
            comment_id = str(item.get("cid"))

            # Skip if this comment already exists
            if comment_id in self.existing_comment_ids:
                self.duplicate_comments_skipped += 1
                continue

            user = item.get("user", {})

            comment_data = {
                "comment_id": comment_id,
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
            # Add to existing IDs to prevent duplicates within this session
            self.existing_comment_ids.add(comment_id)

        return comments

    async def collect_video_comments(self, page, video_id: str, search_keyword: str, max_comments: int = 200):
        """Collect comments for a single video using browser's fetch API"""
        all_comments = []
        cursor = 0
        count = 20

        while len(all_comments) < max_comments:
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
                break

            data = result.get("data", {})

            if data.get("status_code") != 0:
                break

            comments = self.parse_comments(data, video_id, search_keyword)
            all_comments.extend(comments)

            if not data.get("has_more", False) or len(comments) == 0:
                break

            cursor = data.get("cursor", cursor + count)
            await asyncio.sleep(0.5)

        return all_comments

    def _save_comments(self, keyword: str, comments: list):
        """Save comments to keyword folder"""
        if not comments:
            return

        # Create keyword folder
        keyword_safe = keyword.replace("/", "_").replace(" ", "_")
        keyword_dir = os.path.join(self.fear_dir, keyword_safe)
        os.makedirs(keyword_dir, exist_ok=True)

        output_file = os.path.join(keyword_dir, "comments.csv")

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

    async def run(self, batch_size: int = 50, max_comments_per_video: int = 200):
        """Run comment collection for videos needing comments"""

        print(f"Starting comment collection...")
        print(f"Batch size: {batch_size} videos")
        print(f"Max comments per video: {max_comments_per_video}")
        print(f"\n{'='*80}\n")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                viewport={"width": 1512, "height": 982},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )

            cookies = parse_cookies(COOKIE_STRING)
            await context.add_cookies(cookies)

            page = await context.new_page()

            # Go to TikTok to establish session
            await page.goto("https://www.tiktok.com/", timeout=60000)
            await page.wait_for_timeout(3000)

            # Get videos to process
            videos_to_process = self._get_videos_to_process(batch_size)

            # Process videos in batches
            for idx, (_, row) in enumerate(videos_to_process.iterrows(), 1):
                video_id = str(row['video_id'])
                keyword = row['search_keyword']
                author = row['author_unique_id']
                comment_count = int(row.get('comment_count', 0))

                print(f"[{idx}/{len(videos_to_process)}] Video: {video_id}")
                print(f"  Keyword: {keyword}")
                print(f"  Author: @{author}")
                print(f"  Expected comments: {comment_count:,}")

                try:
                    # Collect comments
                    comments = await self.collect_video_comments(
                        page, video_id, keyword, max_comments_per_video
                    )

                    if comments:
                        # Save to keyword folder
                        self._save_comments(keyword, comments)
                        self.total_comments_collected += len(comments)
                        print(f"  ✅ Collected {len(comments):,} comments")
                    else:
                        print(f"  ⚠️  No comments collected")

                    self.videos_processed += 1

                    # Rate limiting
                    await asyncio.sleep(2)

                except Exception as e:
                    print(f"  ❌ Error: {e}")

                # Save progress every 10 videos
                if idx % 10 == 0:
                    print(f"\n  Progress: {self.videos_processed} videos, {self.total_comments_collected:,} comments\n")

            await browser.close()

        # Final summary
        print(f"\n{'='*80}")
        print("COLLECTION COMPLETE")
        print(f"{'='*80}")
        print(f"Videos processed: {self.videos_processed}")
        print(f"New comments collected: {self.total_comments_collected:,}")
        print(f"Duplicate comments skipped: {self.duplicate_comments_skipped:,}")
        print(f"Average new comments per video: {self.total_comments_collected / max(self.videos_processed, 1):.1f}")
        print(f"{'='*80}")


async def main():
    collector = NewCommentCollector()

    # Collect comments for first 100 videos (you can increase this)
    await collector.run(batch_size=100, max_comments_per_video=200)


if __name__ == "__main__":
    asyncio.run(main())
