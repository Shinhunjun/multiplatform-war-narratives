"""
Data processing functions: JSON loading, merging, and Parquet export.
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import PipelineConfig


def load_json(filepath: Path) -> list | dict:
    """Load data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_video_files(videos_dir: Path, pattern: str = "videos_*.json") -> pd.DataFrame:
    """
    Load and merge all video JSON files into a single DataFrame.

    Args:
        videos_dir: Directory containing video JSON files.
        pattern: Glob pattern for video files.

    Returns:
        DataFrame with all videos, deduplicated by id.
    """
    all_videos = []
    files = sorted(videos_dir.glob(pattern))

    for f in files:
        data = load_json(f)
        if isinstance(data, list):
            all_videos.extend(data)
        elif isinstance(data, dict):
            all_videos.extend(data.values())

    if not all_videos:
        print("No video data found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_videos)

    # Deduplicate by video id
    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["id"], keep="first")
        after = len(df)
        if before != after:
            print(f"Deduplicated: {before:,} -> {after:,} videos ({before - after:,} duplicates removed)")

    # Convert create_time to datetime
    if "create_time" in df.columns:
        df["created_at"] = pd.to_datetime(df["create_time"], unit="s", errors="coerce")

    print(f"Loaded {len(df):,} videos from {len(files)} files")
    return df


def merge_comment_files(comments_dir: Path, pattern: str = "comments_*.json") -> pd.DataFrame:
    """
    Load and merge all comment JSON files into a single DataFrame.

    Args:
        comments_dir: Directory containing comment JSON files.
        pattern: Glob pattern for comment files.

    Returns:
        DataFrame with all comments.
    """
    all_comments = []
    files = sorted(comments_dir.glob(pattern))

    for f in files:
        data = load_json(f)
        if isinstance(data, list):
            all_comments.extend(data)
        elif isinstance(data, dict):
            all_comments.extend(data.values())

    if not all_comments:
        print("No comment data found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_comments)

    # Deduplicate by comment id
    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["id"], keep="first")
        after = len(df)
        if before != after:
            print(f"Deduplicated: {before:,} -> {after:,} comments")

    # Convert create_time to datetime
    if "create_time" in df.columns:
        df["created_at"] = pd.to_datetime(df["create_time"], unit="s", errors="coerce")

    print(f"Loaded {len(df):,} comments from {len(files)} files")
    return df


def export_to_parquet(config: PipelineConfig) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Export all collected data to Parquet format.

    Args:
        config: Pipeline configuration.

    Returns:
        Tuple of (videos_df, comments_df).
    """
    config.exports_dir.mkdir(parents=True, exist_ok=True)

    # Export videos
    videos_df = merge_video_files(config.videos_dir)
    if len(videos_df) > 0:
        videos_output = config.exports_dir / "all_videos.parquet"
        videos_df.to_parquet(videos_output, index=False)
        print(f"Videos exported: {videos_output} ({len(videos_df):,} records)")
    else:
        videos_df = None

    # Export comments
    comments_df = merge_comment_files(config.comments_dir)
    if len(comments_df) > 0:
        comments_output = config.exports_dir / "all_comments.parquet"
        comments_df.to_parquet(comments_output, index=False)
        print(f"Comments exported: {comments_output} ({len(comments_df):,} records)")
    else:
        comments_df = None

    return videos_df, comments_df
