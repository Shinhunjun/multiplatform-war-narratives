"""
Main preprocessor for TikTok data.
Loads Parquet exports, applies filters, and saves clean datasets.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .config import EXPORTS_DIR, MIN_WORDS_CAPTION, MIN_WORDS_COMMENT, PREPROCESSED_DIR
from .filters import filter_comment, filter_video


def preprocess_videos(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Preprocess video data: apply filters, clean text.

    Args:
        input_path: Path to videos Parquet file.
        output_path: Path to save clean Parquet file.

    Returns:
        Cleaned DataFrame.
    """
    input_path = input_path or EXPORTS_DIR / "all_videos.parquet"
    output_path = output_path or PREPROCESSED_DIR / "videos_clean.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading videos from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Raw videos: {len(df):,}")

    # Apply filters
    clean_records = []
    filtered_counts = {"bot": 0, "promo": 0, "low_value": 0, "no_content": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering videos"):
        result = filter_video(row.to_dict(), min_words=MIN_WORDS_CAPTION)
        if result:
            clean_records.append(result)

    clean_df = pd.DataFrame(clean_records)
    filtered = len(df) - len(clean_df)

    print(f"\nPreprocessing Results:")
    print(f"  Raw: {len(df):,}")
    print(f"  Clean: {len(clean_df):,}")
    print(f"  Filtered: {filtered:,} ({filtered/len(df)*100:.1f}%)")
    print(f"  Retention: {len(clean_df)/len(df)*100:.1f}%")

    clean_df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return clean_df


def preprocess_comments(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Preprocess comment data: apply filters, clean text.

    Args:
        input_path: Path to comments Parquet file.
        output_path: Path to save clean Parquet file.

    Returns:
        Cleaned DataFrame.
    """
    input_path = input_path or EXPORTS_DIR / "all_comments.parquet"
    output_path = output_path or PREPROCESSED_DIR / "comments_clean.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading comments from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Raw comments: {len(df):,}")

    clean_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering comments"):
        result = filter_comment(row.to_dict(), min_words=MIN_WORDS_COMMENT)
        if result:
            clean_records.append(result)

    clean_df = pd.DataFrame(clean_records)
    filtered = len(df) - len(clean_df)

    print(f"\nPreprocessing Results:")
    print(f"  Raw: {len(df):,}")
    print(f"  Clean: {len(clean_df):,}")
    print(f"  Filtered: {filtered:,} ({filtered/len(df)*100:.1f}%)")
    print(f"  Retention: {len(clean_df)/len(df)*100:.1f}%")

    clean_df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return clean_df


def run_all():
    """Run full preprocessing pipeline."""
    print("=" * 70)
    print("TIKTOK DATA PREPROCESSING")
    print("=" * 70)

    videos_df = preprocess_videos()
    print()
    comments_df = preprocess_comments()

    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"  Videos: {len(videos_df):,}")
    print(f"  Comments: {len(comments_df):,}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_all()
