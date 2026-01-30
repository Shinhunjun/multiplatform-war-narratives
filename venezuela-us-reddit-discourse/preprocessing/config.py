"""
Configuration for Reddit data preprocessing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""

    # Data paths
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent
        / "data-collection"
        / "data"
    )

    @property
    def submissions_dir(self) -> Path:
        return self.data_dir / "submissions"

    @property
    def comments_dir(self) -> Path:
        return self.data_dir / "comments"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "preprocessed"

    # Filtering options
    remove_deleted: bool = True
    remove_bots: bool = True
    min_word_count: int = 5
    max_char_count: int = 10000

    # Text cleaning options
    remove_urls: bool = True
    remove_markdown: bool = True
    remove_edit_markers: bool = True
    lowercase: bool = False  # Keep original case for analysis
    remove_emojis: bool = False  # Keep for sentiment analysis

    # Language options
    normalize_accents: bool = False  # Keep Spanish accents
    detect_language: bool = False  # Skip language detection for speed

    # Output options
    output_format: str = "parquet"  # parquet, csv, json

    def ensure_output_dir(self) -> "PreprocessConfig":
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self


# Bot accounts to filter
BOT_ACCOUNTS: List[str] = [
    "AutoModerator",
    "autotldr",
    "empleadoEstatalBot",
    "RemindMeBot",
    "TweetPoster",
    "imguralbumbot",
    "sneakpeekbot",
    "RepostSleuthBot",
    "WikiTextBot",
    "HelperBot_",
    "GoodBot_BadBot",
    "timezone_bot",
    "WikiSummarizerBot",
    "stabbot",
    "totesmessenger",
    "TotesMessenger",
    "sub_doesnt_exist_bot",
    "SnapshillBot",
    "B0tRank",
    "Flair_Helper",
    "MAGIC_EYE_BOT",
    "SaveVideo",
    "savevideo",
    "downloadvideo",
    "VredditDownloader",
]

# Deleted/removed content markers
DELETED_MARKERS: List[str] = [
    "[deleted]",
    "[removed]",
    "[deleted by user]",
]
