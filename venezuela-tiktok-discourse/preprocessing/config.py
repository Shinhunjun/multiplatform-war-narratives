"""
Configuration for TikTok data preprocessing.
"""

from pathlib import Path

# Input/output paths
DATA_DIR = Path(__file__).parent.parent / "data"
EXPORTS_DIR = DATA_DIR / "exports"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

# Minimum word counts
MIN_WORDS_CAPTION = 2       # TikTok captions are shorter than Reddit posts
MIN_WORDS_COMMENT = 2
MIN_WORDS_VOICE_TEXT = 5    # Voice-to-text transcriptions

# Bot/spam account patterns (TikTok-specific)
BOT_PATTERNS = [
    r"^bot_",
    r"_bot$",
    r"spam",
    r"promo\d+",
    r"follow4follow",
    r"f4f",
]

# Promotional content markers
PROMO_PATTERNS = [
    r"link in bio",
    r"check bio",
    r"use code",
    r"discount code",
    r"shop now",
    r"dm for",
    r"dm me for",
    r"follow for follow",
    r"follow me",
    r"subscribe to my",
]
