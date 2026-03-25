"""
Filtering functions for TikTok data preprocessing.
Adapted from Reddit preprocessing pipeline with TikTok-specific additions.
"""

import re
from typing import Optional

from .config import BOT_PATTERNS, PROMO_PATTERNS

# Compile regex patterns
BOT_REGEX = re.compile("|".join(BOT_PATTERNS), re.IGNORECASE)
PROMO_REGEX = re.compile("|".join(PROMO_PATTERNS), re.IGNORECASE)

# Low-value content patterns (adapted for TikTok)
LOW_VALUE_PATTERNS = [
    r"^(lol|lmao|haha|jaja|xd|ok|yes|no|this|same|true|nice|thanks|agreed)\.?$",
    r"^#\w+(\s+#\w+)*$",           # Only hashtags, no actual text
    r"^@\w+(\s+@\w+)*$",           # Only mentions
    r"^(follow|like|share|comment)$",
    r"^\W+$",                        # Only non-word characters (emojis only)
]
LOW_VALUE_REGEX = re.compile("|".join(LOW_VALUE_PATTERNS), re.IGNORECASE)

# URL pattern
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


def is_bot_account(username: str) -> bool:
    """Check if username matches bot patterns."""
    if not username:
        return False
    return bool(BOT_REGEX.search(username))


def is_promotional(text: str) -> bool:
    """Check if text is primarily promotional content."""
    if not text:
        return False
    return bool(PROMO_REGEX.search(text))


def is_low_value(text: str) -> bool:
    """Check if text is low-value content."""
    if not text:
        return True
    cleaned = text.strip()
    return bool(LOW_VALUE_REGEX.match(cleaned))


def has_meaningful_content(text: str, min_words: int = 2) -> bool:
    """
    Check if text has meaningful content.

    Args:
        text: Input text.
        min_words: Minimum word count.

    Returns:
        True if text has meaningful content.
    """
    if not text:
        return False

    # Remove URLs and hashtags for word counting
    cleaned = URL_PATTERN.sub("", text)
    cleaned = re.sub(r"#\w+", "", cleaned)
    cleaned = re.sub(r"@\w+", "", cleaned)
    words = cleaned.split()

    if len(words) < min_words:
        return False

    # Check for minimum alphabetic content (30%)
    alpha_chars = sum(1 for c in cleaned if c.isalpha())
    total_chars = len(cleaned.strip())
    if total_chars > 0 and alpha_chars / total_chars < 0.3:
        return False

    return True


def clean_text(text: str) -> str:
    """
    Clean TikTok text content.

    Args:
        text: Raw text.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    # Remove URLs
    text = URL_PATTERN.sub("", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def filter_video(video: dict, min_words: int = 2) -> Optional[dict]:
    """
    Apply all filters to a video record.

    Args:
        video: Video dictionary.
        min_words: Minimum words in description.

    Returns:
        Filtered video dict, or None if filtered out.
    """
    username = video.get("username", "")
    description = video.get("video_description", "")
    voice_text = video.get("voice_to_text", "")

    # Filter bot accounts
    if is_bot_account(username):
        return None

    # Filter promotional content
    if is_promotional(description):
        return None

    # Filter low-value
    if is_low_value(description) and not has_meaningful_content(voice_text, 5):
        return None

    # Check meaningful content in either description or voice_to_text
    if not has_meaningful_content(description, min_words) and not has_meaningful_content(voice_text, 5):
        return None

    # Clean text
    video = video.copy()
    video["video_description_clean"] = clean_text(description)
    video["voice_to_text_clean"] = clean_text(voice_text) if voice_text else ""

    return video


def filter_comment(comment: dict, min_words: int = 2) -> Optional[dict]:
    """
    Apply all filters to a comment record.

    Args:
        comment: Comment dictionary.
        min_words: Minimum words.

    Returns:
        Filtered comment dict, or None if filtered out.
    """
    text = comment.get("text", "")

    if is_low_value(text):
        return None

    if not has_meaningful_content(text, min_words):
        return None

    comment = comment.copy()
    comment["text_clean"] = clean_text(text)

    return comment
