"""
Filtering functions for Reddit data preprocessing.
"""

import re
from typing import Optional

from .config import BOT_ACCOUNTS, DELETED_MARKERS


# Patterns for noise content
URL_HEAVY_PATTERN = re.compile(
    r'^https?://|'
    r'^\[.*\]\(https?://|'
    r'^!\[.*\]\(|'
    r'(https?://\S+\s*){2,}',  # Multiple URLs
    re.IGNORECASE
)

# Media link domains to filter
MEDIA_DOMAINS = [
    'imgur.com', 'i.imgur.com', 'i.redd.it', 'v.redd.it',
    'giphy.com', 'gfycat.com', 'streamable.com',
    'youtube.com', 'youtu.be', 'twitter.com', 'x.com',
    'preview.redd.it', 'external-preview.redd.it',
]

# Moderation and meta content patterns
MODERATION_PATTERNS = [
    r'this (post|comment|submission) has been removed',
    r'your (post|comment|submission) has been removed',
    r'removed.*rule',
    r'rule \d+',
    r'violat(es?|ing|ion)',
    r'banned from',
    r'moderator(s)?',
    r'automod(erator)?',
    r'please read (the|our) rules',
    r'flair your post',
    r'this is a reminder',
    r'i am a bot',
    r'beep boop',
    r'\*i am a bot\*',
    r'bot action',
    r'this action was performed automatically',
]
MODERATION_REGEX = re.compile('|'.join(MODERATION_PATTERNS), re.IGNORECASE)

# Low-value content patterns
LOW_VALUE_PATTERNS = [
    r'^(lol|lmao|haha|jaja|xd|ok|yes|no|this|same|true|nice|thanks|agreed)\.?$',
    r'^\^this\.?$',
    r'^r/\w+$',  # Just a subreddit link
    r'^u/\w+$',  # Just a user mention
    r'^\[deleted\]$',
    r'^\[removed\]$',
]
LOW_VALUE_REGEX = re.compile('|'.join(LOW_VALUE_PATTERNS), re.IGNORECASE)


def is_deleted_content(text: Optional[str]) -> bool:
    """Check if text is deleted or removed content."""
    if text is None:
        return True
    if not isinstance(text, str):
        return False
    text_lower = text.strip().lower()
    return any(marker.lower() in text_lower for marker in DELETED_MARKERS)


def is_deleted_author(author: Optional[str]) -> bool:
    """Check if author is deleted."""
    if author is None:
        return True
    if not isinstance(author, str):
        return False
    return author.strip().lower() in ["[deleted]", "deleted", ""]


def is_bot_account(author: Optional[str]) -> bool:
    """Check if author is a known bot account."""
    if author is None or not isinstance(author, str):
        return False
    author_lower = author.strip().lower()
    return any(bot.lower() == author_lower for bot in BOT_ACCOUNTS)


def is_too_short(text: Optional[str], min_words: int = 5) -> bool:
    """Check if text is too short (fewer than min_words)."""
    if text is None or not isinstance(text, str):
        return True
    words = text.split()
    return len(words) < min_words


def is_too_long(text: Optional[str], max_chars: int = 10000) -> bool:
    """Check if text exceeds maximum character limit."""
    if text is None or not isinstance(text, str):
        return False
    return len(text) > max_chars


def is_mostly_url(text: Optional[str], threshold: float = 0.7) -> bool:
    """Check if text is mostly URLs (low content value)."""
    if text is None or not isinstance(text, str):
        return False

    text = text.strip()
    if not text:
        return True

    # Check for media domains
    text_lower = text.lower()
    for domain in MEDIA_DOMAINS:
        if domain in text_lower:
            # If the text is short and contains media domain, likely just a link
            if len(text) < 200:
                return True

    # Remove URLs and check remaining content
    url_pattern = r'https?://\S+'
    text_without_urls = re.sub(url_pattern, '', text).strip()

    if not text_without_urls:
        return True

    # Check ratio of non-URL content
    if len(text_without_urls) / len(text) < (1 - threshold):
        return True

    return False


def is_moderation_content(text: Optional[str]) -> bool:
    """Check if text is moderation-related (rule violations, bot messages, etc.)."""
    if text is None or not isinstance(text, str):
        return False

    return bool(MODERATION_REGEX.search(text))


def is_low_value_content(text: Optional[str]) -> bool:
    """Check if text is low-value (very short responses, just links, etc.)."""
    if text is None or not isinstance(text, str):
        return False

    text = text.strip()

    # Check against low-value patterns
    if LOW_VALUE_REGEX.match(text):
        return True

    return False


def has_meaningful_content(text: Optional[str], min_alpha_ratio: float = 0.5) -> bool:
    """Check if text has meaningful alphabetic content (not just symbols/numbers)."""
    if text is None or not isinstance(text, str):
        return False

    text = text.strip()
    if not text:
        return False

    # Count alphabetic characters
    alpha_chars = sum(1 for c in text if c.isalpha())

    if len(text) == 0:
        return False

    return (alpha_chars / len(text)) >= min_alpha_ratio


def is_valid_submission(
    title: Optional[str],
    selftext: Optional[str],
    author: Optional[str],
    remove_deleted: bool = True,
    remove_bots: bool = True,
    remove_media_posts: bool = True,
    remove_moderation: bool = True,
    min_words: int = 5,
) -> bool:
    """
    Check if a submission is valid for analysis.

    Returns True if submission should be kept.
    """
    # Check author
    if remove_deleted and is_deleted_author(author):
        return False
    if remove_bots and is_bot_account(author):
        return False

    # Check title
    if is_deleted_content(title):
        return False

    # Selftext can be empty for link posts, so only check if deleted
    if selftext and is_deleted_content(selftext):
        return False

    # Check minimum length (title only, selftext can be empty)
    if is_too_short(title, min_words=2):  # Titles can be shorter
        return False

    # NEW: Filter media-only posts
    if remove_media_posts:
        combined = (title or '') + ' ' + (selftext or '')
        if is_mostly_url(combined):
            return False

    # NEW: Filter moderation content
    if remove_moderation:
        combined = (title or '') + ' ' + (selftext or '')
        if is_moderation_content(combined):
            return False

    return True


def is_valid_comment(
    body: Optional[str],
    author: Optional[str],
    remove_deleted: bool = True,
    remove_bots: bool = True,
    remove_media_posts: bool = True,
    remove_moderation: bool = True,
    remove_low_value: bool = True,
    min_words: int = 5,
) -> bool:
    """
    Check if a comment is valid for analysis.

    Returns True if comment should be kept.
    """
    # Check author
    if remove_deleted and is_deleted_author(author):
        return False
    if remove_bots and is_bot_account(author):
        return False

    # Check body content
    if is_deleted_content(body):
        return False

    # Check minimum length
    if is_too_short(body, min_words=min_words):
        return False

    # NEW: Filter media-only comments
    if remove_media_posts and is_mostly_url(body):
        return False

    # NEW: Filter moderation content
    if remove_moderation and is_moderation_content(body):
        return False

    # NEW: Filter low-value content
    if remove_low_value and is_low_value_content(body):
        return False

    # NEW: Check for meaningful content
    if not has_meaningful_content(body, min_alpha_ratio=0.3):
        return False

    return True
