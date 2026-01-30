"""
Filtering functions for Reddit data preprocessing.
"""

from typing import Optional

from .config import BOT_ACCOUNTS, DELETED_MARKERS


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


def is_valid_submission(
    title: Optional[str],
    selftext: Optional[str],
    author: Optional[str],
    remove_deleted: bool = True,
    remove_bots: bool = True,
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

    return True


def is_valid_comment(
    body: Optional[str],
    author: Optional[str],
    remove_deleted: bool = True,
    remove_bots: bool = True,
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

    return True
