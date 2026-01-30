"""
Text cleaning functions for Reddit data preprocessing.
"""

import re
from typing import Optional


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    # Remove http/https URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove www URLs
    text = re.sub(r"www\.\S+", "", text)
    return text


def remove_markdown_links(text: str) -> str:
    """Convert markdown links [text](url) to just text."""
    # Replace [text](url) with text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def remove_reddit_formatting(text: str) -> str:
    """Remove Reddit-specific markdown formatting."""
    # Remove bold **text** or __text__
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)

    # Remove italic *text* or _text_
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)

    # Remove strikethrough ~~text~~
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # Remove superscript ^text or ^(text)
    text = re.sub(r"\^+\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\^+(\S+)", r"\1", text)

    # Remove spoiler tags >!text!<
    text = re.sub(r">!([^!]+)!<", r"\1", text)

    # Remove code blocks ```text``` or `text`
    text = re.sub(r"```[^`]*```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove quote markers (lines starting with >)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)

    # Remove header markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    return text


def remove_edit_markers(text: str, remove_edit_content: bool = False) -> str:
    """
    Remove or clean edit markers from text.

    Args:
        text: Input text
        remove_edit_content: If True, remove everything after "Edit:"
                           If False, just remove the "Edit:" marker
    """
    if remove_edit_content:
        # Remove "Edit:" and everything after it
        patterns = [
            r"\bEdit\s*\d*\s*:.*$",
            r"\bEDIT\s*\d*\s*:.*$",
            r"\bedit\s*\d*\s*:.*$",
            r"\bUpdate\s*\d*\s*:.*$",
            r"\bUPDATE\s*\d*\s*:.*$",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)
    else:
        # Just remove the edit marker, keep the content
        text = re.sub(r"\b[Ee][Dd][Ii][Tt]\s*\d*\s*:", "", text)
        text = re.sub(r"\b[Uu][Pp][Dd][Aa][Tt][Ee]\s*\d*\s*:", "", text)

    return text


def remove_emojis(text: str) -> str:
    """Remove emoji characters from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f700-\U0001f77f"  # alchemical symbols
        "\U0001f780-\U0001f7ff"  # Geometric Shapes
        "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols
        "\U0001fa00-\U0001fa6f"  # Chess Symbols
        "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027b0"  # Dingbats
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def normalize_accents(text: str) -> str:
    """Normalize Spanish/Portuguese accented characters."""
    import unicodedata

    # Normalize to NFD form (decomposed)
    text = unicodedata.normalize("NFD", text)
    # Remove combining diacritical marks
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def clean_text(
    text: Optional[str],
    remove_urls_flag: bool = True,
    remove_markdown_flag: bool = True,
    remove_edit_markers_flag: bool = True,
    remove_emojis_flag: bool = False,
    normalize_accents_flag: bool = False,
    lowercase: bool = False,
) -> str:
    """
    Apply full text cleaning pipeline.

    Args:
        text: Input text to clean
        remove_urls_flag: Remove URLs
        remove_markdown_flag: Remove markdown formatting
        remove_edit_markers_flag: Remove edit markers
        remove_emojis_flag: Remove emojis
        normalize_accents_flag: Normalize accented characters
        lowercase: Convert to lowercase

    Returns:
        Cleaned text
    """
    if text is None or not isinstance(text, str):
        return ""

    # Apply cleaning steps in order
    if remove_urls_flag:
        text = remove_urls(text)

    if remove_markdown_flag:
        text = remove_markdown_links(text)
        text = remove_reddit_formatting(text)

    if remove_edit_markers_flag:
        text = remove_edit_markers(text, remove_edit_content=False)

    if remove_emojis_flag:
        text = remove_emojis(text)

    if normalize_accents_flag:
        text = normalize_accents(text)

    # Always normalize whitespace
    text = normalize_whitespace(text)

    if lowercase:
        text = text.lower()

    return text
