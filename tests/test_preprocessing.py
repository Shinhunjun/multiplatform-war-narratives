"""Unit tests for Reddit preprocessing: filters and text cleaning."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reddit.preprocessing.filters import (
    is_bot_account,
    is_deleted_content,
    is_deleted_author,
    is_too_short,
    is_too_long,
    is_mostly_url,
    is_moderation_content,
    is_low_value_content,
    has_meaningful_content,
    is_valid_comment,
    is_valid_submission,
)
from reddit.preprocessing.text_cleaner import (
    remove_urls,
    remove_markdown_links,
    remove_reddit_formatting,
    remove_edit_markers,
    normalize_whitespace,
    clean_text,
)


# ── Filter tests ────────────────────────────────────────────────────────────


class TestBotDetection:
    """Tests for bot account detection."""

    def test_known_bots_detected(self):
        assert is_bot_account("AutoModerator") is True
        assert is_bot_account("autotldr") is True
        assert is_bot_account("RemindMeBot") is True

    def test_case_insensitive(self):
        assert is_bot_account("automoderator") is True
        assert is_bot_account("AUTOMODERATOR") is True

    def test_regular_users_pass(self):
        assert is_bot_account("regular_user") is False
        assert is_bot_account("venezuela_news") is False

    def test_none_input(self):
        assert is_bot_account(None) is False

    def test_empty_string(self):
        assert is_bot_account("") is False


class TestDeletedContent:
    """Tests for deleted/removed content detection."""

    def test_deleted_markers(self):
        assert is_deleted_content("[deleted]") is True
        assert is_deleted_content("[removed]") is True

    def test_normal_text_passes(self):
        assert is_deleted_content("Venezuela is in crisis") is False

    def test_none_input(self):
        assert is_deleted_content(None) is True

    def test_deleted_author(self):
        assert is_deleted_author("[deleted]") is True
        assert is_deleted_author(None) is True
        assert is_deleted_author("real_user") is False


class TestLengthFilters:
    """Tests for text length validation."""

    def test_too_short(self):
        assert is_too_short("hi") is True
        assert is_too_short("one two three four") is True  # 4 words < 5
        assert is_too_short("one two three four five") is False  # 5 words

    def test_too_short_none(self):
        assert is_too_short(None) is True

    def test_too_long(self):
        assert is_too_long("x" * 10001) is True
        assert is_too_long("short text") is False

    def test_custom_min_words(self):
        assert is_too_short("hello world", min_words=3) is True
        assert is_too_short("hello world", min_words=2) is False


class TestURLDetection:
    """Tests for URL-heavy content detection."""

    def test_url_only(self):
        assert is_mostly_url("https://example.com/article") is True

    def test_text_with_url(self):
        text = "Check this analysis of Venezuela's crisis and the impact on regional stability in Latin America"
        assert is_mostly_url(text) is False

    def test_media_domain(self):
        assert is_mostly_url("https://imgur.com/abc123") is True
        assert is_mostly_url("https://i.redd.it/photo.jpg") is True

    def test_none_input(self):
        assert is_mostly_url(None) is False


class TestModerationContent:
    """Tests for moderation message detection."""

    def test_removal_messages(self):
        assert is_moderation_content("This post has been removed for violating rule 3") is True
        assert is_moderation_content("I am a bot, and this action was performed automatically") is True

    def test_normal_political_text(self):
        assert is_moderation_content("Maduro's government imposed new sanctions") is False

    def test_none_input(self):
        assert is_moderation_content(None) is False


class TestLowValueContent:
    """Tests for low-value content detection."""

    def test_low_value_responses(self):
        assert is_low_value_content("lol") is True
        assert is_low_value_content("this") is True
        assert is_low_value_content("r/politics") is True

    def test_substantive_text(self):
        assert is_low_value_content("The situation in Venezuela has deteriorated significantly") is False


class TestMeaningfulContent:
    """Tests for meaningful content validation."""

    def test_numeric_only(self):
        assert has_meaningful_content("12345 67890") is False

    def test_normal_text(self):
        assert has_meaningful_content("Venezuela crisis analysis") is True

    def test_empty(self):
        assert has_meaningful_content("") is False
        assert has_meaningful_content(None) is False


class TestValidComment:
    """Integration tests for full comment validation."""

    def test_valid_comment(self):
        assert is_valid_comment(
            body="The economic situation in Venezuela continues to worsen under Maduro",
            author="political_analyst",
        ) is True

    def test_bot_comment_rejected(self):
        assert is_valid_comment(
            body="Here is a summary of the article for you",
            author="AutoModerator",
        ) is False

    def test_deleted_comment_rejected(self):
        assert is_valid_comment(body="[deleted]", author="someone") is False

    def test_short_comment_rejected(self):
        assert is_valid_comment(body="yes", author="user") is False


class TestValidSubmission:
    """Integration tests for full submission validation."""

    def test_valid_submission(self):
        assert is_valid_submission(
            title="Breaking: New sanctions imposed on Venezuela",
            selftext="The United States government announced new economic sanctions targeting Venezuelan officials.",
            author="news_reporter",
        ) is True

    def test_deleted_submission_rejected(self):
        assert is_valid_submission(
            title="[deleted]", selftext="", author="someone"
        ) is False

    def test_bot_submission_rejected(self):
        assert is_valid_submission(
            title="Auto summary", selftext="Summary text here", author="autotldr"
        ) is False


# ── Text cleaner tests ──────────────────────────────────────────────────────


class TestRemoveURLs:
    """Tests for URL removal."""

    def test_http_url(self):
        result = remove_urls("Check https://example.com for details")
        assert "https://example.com" not in result
        assert "Check" in result

    def test_multiple_urls(self):
        result = remove_urls("See https://a.com and https://b.com")
        assert "https://" not in result

    def test_no_urls(self):
        text = "Venezuela faces economic crisis"
        assert remove_urls(text) == text


class TestRemoveMarkdown:
    """Tests for markdown formatting removal."""

    def test_bold(self):
        assert "Maduro" in remove_reddit_formatting("**Maduro** is president")
        assert "**" not in remove_reddit_formatting("**Maduro** is president")

    def test_italic(self):
        result = remove_reddit_formatting("*crisis* in Venezuela")
        assert "*" not in result
        assert "crisis" in result

    def test_code_blocks(self):
        result = remove_reddit_formatting("Use `python` for analysis")
        assert "`" not in result

    def test_quote_markers(self):
        result = remove_reddit_formatting("> This is a quote\nNormal text")
        assert result.strip().startswith("This is a quote")

    def test_markdown_links(self):
        result = remove_markdown_links("[article](https://example.com)")
        assert result == "article"
        assert "https://" not in result


class TestEditMarkers:
    """Tests for edit marker removal."""

    def test_edit_marker(self):
        result = remove_edit_markers("Original text Edit: added more info")
        assert "Edit:" not in result and "edit:" not in result
        assert "Original text" in result

    def test_no_edit(self):
        text = "Normal text without edits"
        assert remove_edit_markers(text) == text


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_multiple_spaces(self):
        assert normalize_whitespace("too   many    spaces") == "too many spaces"

    def test_multiple_newlines(self):
        result = normalize_whitespace("line1\n\n\n\n\nline2")
        assert "\n\n\n" not in result

    def test_leading_trailing(self):
        assert normalize_whitespace("  padded  ") == "padded"


class TestCleanTextPipeline:
    """Integration tests for the full cleaning pipeline."""

    def test_full_pipeline(self):
        dirty = "Check **this** https://example.com for [details](https://link.com) Edit: updated"
        result = clean_text(dirty)
        assert "https://" not in result
        assert "**" not in result
        assert "Edit:" not in result

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_non_string_input(self):
        assert clean_text(12345) == ""

    def test_preserves_content(self):
        text = "Maduro imposed sanctions on opposition leaders in Venezuela"
        result = clean_text(text)
        assert "Maduro" in result
        assert "Venezuela" in result
