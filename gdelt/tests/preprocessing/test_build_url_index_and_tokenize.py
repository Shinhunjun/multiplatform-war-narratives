from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import build_url_index
import tokenize_url_lookup


def test_canonicalize_url_normalizes_and_strips_tracking() -> None:
    assert (
        build_url_index.canonicalize_url(
            "HTTPS://Example.COM:443/path/?b=2&utm_source=x&a=1&ref=abc#frag"
        )
        == "https://example.com/path?a=1&b=2"
    )
    assert build_url_index.canonicalize_url("example.com/path/") == "http://example.com/path"
    assert build_url_index.canonicalize_url(None) == ""


def test_choose_representative_rows_prefers_content_rich_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "SourceURL_Canonical": "http://example.com/a",
                "SourceURL": "http://example.com/a?x=1",
                "Title": "Detailed title",
                "Text": "",
                "Scrape_Status": "success",
            },
            {
                "SourceURL_Canonical": "http://example.com/a",
                "SourceURL": "http://example.com/a?x=2",
                "Title": "",
                "Text": "More informative body text",
                "Scrape_Status": "failed",
            },
            {
                "SourceURL_Canonical": "http://example.com/b",
                "SourceURL": "http://example.com/b",
                "Title": "Only row",
                "Text": "Unique body",
                "Scrape_Status": "success",
            },
        ]
    )

    chosen = build_url_index.choose_representative_rows(frame)

    assert len(chosen) == 2
    first = chosen.loc[chosen["SourceURL_Canonical"] == "http://example.com/a"].iloc[0]
    assert first["SourceURL"] == "http://example.com/a?x=2"
    assert int(first["row_count"]) == 2


def test_load_existing_lookup_handles_missing_and_invalid_files(tmp_path: Path) -> None:
    missing = build_url_index.load_existing_lookup(tmp_path / "missing.csv")
    assert list(missing.columns) == [
        "url_id",
        "SourceURL",
        "SourceURL_Canonical",
        "Title",
        "Text",
        "Tokens",
        "Scrape_Status",
        "row_count",
    ]

    bad_path = tmp_path / "bad_lookup.csv"
    pd.DataFrame([{"url_id": 1, "Title": "missing canonical"}]).to_csv(bad_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        build_url_index.load_existing_lookup(bad_path)


def test_upsert_lookup_updates_metadata_and_preserves_existing_tokens() -> None:
    existing = pd.DataFrame(
        [
            {
                "url_id": 10,
                "SourceURL": "http://example.com/original",
                "SourceURL_Canonical": "http://example.com/original",
                "Title": "Old title",
                "Text": "Old text",
                "Tokens": '["kept"]',
                "Scrape_Status": "success",
                "row_count": 1,
            }
        ]
    )
    incoming = pd.DataFrame(
        [
            {
                "url_id": 10,
                "SourceURL": "http://example.com/original?ref=1",
                "SourceURL_Canonical": "http://example.com/original",
                "Title": "New title",
                "Text": "New text",
                "Tokens": "",
                "Scrape_Status": "updated",
                "row_count": 2,
            },
            {
                "url_id": 11,
                "SourceURL": "http://example.com/new",
                "SourceURL_Canonical": "http://example.com/new",
                "Title": "Brand new",
                "Text": "Brand new text",
                "Tokens": '["fresh"]',
                "Scrape_Status": "success",
                "row_count": 1,
            },
        ]
    )

    out = build_url_index.upsert_lookup(existing, incoming)

    kept = out.loc[out["url_id"] == 10].iloc[0]
    inserted = out.loc[out["url_id"] == 11].iloc[0]
    assert kept["Title"] == "New title"
    assert kept["Text"] == "New text"
    assert kept["Tokens"] == '["kept"]'
    assert int(kept["row_count"]) == 2
    assert inserted["Tokens"] == '["fresh"]'


def test_build_url_index_main_assigns_stable_ids_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "gdelt_scraped.csv"
    lookup_path = tmp_path / "url_lookup.csv"
    output_path = tmp_path / "gdelt_with_ids.csv"

    pd.DataFrame(
        [
            {
                "SourceURL": "https://Example.com/news?utm_source=x",
                "Title": "",
                "Text": "Body text from the representative row.",
                "Scrape_Status": "success",
            },
            {
                "SourceURL": "https://example.com/news/",
                "Title": "Alternate title",
                "Text": "",
                "Scrape_Status": "success",
            },
            {
                "SourceURL": "other.com/story/?b=2&a=1&ref=abc",
                "Title": "Second URL",
                "Text": "Another body",
                "Scrape_Status": "success",
            },
        ]
    ).to_csv(input_path, index=False)

    pd.DataFrame(
        [
            {
                "url_id": 10,
                "SourceURL": "https://example.com/news",
                "SourceURL_Canonical": "https://example.com/news",
                "Title": "Old title",
                "Text": "Old text",
                "Tokens": '["kept"]',
                "Scrape_Status": "success",
                "row_count": 1,
            }
        ]
    ).to_csv(lookup_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_url_index.py",
            "--input",
            str(input_path),
            "--lookup",
            str(lookup_path),
            "--output",
            str(output_path),
        ],
    )

    build_url_index.main()

    lookup_df = pd.read_csv(lookup_path, low_memory=False)
    output_df = pd.read_csv(output_path, low_memory=False)

    assert lookup_df["url_id"].tolist() == [10, 11]
    assert lookup_df.loc[lookup_df["url_id"] == 10, "Tokens"].item() == '["kept"]'
    assert lookup_df.loc[lookup_df["url_id"] == 10, "row_count"].item() == 2
    assert (
        lookup_df.loc[lookup_df["url_id"] == 11, "SourceURL_Canonical"].item()
        == "http://other.com/story?a=1&b=2"
    )
    assert "SourceURL_Canonical" not in output_df.columns
    assert output_df["url_id"].tolist() == [10, 10, 11]


@pytest.mark.parametrize("value", [None, "", "   ", pd.NA, float("nan")])
def test_tokenize_url_lookup_is_blank_recognizes_empty_values(value: object) -> None:
    assert tokenize_url_lookup.is_blank(value) is True


def test_tokenize_url_lookup_main_only_fills_blank_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_path = tmp_path / "url_lookup.csv"
    output_path = tmp_path / "tokenized.csv"
    pd.DataFrame(
        [
            {"Text": "Alpha beta", "Tokens": ""},
            {"Text": "Gamma", "Tokens": '["existing"]'},
            {"Text": "", "Tokens": ""},
        ]
    ).to_csv(lookup_path, index=False)

    def fake_tokenize(text: str, lemmatizer: object, stopword_set: set[str]) -> set[str]:
        return {token.lower() for token in text.split() if token.lower() not in stopword_set}

    monkeypatch.setattr(tokenize_url_lookup, "build_stopword_set", lambda: {"beta"})
    monkeypatch.setattr(tokenize_url_lookup, "tokenize", fake_tokenize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tokenize_url_lookup.py",
            "--lookup",
            str(lookup_path),
            "--output",
            str(output_path),
        ],
    )

    tokenize_url_lookup.main()

    out = pd.read_csv(output_path, low_memory=False).fillna("")
    assert out.loc[0, "Tokens"] == '["alpha"]'
    assert out.loc[1, "Tokens"] == '["existing"]'
    assert out.loc[2, "Tokens"] == ""


def test_tokenize_url_lookup_main_force_retokenizes_existing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_path = tmp_path / "url_lookup.csv"
    output_path = tmp_path / "tokenized_force.csv"
    pd.DataFrame([{"Text": "Alpha beta", "Tokens": '["stale"]'}]).to_csv(lookup_path, index=False)

    monkeypatch.setattr(tokenize_url_lookup, "build_stopword_set", lambda: set())
    monkeypatch.setattr(
        tokenize_url_lookup,
        "tokenize",
        lambda text, lemmatizer, stopword_set: {token.lower() for token in text.split()},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tokenize_url_lookup.py",
            "--lookup",
            str(lookup_path),
            "--output",
            str(output_path),
            "--force",
        ],
    )

    tokenize_url_lookup.main()

    out = pd.read_csv(output_path, low_memory=False)
    assert out.loc[0, "Tokens"] == '["alpha", "beta"]'
