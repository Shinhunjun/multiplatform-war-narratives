from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import build_analysis_ready_datasets


def test_analysis_ready_helpers_cover_review_policy_and_token_parsing() -> None:
    assert build_analysis_ready_datasets.parse_token_list('["alpha", "beta"]') == ["alpha", "beta"]
    assert build_analysis_ready_datasets.parse_token_list("alpha, beta") == ["alpha", "beta"]
    assert build_analysis_ready_datasets.parse_token_list("") == []

    assert (
        build_analysis_ready_datasets.effective_filter_decision("review", "include_with_flag")
        == "keep"
    )
    assert build_analysis_ready_datasets.effective_filter_decision("review", "drop") == "drop"
    assert (
        build_analysis_ready_datasets.effective_filter_decision("review", "manual_adjudication")
        == "review"
    )
    assert build_analysis_ready_datasets.analysis_include_flag("keep") is True
    assert build_analysis_ready_datasets.analysis_include_flag("drop") is False
    assert pd.isna(build_analysis_ready_datasets.analysis_include_flag("review"))


def test_build_analysis_ready_datasets_main_writes_event_and_url_parquet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events_path = tmp_path / "gdelt_scraped.csv"
    lookup_path = tmp_path / "url_lookup.csv"
    eval_path = tmp_path / "url_filter_eval.csv"
    rules_path = tmp_path / "filter_rule_config.json"
    events_output = tmp_path / "analysis_events.parquet"
    url_output = tmp_path / "analysis_url_content.parquet"

    pd.DataFrame(
        [
            {
                "Date": 20200101,
                "Actor1Name": "Venezuela",
                "Actor2Name": "USA",
                "SourceURL": "https://example.com/story?utm_source=x",
                "Title": "Raw row title A",
                "Text": "Raw row text A",
                "Scrape_Status": "success",
                "Error_Details": "",
            },
            {
                "Date": 20200102,
                "Actor1Name": "Venezuela",
                "Actor2Name": "USA",
                "SourceURL": "https://example.com/story/",
                "Title": "Raw row title B",
                "Text": "Raw row text B",
                "Scrape_Status": "success",
                "Error_Details": "",
            },
            {
                "Date": 20200103,
                "Actor1Name": "USA",
                "Actor2Name": "Venezuela",
                "SourceURL": "other.com/path?a=1&ref=abc",
                "Title": "Raw row title C",
                "Text": "Raw row text C",
                "Scrape_Status": "success",
                "Error_Details": "",
            },
        ]
    ).to_csv(events_path, index=False)

    pd.DataFrame(
        [
            {
                "url_id": 10,
                "SourceURL": "https://example.com/story",
                "SourceURL_Canonical": "https://example.com/story",
                "Title": "Representative title A",
                "Text": "Representative text A",
                "Tokens": '["alpha", "beta"]',
                "Scrape_Status": "success",
                "row_count": 2,
                "doc_relevance_sum": 6.0,
                "doc_relevance_matches": 2,
                "doc_token_count": 2,
                "doc_relevance_score": 4.2,
            },
            {
                "url_id": 11,
                "SourceURL": "http://other.com/path?a=1",
                "SourceURL_Canonical": "http://other.com/path?a=1",
                "Title": "Representative title B",
                "Text": "Representative text B",
                "Tokens": '["venezuela", "sanction"]',
                "Scrape_Status": "success",
                "row_count": 1,
                "doc_relevance_sum": 3.0,
                "doc_relevance_matches": 2,
                "doc_token_count": 2,
                "doc_relevance_score": 2.1,
            },
        ]
    ).to_csv(lookup_path, index=False)

    pd.DataFrame(
        [
            {
                "url_id": 10,
                "in_filter_scope": True,
                "duplicate_text_hash": "hash-a",
                "duplicate_cluster_size": 1,
                "text_word_count": 120,
                "doc_relevance_score": 4.2,
                "has_ven_anchor": True,
                "has_us_primary_token": True,
                "has_us_primary_pair": False,
                "has_us_primary": True,
                "has_relation_secondary": False,
                "filter_duplicate_decision": "keep",
                "filter_length_decision": "keep",
                "filter_score_decision": "keep",
                "filter_anchor_decision": "keep",
                "filter_final_decision": "keep",
                "filter_reasons": "pass_all",
            },
            {
                "url_id": 11,
                "in_filter_scope": True,
                "duplicate_text_hash": "hash-b",
                "duplicate_cluster_size": 1,
                "text_word_count": 55,
                "doc_relevance_score": 2.1,
                "has_ven_anchor": True,
                "has_us_primary_token": False,
                "has_us_primary_pair": False,
                "has_us_primary": False,
                "has_relation_secondary": True,
                "filter_duplicate_decision": "keep",
                "filter_length_decision": "review",
                "filter_score_decision": "drop",
                "filter_anchor_decision": "review",
                "filter_final_decision": "review",
                "filter_reasons": "length_review|anchor_review",
            },
        ]
    ).to_csv(eval_path, index=False)

    rules_path.write_text(
        json.dumps(
            {
                "version": "test",
                "scope": {
                    "require_success_status": True,
                    "require_nonempty_text": True,
                    "require_nonempty_tokens": True,
                    "require_numeric_score": True,
                },
                "thresholds": {
                    "duplicate_drop_cluster_size_gt": 1,
                    "length_drop_lt": 40,
                    "length_review_lt": 80,
                    "score_drop_lt": 25,
                    "score_review_lt": 40,
                },
                "final_decision_priority": ["drop", "review", "keep"],
                "review_handling": "include_with_flag",
            }
        ),
        encoding="utf-8",
    )

    parquet_store: dict[str, pd.DataFrame] = {}

    def fake_to_parquet(self: pd.DataFrame, path: str | Path, index: bool = False) -> None:
        parquet_store[str(path)] = self.copy(deep=True)
        Path(path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_analysis_ready_datasets.py",
            "--events",
            str(events_path),
            "--lookup",
            str(lookup_path),
            "--eval",
            str(eval_path),
            "--filter-rules",
            str(rules_path),
            "--events-output",
            str(events_output),
            "--url-output",
            str(url_output),
        ],
    )

    build_analysis_ready_datasets.main()

    events_df = parquet_store[str(events_output)]
    url_df = parquet_store[str(url_output)]

    assert len(events_df) == 3
    assert len(url_df) == 2
    assert "Title" not in events_df.columns
    assert "Text" not in events_df.columns
    assert events_df["url_id"].tolist() == [10, 10, 11]
    assert events_df.loc[events_df["url_id"] == 11, "filter_final_decision"].iloc[0] == "review"
    assert events_df.loc[events_df["url_id"] == 11, "filter_final_decision_effective"].iloc[0] == "keep"
    assert bool(events_df.loc[events_df["url_id"] == 11, "analysis_include"].iloc[0]) is True

    assert url_df.loc[url_df["url_id"] == 10, "Tokens"].iloc[0] == ["alpha", "beta"]
    assert bool(url_df.loc[url_df["url_id"] == 11, "analysis_review_flag"].iloc[0]) is True
    assert url_df.loc[url_df["url_id"] == 11, "filter_final_decision_effective"].iloc[0] == "keep"
    assert bool(url_df.loc[url_df["url_id"] == 11, "analysis_include"].iloc[0]) is True
    assert events_output.exists()
    assert url_output.exists()
