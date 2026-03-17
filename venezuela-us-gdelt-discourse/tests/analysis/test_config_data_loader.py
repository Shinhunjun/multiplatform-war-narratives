from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.config import AnalysisConfig
from analysis.data_loader import (
    _extract_domain,
    filter_by_period,
    get_time_periods,
    load_gdelt_events,
    load_relevance_tokens,
    load_relevant_terms,
    load_url_lookup,
    sample_from_ids,
)


def build_analysis_event_rows() -> list[dict[str, object]]:
    return [
        {
            "Date": 20200101,
            "Actor1Name": "Venezuela Gov",
            "Actor1CountryCode": "VEN",
            "Actor2Name": "US Gov",
            "Actor2CountryCode": "USA",
            "EventCode": 120,
            "QuadClass": 1,
            "GoldsteinScale": 3.0,
            "AvgTone": 0.4,
            "SourceURL": "https://www.example.com/a",
            "Scrape_Status": "Success",
            "Error_Details": "",
            "url_id": 1,
            "doc_relevance_score": 80.0,
            "suspect_redirect_content": False,
            "analysis_include": True,
            "row_count": 2,
        },
        {
            "Date": 20200102,
            "Actor1Name": "US Gov",
            "Actor1CountryCode": "USA",
            "Actor2Name": "Venezuela Gov",
            "Actor2CountryCode": "VEN",
            "EventCode": 190,
            "QuadClass": 4,
            "GoldsteinScale": -5.0,
            "AvgTone": -0.5,
            "SourceURL": "https://news.test/b",
            "Scrape_Status": "Error",
            "Error_Details": "failed",
            "url_id": 2,
            "doc_relevance_score": 10.0,
            "suspect_redirect_content": False,
            "analysis_include": True,
            "row_count": 1,
        },
        {
            "Date": 20200103,
            "Actor1Name": "Regional",
            "Actor1CountryCode": "COL",
            "Actor2Name": "Venezuela Gov",
            "Actor2CountryCode": "VEN",
            "EventCode": 40,
            "QuadClass": 2,
            "GoldsteinScale": 1.0,
            "AvgTone": 0.1,
            "SourceURL": "https://empty.test/c",
            "Scrape_Status": "Success",
            "Error_Details": "",
            "url_id": 3,
            "doc_relevance_score": 90.0,
            "suspect_redirect_content": False,
            "analysis_include": False,
            "row_count": 1,
        },
        {
            "Date": 20200104,
            "Actor1Name": "Venezuela Gov",
            "Actor1CountryCode": "VEN",
            "Actor2Name": "US Gov",
            "Actor2CountryCode": "USA",
            "EventCode": 120,
            "QuadClass": 1,
            "GoldsteinScale": 2.0,
            "AvgTone": 0.2,
            "SourceURL": None,
            "Scrape_Status": "Success",
            "Error_Details": "",
            "url_id": 4,
            "doc_relevance_score": 65.0,
            "suspect_redirect_content": True,
            "analysis_include": True,
            "row_count": 1,
        },
        {
            "Date": 20200105,
            "Actor1Name": "Opposition",
            "Actor1CountryCode": "VEN",
            "Actor2Name": "US Gov",
            "Actor2CountryCode": "USA",
            "EventCode": 140,
            "QuadClass": 1,
            "GoldsteinScale": 2.5,
            "AvgTone": 0.3,
            "SourceURL": "https://www.example.com/a",
            "Scrape_Status": "Success",
            "Error_Details": "",
            "url_id": 1,
            "doc_relevance_score": 80.0,
            "suspect_redirect_content": False,
            "analysis_include": True,
            "row_count": 2,
        },
    ]


def build_analysis_url_rows() -> list[dict[str, object]]:
    return [
        {
            "url_id": 1,
            "SourceURL": "https://www.example.com/a",
            "Title": "Good title",
            "Text": "Good body text",
            "Scrape_Status": "Success",
            "doc_relevance_score": 80.0,
            "suspect_redirect_content": False,
            "analysis_include": True,
            "row_count": 2,
        },
        {
            "url_id": 2,
            "SourceURL": "https://news.test/b",
            "Title": "Non success",
            "Text": "Still has text",
            "Scrape_Status": "Error",
            "doc_relevance_score": 10.0,
            "suspect_redirect_content": False,
            "analysis_include": True,
            "row_count": 1,
        },
        {
            "url_id": 3,
            "SourceURL": "https://empty.test/c",
            "Title": "",
            "Text": "",
            "Scrape_Status": "Success",
            "doc_relevance_score": 90.0,
            "suspect_redirect_content": False,
            "analysis_include": False,
            "row_count": 1,
        },
        {
            "url_id": 4,
            "SourceURL": "https://fallback.test/d",
            "Title": "Needs content fallback",
            "Text": "Body present",
            "Scrape_Status": "Success",
            "doc_relevance_score": 65.0,
            "suspect_redirect_content": True,
            "analysis_include": True,
            "row_count": 1,
        },
    ]


def build_legacy_gdelt_rows() -> list[dict[str, object]]:
    content_by_url = {row["url_id"]: row for row in build_analysis_url_rows()}
    rows = []
    for row in build_analysis_event_rows():
        content = content_by_url[row["url_id"]]
        rows.append(
            {
                **row,
                "Title": content["Title"],
                "Text": content["Text"],
            }
        )
    return rows


def write_core_files(base_dir: Path) -> None:
    (base_dir / "data").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "analysis_ready").mkdir(parents=True, exist_ok=True)
    (base_dir / "preprocessing").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(build_legacy_gdelt_rows()).to_csv(base_dir / "data" / "gdelt_scraped.csv", index=False)
    pd.DataFrame(build_analysis_event_rows()).to_parquet(
        base_dir / "data" / "analysis_ready" / "analysis_events.parquet",
        index=False,
    )
    pd.DataFrame(build_analysis_url_rows()).to_parquet(
        base_dir / "data" / "analysis_ready" / "analysis_url_content.parquet",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "url_id": 1,
                "doc_relevance_score": 80.0,
                "source_domain": "example.com",
                "domain": "example.com",
                "suspect_redirect_content": False,
                "row_count": 2,
            },
            {
                "url_id": 2,
                "doc_relevance_score": 10.0,
                "source_domain": "news.test",
                "domain": "news.test",
                "suspect_redirect_content": False,
                "row_count": 1,
            },
            {
                "url_id": 3,
                "doc_relevance_score": 90.0,
                "source_domain": "empty.test",
                "domain": "empty.test",
                "suspect_redirect_content": False,
                "row_count": 1,
            },
            {
                "url_id": 4,
                "doc_relevance_score": 65.0,
                "source_domain": "fallback.test",
                "domain": "fallback.test",
                "suspect_redirect_content": True,
                "row_count": 2,
            },
        ]
    ).to_csv(base_dir / "preprocessing" / "url_lookup.csv", index=False)

    pd.DataFrame(
        [
            {"token": "venezuela", "relevance_score": 5.0},
            {"token": "sanction", "relevance_score": 8.0},
        ]
    ).to_csv(base_dir / "preprocessing" / "relevant_terms.csv", index=False)

    pd.DataFrame(
        [
            {"token": "us", "relevance_score": 1.0},
            {"token": "venezuela", "relevance_score": 9.0},
        ]
    ).to_csv(base_dir / "preprocessing" / "text_relevance_tokens.csv", index=False)


def test_analysis_config_paths_and_directory_creation(tmp_path: Path) -> None:
    cfg = AnalysisConfig(base_dir=tmp_path)
    assert cfg.data_dir == tmp_path / "data"
    assert cfg.analysis_ready_dir == tmp_path / "data" / "analysis_ready"
    assert cfg.preprocessing_dir == tmp_path / "preprocessing"
    assert cfg.output_dir == tmp_path / "analysis" / "outputs"
    assert cfg.analysis_events_path == tmp_path / "data" / "analysis_ready" / "analysis_events.parquet"
    assert cfg.analysis_url_content_path == tmp_path / "data" / "analysis_ready" / "analysis_url_content.parquet"
    cfg.ensure_directories()
    assert (cfg.output_dir / "sentiment").exists()
    assert (cfg.output_dir / "topics").exists()
    assert (cfg.output_dir / "clusters").exists()
    assert (cfg.output_dir / "visualizations").exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://www.Example.com/a", "example.com"),
        ("example.org/path", "example.org/path"),
        ("", None),
        (None, None),
    ],
)
def test_extract_domain(raw: object, expected: object) -> None:
    assert _extract_domain(raw) == expected


def test_load_lookup_and_relevance_tables(tmp_path: Path) -> None:
    cfg = AnalysisConfig(base_dir=tmp_path)
    assert load_url_lookup(cfg).empty
    assert list(load_relevant_terms(cfg).columns) == ["token", "relevance_score"]
    assert list(load_relevance_tokens(cfg).columns) == ["token", "relevance_score"]

    write_core_files(tmp_path)
    lookup = load_url_lookup(cfg)
    terms = load_relevant_terms(cfg, top_k=1)
    tokens = load_relevance_tokens(cfg, top_k=1)
    assert len(lookup) == 4
    assert terms.iloc[0]["token"] == "sanction"
    assert tokens.iloc[0]["token"] == "venezuela"


def test_load_gdelt_events_from_analysis_ready_parquet(tmp_path: Path) -> None:
    write_core_files(tmp_path)
    cfg = AnalysisConfig(base_dir=tmp_path)
    df = load_gdelt_events(cfg, merge_lookup=True)

    assert len(df) == 3
    assert set(df["url_id"].tolist()) == {1, 4}
    assert (df["url_id"] == 1).sum() == 2
    assert set(df["id"].tolist()) == {"gdelt_1", "gdelt_2", "gdelt_3"}
    assert set(df["type"].tolist()) == {"event"}
    assert set(df["event_category"].tolist()) == {"Verbal Cooperation"}
    assert "doc_relevance_score" in df.columns
    assert "source_domain" in df.columns
    assert "fallback.test" in set(df["source_domain"].dropna().tolist())
    assert df["created_datetime"].notna().all()
    assert df.loc[df["url_id"] == 4, "source_url"].iloc[0] == "https://fallback.test/d"
    assert df.loc[df["url_id"] == 4, "title"].iloc[0] == "Needs content fallback"
    assert set(df.loc[df["url_id"] == 1, "text"].unique()) == {"Good title Good body text"}


def test_load_gdelt_events_optional_filters(tmp_path: Path) -> None:
    write_core_files(tmp_path)
    cfg = AnalysisConfig(base_dir=tmp_path)
    cfg.require_successful_scrape = False
    cfg.min_doc_relevance_score = 50
    cfg.exclude_suspect_redirect_content = True

    df = load_gdelt_events(cfg, merge_lookup=True)
    assert len(df) == 2
    assert set(df["url_id"]) == {1}


def test_load_gdelt_events_parquet_mode_keeps_text_when_merge_lookup_false(tmp_path: Path) -> None:
    write_core_files(tmp_path)
    cfg = AnalysisConfig(base_dir=tmp_path)
    cfg.require_analysis_include = False

    df = load_gdelt_events(cfg, merge_lookup=False)
    assert len(df) == 3
    assert df["text"].str.len().gt(0).sum() == 3
    assert df.loc[df["url_id"] == 4, "source_url"].iloc[0] == "https://fallback.test/d"


def test_load_gdelt_events_legacy_csv_mode_still_supported(tmp_path: Path) -> None:
    write_core_files(tmp_path)
    cfg = AnalysisConfig(base_dir=tmp_path)
    cfg.use_analysis_ready_parquet = False

    df = load_gdelt_events(cfg, merge_lookup=True)
    assert len(df) == 3
    assert set(df["url_id"]) == {1, 4}
    assert df.loc[df["url_id"] == 4, "source_domain"].iloc[0] == "fallback.test"


def test_load_gdelt_events_missing_input_raises(tmp_path: Path) -> None:
    cfg = AnalysisConfig(base_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_gdelt_events(cfg)


def test_time_helpers_and_sampling() -> None:
    df = pd.DataFrame(
        {
            "id": [f"gdelt_{i}" for i in range(1, 7)],
            "created_datetime": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2021-01-01",
                    "2021-02-01",
                    "2021-03-01",
                ]
            ),
        }
    )
    df["year"] = df["created_datetime"].dt.year
    df["year_month"] = df["created_datetime"].dt.to_period("M").astype(str)

    months = get_time_periods(df, "month")
    quarters = get_time_periods(df, "quarter")
    years = get_time_periods(df, "year")
    assert months[0] == "2020-01"
    assert "2020Q1" in quarters
    assert years == ["2020", "2021"]

    sliced = filter_by_period(df, "2020-02-01", "2021-01-01")
    assert set(sliced["id"]) == {"gdelt_2", "gdelt_3", "gdelt_4"}

    sampled_small = sample_from_ids(df, ["gdelt_1", "gdelt_2"], n=5)
    assert len(sampled_small) == 2

    sampled = sample_from_ids(df, df["id"].tolist(), n=3, random_state=7)
    assert len(sampled) == 3
    assert set(sampled["id"]).issubset(set(df["id"]))
