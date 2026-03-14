from __future__ import annotations

from collections import Counter
from pathlib import Path
import warnings

import pandas as pd
import pytest

import run_eda_scraped as eda


def build_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "20190115",
                "Actor1Name": "Venezuela Gov",
                "Actor1CountryCode": "VEN",
                "Actor2Name": "US Gov",
                "Actor2CountryCode": "USA",
                "EventCode": 120,
                "QuadClass": 1,
                "GoldsteinScale": 2.5,
                "AvgTone": 0.2,
                "SourceURL": "http://example.com/a",
                "Title": "Venezuela and US talks",
                "Text": "Venezuela and US discuss sanctions policy.",
                "Scrape_Status": "success",
            },
            {
                "Date": "20190215",
                "Actor1Name": "US Gov",
                "Actor1CountryCode": "USA",
                "Actor2Name": "Venezuela Gov",
                "Actor2CountryCode": "VEN",
                "EventCode": 173,
                "QuadClass": 4,
                "GoldsteinScale": -6.0,
                "AvgTone": -1.2,
                "SourceURL": "http://example.com/a",
                "Title": "US sanctions expanded",
                "Text": "US expands sanctions on Venezuela officials.",
                "Scrape_Status": "SUCCESS_RETRY",
            },
            {
                "Date": "20200110",
                "Actor1Name": "Regional Org",
                "Actor1CountryCode": "COL",
                "Actor2Name": "Venezuela Gov",
                "Actor2CountryCode": "VEN",
                "EventCode": 40,
                "QuadClass": 2,
                "GoldsteinScale": 4.0,
                "AvgTone": 1.5,
                "SourceURL": "http://example.com/b",
                "Title": "Regional cooperation",
                "Text": "Regional support for humanitarian efforts.",
                "Scrape_Status": "timeout",
            },
        ]
    )


def test_load_data_missing_file_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    assert eda.load_data(missing) is None


def test_load_data_raises_on_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"Date": "20190101"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        eda.load_data(path)


def test_load_data_and_preprocess_create_derived_columns(tmp_path: Path) -> None:
    path = tmp_path / "gdelt_scraped.csv"
    raw = build_raw_df()
    raw.to_csv(path, index=False)

    loaded = eda.load_data(path)
    assert loaded is not None
    assert len(loaded) == 3

    processed = eda.preprocess_data(loaded.copy())
    assert "DateObject" in processed.columns
    assert "Year" in processed.columns
    assert "Month" in processed.columns
    assert "EventCategory" in processed.columns
    assert "Initiator" in processed.columns
    assert processed["EventCategory"].tolist() == [
        "Verbal Cooperation",
        "Material Conflict",
        "Material Cooperation",
    ]
    assert processed["Initiator"].tolist() == ["Venezuela", "USA", "Other"]


def test_plot_functions_write_expected_png_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eda, "OUTPUT_DIR", tmp_path)
    processed = eda.preprocess_data(build_raw_df().copy())

    eda.plot_timeline(processed)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="vert: bool will be deprecated in a future version.*",
            category=PendingDeprecationWarning,
        )
        eda.plot_yearly_distribution(processed)
    eda.plot_quadclass_distribution(processed)
    eda.plot_intensity_metrics(processed)
    eda.plot_tone_trend(processed, rolling_window=3)

    expected = [
        "01_gdelt_timeline.png",
        "02_gdelt_yearly_stats.png",
        "03_gdelt_categories.png",
        "04_gdelt_intensity.png",
        "05_gdelt_tone_trend.png",
    ]
    for name in expected:
        out = tmp_path / name
        assert out.exists(), f"missing chart: {name}"
        assert out.stat().st_size > 0, f"empty chart: {name}"


def test_scrape_and_url_metric_plots_return_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eda, "OUTPUT_DIR", tmp_path)
    processed = eda.preprocess_data(build_raw_df().copy())

    scrape = eda.plot_scrape_status(processed)
    urls = eda.plot_url_uniqueness(processed)

    assert scrape["total_rows"] == 3
    assert scrape["success_count"] == 2
    assert scrape["success_rate"] == pytest.approx(66.6666, rel=1e-3)
    assert urls == {"valid_url_rows": 3, "unique_urls": 2, "duplicate_url_rows": 1}
    assert (tmp_path / "06_scraped_status.png").exists()
    assert (tmp_path / "07_scraped_url_uniqueness.png").exists()


def test_token_counter_applies_normalization_and_stopwords(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eda, "ensure_nltk_resources", lambda: None)
    monkeypatch.setattr(eda, "build_stopword_set", lambda: {"news"})
    monkeypatch.setattr(eda, "parse_text_tokens", lambda text: text.lower().split())
    monkeypatch.setattr(eda, "pos_tag", lambda toks: [(tok, "NN") for tok in toks])

    class FakeLemmatizer:
        def lemmatize(self, token: str, pos: str | None = None) -> str:
            return token.rstrip("s")

    monkeypatch.setattr(eda, "WordNetLemmatizer", FakeLemmatizer)

    counts = eda._token_counter(pd.Series(["US sanctions news", "Venezuela sanctions"]))

    assert counts["us"] == 1
    assert counts["sanction"] == 2
    assert counts["venezuela"] == 1
    assert "news" not in counts


def test_make_wordcloud_and_top_words(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(eda, "OUTPUT_DIR", tmp_path)

    class FakeWordCloud:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def generate_from_frequencies(self, freqs: dict[str, int]) -> "FakeWordCloud":
            self.freqs = freqs
            return self

    monkeypatch.setattr(eda, "WordCloud", FakeWordCloud)
    monkeypatch.setattr(eda.plt, "imshow", lambda *args, **kwargs: None)

    counts = Counter({"venezuela": 5, "us": 3, "sanction": 2})
    eda.make_wordcloud_from_counts(counts, "Test Cloud", "test_wordcloud.png", max_words=25)
    assert (tmp_path / "test_wordcloud.png").exists()

    top = eda.top_words_with_share(counts, top_n=2)
    assert top[0][0] == "venezuela"
    assert top[0][1] == 5
    assert top[0][2] == pytest.approx(50.0)
    assert len(top) == 2


def test_generate_report_writes_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eda, "OUTPUT_DIR", tmp_path)
    processed = eda.preprocess_data(build_raw_df().copy())

    eda.generate_report(
        processed,
        scrape_metrics={"total_rows": 3, "success_count": 2, "success_rate": 66.67},
        url_metrics={"valid_url_rows": 3, "unique_urls": 2, "duplicate_url_rows": 1},
        top_title_words=[("venezuela", 3, 30.0)],
        top_text_words=[("sanction", 4, 20.0)],
    )

    report = tmp_path / "GDELT_Scraped_EDA_Report.md"
    text = report.read_text(encoding="utf-8")
    assert report.exists()
    assert "Venezuela-US GDELT Comprehensive Scraped Analysis Report" in text
    assert "Top Conflict Events" in text
    assert "| venezuela | 3 | 30.00% |" in text
    assert "| sanction | 4 | 20.00% |" in text


def test_main_returns_early_when_data_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"preprocess": False}
    monkeypatch.setattr(eda, "load_data", lambda path: None)
    monkeypatch.setattr(
        eda,
        "preprocess_data",
        lambda df: called.__setitem__("preprocess", True),
    )

    eda.main()
    assert called["preprocess"] is False


def test_main_runs_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    raw = build_raw_df()

    def record(name: str) -> None:
        calls.append(name)

    monkeypatch.setattr(eda, "load_data", lambda path: raw.copy())
    monkeypatch.setattr(eda, "preprocess_data", lambda df: df)
    monkeypatch.setattr(eda, "plot_timeline", lambda df: record("timeline"))
    monkeypatch.setattr(eda, "plot_yearly_distribution", lambda df: record("yearly"))
    monkeypatch.setattr(eda, "plot_quadclass_distribution", lambda df: record("quad"))
    monkeypatch.setattr(eda, "plot_intensity_metrics", lambda df: record("intensity"))
    monkeypatch.setattr(eda, "plot_tone_trend", lambda df: record("tone"))
    monkeypatch.setattr(
        eda,
        "plot_scrape_status",
        lambda df: (record("scrape"), {"total_rows": 3, "success_count": 2, "success_rate": 66.67})[1],
    )
    monkeypatch.setattr(
        eda,
        "plot_url_uniqueness",
        lambda df: (record("urls"), {"valid_url_rows": 3, "unique_urls": 2, "duplicate_url_rows": 1})[1],
    )

    token_call_counter = {"count": 0}

    def fake_token_counter(series: pd.Series) -> Counter:
        token_call_counter["count"] += 1
        return Counter({"venezuela": 2, "us": 1})

    monkeypatch.setattr(eda, "_token_counter", fake_token_counter)
    monkeypatch.setattr(
        eda,
        "make_wordcloud_from_counts",
        lambda counts, chart_title, out_name: record(f"wc:{out_name}"),
    )
    monkeypatch.setattr(
        eda,
        "top_words_with_share",
        lambda counts, top_n=10: [("venezuela", 2, 66.0)],
    )
    monkeypatch.setattr(
        eda,
        "generate_report",
        lambda df, scrape_metrics, url_metrics, top_title_words, top_text_words: record("report"),
    )

    eda.main()

    assert token_call_counter["count"] == 2
    assert "timeline" in calls
    assert "yearly" in calls
    assert "quad" in calls
    assert "intensity" in calls
    assert "tone" in calls
    assert "scrape" in calls
    assert "urls" in calls
    assert "wc:08_title_wordcloud.png" in calls
    assert "wc:09_text_wordcloud.png" in calls
    assert "report" in calls
