from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import flag_redirect_duplicates
import plot_filter_stage_score_histograms
import remove_dup_articles
import run_preprocessing_pipeline


def test_flag_redirect_duplicates_helpers_and_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_path = tmp_path / "url_lookup.csv"
    output_path = tmp_path / "url_lookup_flagged.csv"
    review_path = tmp_path / "redirect_review.csv"
    repeated_text = "Repeated redirect body " * 20

    assert flag_redirect_duplicates.normalize_text("A\nB") == "a b"
    assert flag_redirect_duplicates.extract_domain("Example.com/path") == "example.com"
    assert flag_redirect_duplicates.stable_text_hash("abc") == flag_redirect_duplicates.stable_text_hash("abc")

    pd.DataFrame(
        [
            {"url_id": 1, "SourceURL": "http://a.com/1", "Title": "One", "Text": repeated_text},
            {"url_id": 2, "SourceURL": "http://b.com/2", "Title": "Two", "Text": repeated_text},
            {"url_id": 3, "SourceURL": "http://c.com/3", "Title": "Three", "Text": "short"},
        ]
    ).to_csv(lookup_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flag_redirect_duplicates.py",
            "--lookup",
            str(lookup_path),
            "--output",
            str(output_path),
            "--review-output",
            str(review_path),
            "--min-cluster-size",
            "2",
            "--min-text-length",
            "10",
        ],
    )

    flag_redirect_duplicates.main()

    out = pd.read_csv(output_path, low_memory=False)
    review = pd.read_csv(review_path, low_memory=False)
    assert out["suspect_redirect_content"].sum() == 2
    assert review.loc[0, "cluster_url_count"] == 2
    assert review.loc[0, "cluster_domain_count"] == 2


def test_remove_dup_articles_main_drops_entire_duplicate_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_path = tmp_path / "url_lookup.csv"
    gdelt_path = tmp_path / "gdelt_scraped.csv"

    pd.DataFrame(
        [
            {"url_id": 1, "Text": "Same article"},
            {"url_id": 2, "Text": "Same article"},
            {"url_id": 3, "Text": "Unique article"},
        ]
    ).to_csv(lookup_path, index=False)
    pd.DataFrame(
        [
            {"url_id": 1, "SourceURL": "http://example.com/1"},
            {"url_id": 2, "SourceURL": "http://example.com/2"},
            {"url_id": 3, "SourceURL": "http://example.com/3"},
        ]
    ).to_csv(gdelt_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remove_dup_articles.py",
            "--lookup",
            str(lookup_path),
            "--gdelt",
            str(gdelt_path),
            "--min-text-length",
            "1",
        ],
    )

    remove_dup_articles.main()

    lookup_out = pd.read_csv(lookup_path, low_memory=False)
    gdelt_out = pd.read_csv(gdelt_path, low_memory=False)
    assert lookup_out["url_id"].tolist() == [3]
    assert gdelt_out["url_id"].tolist() == [3]


def test_run_preprocessing_pipeline_helpers_and_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Path(run_preprocessing_pipeline.__file__).resolve().parent
    artifact_dir = base.parent / "data" / "preprocessing"
    assert run_preprocessing_pipeline.sanitize_tag("My File.csv") == "my_file_csv"

    custom_input = tmp_path / "Weekly Input.csv"
    custom_input.write_text("placeholder", encoding="utf-8")
    defaults = run_preprocessing_pipeline.derive_default_paths(custom_input, base)
    assert defaults["lookup"].name == "url_lookup_weekly_input.csv"
    assert defaults["lookup"].parent == artifact_dir

    anchors = tmp_path / "anchors.json"
    anchors.write_text("{}", encoding="utf-8")
    filter_rules = tmp_path / "filter_rule_config.json"
    filter_rules.write_text("{}", encoding="utf-8")

    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(run_preprocessing_pipeline, "run_step", lambda label, command: calls.append((label, command)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_preprocessing_pipeline.py",
            str(custom_input),
            "--anchors",
            str(anchors),
            "--filter-rules",
            str(filter_rules),
            "--sample-size",
            "5",
            "--seed",
            "99",
        ],
    )

    run_preprocessing_pipeline.main()

    assert len(calls) == 7
    assert calls[0][0] == "Step 1/7: Build URL Index"
    assert Path(calls[0][1][1]).name == "build_url_index.py"
    assert "url_lookup_weekly_input.csv" in " ".join(calls[0][1])
    assert str(artifact_dir / "filter_samples_weekly_input") in calls[5][1]
    assert "--filter-rules" in calls[5][1]
    assert str(filter_rules) in calls[5][1]
    assert Path(calls[-1][1][1]).name == "plot_filter_stage_score_histograms.py"


def test_plot_filter_stage_histogram_helpers_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fig, ax = plt.subplots()
    plot_filter_stage_score_histograms.annotate_counts(ax, {"drop": 1, "review": 2, "keep": 3}, 6)
    assert "drop: 1" in ax.texts[0].get_text()
    plt.close(fig)

    frame = pd.DataFrame(
        {
            "doc_relevance_score": [10, 20, 30],
            "decision": ["drop", "review", "keep"],
        }
    )
    fig, ax = plt.subplots()
    counts = plot_filter_stage_score_histograms.plot_stage(
        ax,
        frame.rename(columns={"decision": "filter_score_decision"}),
        "filter_score_decision",
        "Title",
        bins=3,
        x_min=10,
        x_max=30,
        mode="overlay",
    )
    plt.close(fig)
    assert counts == {"drop": 1, "review": 1, "keep": 1}

    eval_path = tmp_path / "url_filter_eval.csv"
    output_path = tmp_path / "histograms.png"
    pd.DataFrame(
        [
            {
                "in_filter_scope": True,
                "doc_relevance_score": 10,
                "filter_duplicate_decision": "drop",
                "filter_length_decision": "drop",
                "filter_score_decision": "drop",
                "filter_anchor_decision": "drop",
            },
            {
                "in_filter_scope": True,
                "doc_relevance_score": 25,
                "filter_duplicate_decision": "keep",
                "filter_length_decision": "review",
                "filter_score_decision": "review",
                "filter_anchor_decision": "review",
            },
            {
                "in_filter_scope": True,
                "doc_relevance_score": 40,
                "filter_duplicate_decision": "keep",
                "filter_length_decision": "keep",
                "filter_score_decision": "keep",
                "filter_anchor_decision": "keep",
            },
        ]
    ).to_csv(eval_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_filter_stage_score_histograms.py",
            "--eval-csv",
            str(eval_path),
            "--output",
            str(output_path),
            "--bins",
            "5",
            "--mode",
            "stacked",
        ],
    )

    plot_filter_stage_score_histograms.main()
    assert output_path.exists()
