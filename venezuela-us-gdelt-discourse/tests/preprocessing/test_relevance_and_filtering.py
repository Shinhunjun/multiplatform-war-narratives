from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

import build_duplicate_filter_eval
import build_text_relevance_tokens
import evaluate_filter_strategy
import score_url_relevance


def repeated_words(word: str, count: int) -> str:
    return " ".join([word] * count)


def test_build_stopword_set_combines_nltk_and_domain_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(build_text_relevance_tokens, "ensure_nltk_resources", lambda: calls.append("called"))
    monkeypatch.setattr(build_text_relevance_tokens.stopwords, "words", lambda lang: ["common", "house"])

    stopword_set = build_text_relevance_tokens.build_stopword_set()

    assert calls == ["called"]
    assert "common" in stopword_set
    assert "news" in stopword_set
    assert "house" in stopword_set


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("JJ", build_text_relevance_tokens.wordnet.ADJ),
        ("VB", build_text_relevance_tokens.wordnet.VERB),
        ("NN", build_text_relevance_tokens.wordnet.NOUN),
        ("RB", build_text_relevance_tokens.wordnet.ADV),
        ("FW", build_text_relevance_tokens.wordnet.NOUN),
    ],
)
def test_penn_to_wordnet_maps_expected_categories(tag: str, expected: str) -> None:
    assert build_text_relevance_tokens.penn_to_wordnet(tag) == expected


def test_normalize_raw_token_and_parse_text_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        build_text_relevance_tokens,
        "word_tokenize",
        lambda text: ["U.S.", "officials", "n't", "Venezuela's", "!!!"],
    )

    parsed = build_text_relevance_tokens.parse_text_tokens("U.S. officials Venezuela's")

    assert build_text_relevance_tokens.normalize_raw_token("`Quote`") == "quote"
    assert "u.s." in parsed
    assert "officials" in parsed
    assert "venezuela" in parsed
    assert "n't" not in parsed


def test_tokenize_filters_stopwords_and_keeps_special_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_text_relevance_tokens,
        "parse_text_tokens",
        lambda text: ["running", "us", "news", "123"],
    )
    monkeypatch.setattr(
        build_text_relevance_tokens,
        "pos_tag",
        lambda tokens: [("running", "VBG"), ("news", "NN")],
    )

    class FakeLemmatizer:
        def lemmatize(self, token: str, pos: str | None = None) -> str:
            return {"running": "run", "news": "news"}.get(token, token)

    tokens = build_text_relevance_tokens.tokenize(
        "ignored",
        FakeLemmatizer(),
        {"news"},
    )

    assert tokens == {"run", "us"}


def test_normalize_seed_terms_and_parse_token_array(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        build_text_relevance_tokens,
        "parse_text_tokens",
        lambda text: text.lower().split(),
    )

    class FakeLemmatizer:
        def lemmatize(self, token: str, pos: str | None = None) -> str:
            return token.rstrip("s")

    seeds = build_text_relevance_tokens.normalize_seed_terms(["Sanctions", "U.S."], FakeLemmatizer())

    assert seeds == {"sanction", "u.s."}
    assert build_text_relevance_tokens.parse_token_array('["US", "policy", "123"]') == {"us", "policy"}
    assert build_text_relevance_tokens.parse_token_array("alpha, beta") == {"alpha", "beta"}


def test_build_text_relevance_tokens_main_writes_ranked_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_path = tmp_path / "url_lookup.csv"
    eval_path = tmp_path / "url_filter_eval.csv"
    output_path = tmp_path / "text_relevance_tokens.csv"

    pd.DataFrame(
        [
            {"url_id": 1, "Tokens": '["venezuela", "oil"]', "Scrape_Status": "success"},
            {"url_id": 2, "Tokens": '["market"]', "Scrape_Status": "success"},
            {"url_id": 3, "Tokens": '["venezuela", "duplicate"]', "Scrape_Status": "success"},
        ]
    ).to_csv(lookup_path, index=False)
    pd.DataFrame(
        [
            {"url_id": 1, "used_for_token_training": True},
            {"url_id": 2, "used_for_token_training": True},
            {"url_id": 3, "used_for_token_training": False},
        ]
    ).to_csv(eval_path, index=False)

    class FakeLemmatizer:
        def lemmatize(self, token: str, pos: str | None = None) -> str:
            return token

    monkeypatch.setattr(build_text_relevance_tokens, "ensure_nltk_resources", lambda: None)
    monkeypatch.setattr(build_text_relevance_tokens, "WordNetLemmatizer", FakeLemmatizer)
    monkeypatch.setattr(build_text_relevance_tokens, "word_tokenize", lambda text: text.split())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_text_relevance_tokens.py",
            "--lookup",
            str(lookup_path),
            "--output",
            str(output_path),
            "--eval",
            str(eval_path),
            "--exclude-duplicate-drops",
            "--require-success-status",
            "--seed-terms",
            "venezuela",
            "--min-doc-frac",
            "0",
            "--max-doc-frac",
            "1",
            "--min-seed-doc-freq",
            "1",
            "--min-nonseed-doc-freq",
            "0",
        ],
    )

    build_text_relevance_tokens.main()

    out = pd.read_csv(output_path, low_memory=False)
    assert not out.empty
    assert out["rank"].tolist() == list(range(1, len(out) + 1))
    assert "venezuela" in set(out["token"])
    assert out.loc[out["token"] == "venezuela", "is_protected_seed_token"].item() is True


def test_score_url_relevance_parse_token_array() -> None:
    assert score_url_relevance.parse_token_array('["alpha", "beta"]') == {"alpha", "beta"}
    assert score_url_relevance.parse_token_array("alpha, beta") == {"alpha", "beta"}
    assert score_url_relevance.parse_token_array(None) == set()


def test_score_url_relevance_main_scores_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_path = tmp_path / "url_lookup.csv"
    relevance_path = tmp_path / "text_relevance_tokens.csv"
    output_path = tmp_path / "scored_lookup.csv"

    pd.DataFrame(
        [
            {"url_id": 1, "Tokens": '["alpha", "beta"]'},
            {"url_id": 2, "Tokens": ""},
            {"url_id": 3, "Tokens": "gamma, delta"},
        ]
    ).to_csv(lookup_path, index=False)
    pd.DataFrame(
        [
            {"token": "alpha", "relevance_score": 4.0},
            {"token": "beta", "relevance_score": 1.0},
            {"token": "delta", "relevance_score": 9.0},
        ]
    ).to_csv(relevance_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_url_relevance.py",
            "--lookup",
            str(lookup_path),
            "--relevance",
            str(relevance_path),
            "--output",
            str(output_path),
        ],
    )

    score_url_relevance.main()

    out = pd.read_csv(output_path, low_memory=False)
    assert math.isclose(out.loc[0, "doc_relevance_sum"], 5.0)
    assert out.loc[0, "doc_relevance_matches"] == 2
    assert out.loc[1, "doc_relevance_score"] == 0.0
    assert math.isclose(out.loc[2, "doc_relevance_score"], 9.0 / math.sqrt(2))


def test_build_duplicate_filter_eval_helpers_and_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_path = tmp_path / "url_lookup.csv"
    output_path = tmp_path / "url_filter_eval.csv"
    summary_path = tmp_path / "summary.csv"

    assert build_duplicate_filter_eval.normalize_text_for_hash("A\nB  C") == "a b c"
    assert build_duplicate_filter_eval.text_hash("") == ""

    existing = pd.DataFrame([{"url_id": 1, "filter_duplicate_decision": "keep"}])
    incoming = pd.DataFrame([{"url_id": 1, "filter_duplicate_decision": "drop"}])
    merged = build_duplicate_filter_eval.upsert_eval(existing, incoming)
    assert merged.loc[0, "filter_duplicate_decision"] == "drop"

    pd.DataFrame(
        [
            {"url_id": 1, "Text": "Same article", "Tokens": '["a"]', "Scrape_Status": "success"},
            {"url_id": 2, "Text": "Same article", "Tokens": '["b"]', "Scrape_Status": "success"},
            {"url_id": 3, "Text": "", "Tokens": "", "Scrape_Status": "success"},
            {"url_id": 4, "Text": "Ignored", "Tokens": '["x"]', "Scrape_Status": "failed"},
        ]
    ).to_csv(lookup_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_duplicate_filter_eval.py",
            "--lookup",
            str(lookup_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
    )

    build_duplicate_filter_eval.main()

    out = pd.read_csv(output_path, low_memory=False)
    assert out.loc[out["url_id"] == 1, "filter_duplicate_decision"].item() == "drop"
    assert out.loc[out["url_id"] == 2, "used_for_token_training"].item() is False
    assert out.loc[out["url_id"] == 3, "filter_duplicate_decision"].item() == "out_of_scope"
    assert summary_path.exists()


def test_evaluate_filter_strategy_helpers_cover_key_branches() -> None:
    assert evaluate_filter_strategy.parse_token_set('["A", "b"]') == {"a", "b"}
    assert evaluate_filter_strategy.count_words("One two three.") == 3
    assert evaluate_filter_strategy.normalize_text_for_hash("A\nB") == "a b"
    assert evaluate_filter_strategy.text_hash("") == ""
    assert evaluate_filter_strategy.decision_duplicate(2) == "drop"
    assert evaluate_filter_strategy.decision_duplicate(2, drop_cluster_size_gt=2) == "keep"
    assert evaluate_filter_strategy.decision_length(60) == "review"
    assert evaluate_filter_strategy.decision_length(50, drop_lt=60, review_lt=90) == "drop"
    assert evaluate_filter_strategy.decision_score(10) == "drop"
    assert evaluate_filter_strategy.decision_score(30, drop_lt=20, review_lt=35) == "review"
    assert evaluate_filter_strategy.decision_anchor(True, False, True) == "review"
    assert evaluate_filter_strategy.final_decision("keep", "review", "keep", "keep") == "review"
    assert (
        evaluate_filter_strategy.final_decision(
            "drop",
            "keep",
            "keep",
            "review",
            priority=("review", "drop", "keep"),
        )
        == "review"
    )
    assert (
        evaluate_filter_strategy.reasons_for_row("keep", "review", "drop", "keep", True)
        == "length_review|score_drop"
    )

    sample = pd.DataFrame(
        [
            {"decision": "drop", "value": 1},
            {"decision": "drop", "value": 2},
            {"decision": "keep", "value": 3},
        ]
    )
    sampled = evaluate_filter_strategy.stratified_sample(sample, "decision", sample_size=1, seed=7)
    assert len(sampled) == 2

    merged = evaluate_filter_strategy.upsert_eval(
        pd.DataFrame([{"url_id": 1, "filter_final_decision": "drop"}]),
        pd.DataFrame([{"url_id": 1, "filter_final_decision": "keep"}]),
    )
    assert merged.loc[0, "filter_final_decision"] == "keep"


def test_load_filter_rules_validates_and_normalizes_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "filter_rule_config.json"
    cfg_path.write_text(
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

    loaded = evaluate_filter_strategy.load_filter_rules(cfg_path)
    assert loaded["thresholds"]["length_drop_lt"] == 40
    assert loaded["final_decision_priority"] == ("drop", "review", "keep")


def test_evaluate_filter_strategy_main_writes_eval_summary_and_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_path = tmp_path / "url_lookup.csv"
    anchors_path = tmp_path / "anchor_token_sets.json"
    filter_rules_path = tmp_path / "filter_rule_config.json"
    output_path = tmp_path / "url_filter_eval.csv"
    summary_path = tmp_path / "summary.csv"
    sample_dir = tmp_path / "samples"

    keep_text = repeated_words("keep", 90)
    review_text = repeated_words("review", 50)
    duplicate_text = repeated_words("duplicate", 90)

    pd.DataFrame(
        [
            {
                "url_id": 1,
                "SourceURL": "http://example.com/1",
                "Title": "Keep",
                "Text": keep_text,
                "Tokens": '["venezuela", "us", "policy"]',
                "Scrape_Status": "success",
                "doc_relevance_score": 55,
            },
            {
                "url_id": 2,
                "SourceURL": "http://example.com/2",
                "Title": "Duplicate",
                "Text": duplicate_text,
                "Tokens": '["venezuela", "us", "policy"]',
                "Scrape_Status": "success",
                "doc_relevance_score": 55,
            },
            {
                "url_id": 3,
                "SourceURL": "http://example.com/3",
                "Title": "Duplicate 2",
                "Text": duplicate_text,
                "Tokens": '["venezuela", "us", "policy"]',
                "Scrape_Status": "success",
                "doc_relevance_score": 55,
            },
            {
                "url_id": 4,
                "SourceURL": "http://example.com/4",
                "Title": "Review",
                "Text": review_text,
                "Tokens": '["venezuela", "sanction"]',
                "Scrape_Status": "success",
                "doc_relevance_score": 30,
            },
            {
                "url_id": 5,
                "SourceURL": "http://example.com/5",
                "Title": "Out of scope",
                "Text": "",
                "Tokens": "",
                "Scrape_Status": "success",
                "doc_relevance_score": 99,
            },
        ]
    ).to_csv(lookup_path, index=False)

    anchors_path.write_text(
        """
{
  "anchors": {
    "venezuela_primary": ["venezuela"],
    "us_primary": ["us"],
    "us_primary_token_pairs": [["white", "house"]],
    "relation_context_secondary": ["sanction"]
  }
}
""".strip(),
        encoding="utf-8",
    )
    filter_rules_path.write_text(
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

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_filter_strategy.py",
            "--lookup",
            str(lookup_path),
            "--anchors",
            str(anchors_path),
            "--filter-rules",
            str(filter_rules_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--sample-dir",
            str(sample_dir),
            "--sample-size",
            "2",
            "--seed",
            "123",
        ],
    )

    evaluate_filter_strategy.main()

    out = pd.read_csv(output_path, low_memory=False)
    assert out.loc[out["url_id"] == 1, "filter_final_decision"].item() == "keep"
    assert out.loc[out["url_id"] == 2, "filter_duplicate_decision"].item() == "drop"
    assert out.loc[out["url_id"] == 4, "filter_final_decision"].item() == "review"
    assert out.loc[out["url_id"] == 5, "filter_reasons"].item() == "out_of_scope"
    assert summary_path.exists()
    assert (sample_dir / "sample_step0_duplicate.csv").exists()
    assert (sample_dir / "sample_step1_length.csv").exists()
    assert (sample_dir / "sample_step2_score.csv").exists()
    assert (sample_dir / "sample_step3_anchor.csv").exists()
    assert (sample_dir / "sample_final_decision.csv").exists()
