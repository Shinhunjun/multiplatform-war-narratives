from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import (
    ANCHORS_PATH,
    FILTER_EVAL_PATH,
    FILTER_PLOT_PATH,
    FILTER_RULES_PATH,
    FILTER_SUMMARY_PATH,
    LOOKUP_PATH,
    load_changed_url_ids,
    ensure_empty_csv,
    bootstrap_project_paths,
)


bootstrap_project_paths()
from evaluate_filter_strategy import (
    count_words,
    decision_anchor,
    decision_duplicate,
    decision_length,
    decision_score,
    final_decision,
    load_filter_rules,
    normalize_text_for_hash,
    parse_token_set,
    reasons_for_row,
    stratified_sample,
    text_hash,
    upsert_eval,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for weekly static-filter application."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply the existing static filtering rules only to changed url_lookup rows while preserving "
            "historical evaluation decisions for unchanged URLs."
        )
    )
    parser.add_argument("--lookup", type=Path, default=LOOKUP_PATH, help="Path to data/preprocessing/url_lookup.csv")
    parser.add_argument("--eval", type=Path, default=FILTER_EVAL_PATH, help="Path to data/preprocessing/url_filter_eval.csv")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=FILTER_SUMMARY_PATH,
        help="Path to data/preprocessing/url_filter_summary_counts.csv",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=FILTER_PLOT_PATH,
        help="Path to data/preprocessing/filter_stage_score_histograms.png",
    )
    parser.add_argument("--changed-url-ids", type=Path, required=True, help="CSV of changed url_ids")
    parser.add_argument("--filter-rules", type=Path, default=FILTER_RULES_PATH, help="Path to filter_rule_config.json")
    parser.add_argument("--anchors", type=Path, default=ANCHORS_PATH, help="Path to anchor_token_sets.json")
    parser.add_argument("--sample-dir", type=Path, required=True, help="Weekly QA sample output directory")
    parser.add_argument("--sample-size", type=int, default=25, help="Sample size per decision bucket")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic samples")
    return parser.parse_args()


def write_weekly_samples(df: pd.DataFrame, sample_dir: Path, sample_size: int, seed: int) -> None:
    """Write weekly QA sample CSVs using changed rows only."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        for filename in [
            "sample_step0_duplicate.csv",
            "sample_step1_length.csv",
            "sample_step2_score.csv",
            "sample_step3_anchor.csv",
            "sample_final_decision.csv",
        ]:
            ensure_empty_csv(sample_dir / filename, ["url_id", "url", "doc_relevance_score", "filter_final_decision"])
        return

    sample_cols = [
        "url_id",
        "SourceURL",
        "Title",
        "Text",
        "duplicate_cluster_size",
        "text_word_count",
        "doc_relevance_score_num",
        "has_ven_anchor",
        "has_us_primary_token",
        "has_us_primary_pair",
        "has_us_primary",
        "has_relation_secondary",
        "filter_duplicate_decision",
        "filter_length_decision",
        "filter_score_decision",
        "filter_anchor_decision",
        "filter_final_decision",
        "filter_reasons",
    ]

    scope_df = df[df["in_filter_scope"]].copy()
    step0_df = scope_df.copy()
    step1_df = scope_df[scope_df["filter_duplicate_decision"] != "drop"].copy()
    step2_df = step1_df[step1_df["filter_length_decision"] != "drop"].copy()
    step3_df = step2_df[step2_df["filter_score_decision"] != "drop"].copy()

    samples = {
        "sample_step0_duplicate.csv": stratified_sample(step0_df, "filter_duplicate_decision", sample_size, seed),
        "sample_step1_length.csv": stratified_sample(step1_df, "filter_length_decision", sample_size, seed),
        "sample_step2_score.csv": stratified_sample(step2_df, "filter_score_decision", sample_size, seed),
        "sample_step3_anchor.csv": stratified_sample(step3_df, "filter_anchor_decision", sample_size, seed),
        "sample_final_decision.csv": stratified_sample(scope_df, "filter_final_decision", sample_size, seed),
    }

    for name, sample_df in samples.items():
        out = sample_df[sample_cols].rename(
            columns={"SourceURL": "url", "doc_relevance_score_num": "doc_relevance_score"}
        )
        out.to_csv(sample_dir / name, index=False)


def main() -> None:
    """Evaluate static filters for changed lookup rows and upsert url_filter_eval.csv."""
    args = parse_args()
    lookup_df = pd.read_csv(args.lookup, low_memory=False)
    existing_eval = pd.read_csv(args.eval, low_memory=False) if args.eval.exists() else pd.DataFrame()
    changed_df = load_changed_url_ids(args.changed_url_ids)

    for col in ["Tokens", "Text", "Scrape_Status", "doc_relevance_score"]:
        if col not in lookup_df.columns:
            lookup_df[col] = "" if col != "doc_relevance_score" else 0.0

    if changed_df.empty:
        if existing_eval.empty:
            ensure_empty_csv(
                args.eval,
                [
                    "url_id",
                    "in_filter_scope",
                    "duplicate_text_hash",
                    "duplicate_cluster_size",
                    "text_word_count",
                    "doc_relevance_score",
                    "has_ven_anchor",
                    "has_us_primary_token",
                    "has_us_primary_pair",
                    "has_us_primary",
                    "has_relation_secondary",
                    "filter_duplicate_decision",
                    "filter_length_decision",
                    "filter_score_decision",
                    "filter_anchor_decision",
                    "filter_final_decision",
                    "filter_reasons",
                ],
            )
        else:
            existing_eval.to_csv(args.eval, index=False)
        write_weekly_samples(pd.DataFrame(), args.sample_dir, args.sample_size, args.seed)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "preprocessing" / "plot_filter_stage_score_histograms.py"),
                "--eval-csv",
                str(args.eval),
                "--output",
                str(args.plot_output),
            ],
            check=True,
        )
        print(f"No changed url_ids to filter. Existing evaluation retained: {args.eval}")
        return

    changed_ids = set(changed_df["url_id"].dropna().astype(int).tolist())
    rule_cfg = load_filter_rules(args.filter_rules)
    thresholds = rule_cfg["thresholds"]
    scope_cfg = rule_cfg["scope"]
    final_priority = rule_cfg["final_decision_priority"]

    anchor_cfg = json.loads(args.anchors.read_text(encoding="utf-8"))
    anchors = anchor_cfg.get("anchors", {})
    ven_primary = set(anchors.get("venezuela_primary", []))
    us_primary = set(anchors.get("us_primary", []))
    us_pairs = [tuple(pair) for pair in anchors.get("us_primary_token_pairs", []) if len(pair) >= 2]
    relation_secondary = set(anchors.get("relation_context_secondary", []))

    lookup_df["url_id"] = pd.to_numeric(lookup_df["url_id"], errors="coerce").astype("Int64")
    success_mask = lookup_df["Scrape_Status"].fillna("").astype(str).str.contains("success", case=False)
    nonempty_text_mask = lookup_df["Text"].fillna("").astype(str).str.strip() != ""
    nonempty_tokens_mask = lookup_df["Tokens"].fillna("").astype(str).str.strip() != ""
    score = pd.to_numeric(lookup_df["doc_relevance_score"], errors="coerce")

    in_scope = pd.Series(True, index=lookup_df.index, dtype=bool)
    if scope_cfg["require_success_status"]:
        in_scope &= success_mask
    if scope_cfg["require_nonempty_text"]:
        in_scope &= nonempty_text_mask
    if scope_cfg["require_nonempty_tokens"]:
        in_scope &= nonempty_tokens_mask
    if scope_cfg["require_numeric_score"]:
        in_scope &= score.notna()

    lookup_df["in_filter_scope"] = in_scope
    scope_df = lookup_df.loc[in_scope, ["url_id", "Text"]].copy()
    if scope_df.empty:
        hash_lookup = {}
        cluster_lookup = {}
    else:
        scope_df["duplicate_text_hash"] = scope_df["Text"].map(normalize_text_for_hash).map(text_hash)
        scope_df["duplicate_cluster_size"] = scope_df["duplicate_text_hash"].map(scope_df["duplicate_text_hash"].value_counts()).fillna(0).astype(int)
        hash_lookup = dict(zip(scope_df["url_id"].astype(int), scope_df["duplicate_text_hash"]))
        cluster_lookup = dict(zip(scope_df["url_id"].astype(int), scope_df["duplicate_cluster_size"]))

    changed_rows = lookup_df[lookup_df["url_id"].isin(list(changed_ids))].copy()
    changed_rows["text_word_count"] = changed_rows["Text"].apply(count_words)
    changed_rows["doc_relevance_score_num"] = pd.to_numeric(changed_rows["doc_relevance_score"], errors="coerce")

    has_ven_anchor: list[bool] = []
    has_us_primary_token: list[bool] = []
    has_us_primary_pair: list[bool] = []
    has_us_primary: list[bool] = []
    has_relation_secondary: list[bool] = []
    duplicate_hashes: list[str] = []
    duplicate_cluster_sizes: list[int] = []
    duplicate_decisions: list[str] = []
    length_decisions: list[str] = []
    score_decisions: list[str] = []
    anchor_decisions: list[str] = []
    final_decisions: list[str] = []
    reasons: list[str] = []

    for record in changed_rows.to_dict(orient="records"):
        url_id = int(record["url_id"])
        row_in_scope = bool(record["in_filter_scope"])
        dup_hash = hash_lookup.get(url_id, "")
        dup_cluster_size = int(cluster_lookup.get(url_id, 0))
        tokens = parse_token_set(record.get("Tokens"))

        h_ven = bool(tokens & ven_primary)
        h_us_token = bool(tokens & us_primary)
        h_us_pair = any(all(token in tokens for token in pair) for pair in us_pairs)
        h_us = h_us_token or h_us_pair
        h_rel = bool(tokens & relation_secondary)

        if row_in_scope:
            duplicate_dec = decision_duplicate(
                dup_cluster_size,
                drop_cluster_size_gt=thresholds["duplicate_drop_cluster_size_gt"],
            )
            length_dec = decision_length(
                int(record["text_word_count"]),
                drop_lt=thresholds["length_drop_lt"],
                review_lt=thresholds["length_review_lt"],
            )
            score_dec = decision_score(
                float(record["doc_relevance_score_num"]),
                drop_lt=thresholds["score_drop_lt"],
                review_lt=thresholds["score_review_lt"],
            )
            anchor_dec = decision_anchor(h_ven, h_us, h_rel)
            final_dec = final_decision(
                duplicate_dec,
                length_dec,
                score_dec,
                anchor_dec,
                priority=final_priority,
            )
        else:
            duplicate_dec = "out_of_scope"
            length_dec = "out_of_scope"
            score_dec = "out_of_scope"
            anchor_dec = "out_of_scope"
            final_dec = "out_of_scope"

        has_ven_anchor.append(h_ven)
        has_us_primary_token.append(h_us_token)
        has_us_primary_pair.append(h_us_pair)
        has_us_primary.append(h_us)
        has_relation_secondary.append(h_rel)
        duplicate_hashes.append(dup_hash)
        duplicate_cluster_sizes.append(dup_cluster_size)
        duplicate_decisions.append(duplicate_dec)
        length_decisions.append(length_dec)
        score_decisions.append(score_dec)
        anchor_decisions.append(anchor_dec)
        final_decisions.append(final_dec)
        reasons.append(reasons_for_row(duplicate_dec, length_dec, score_dec, anchor_dec, row_in_scope))

    changed_rows["duplicate_text_hash"] = duplicate_hashes
    changed_rows["duplicate_cluster_size"] = duplicate_cluster_sizes
    changed_rows["has_ven_anchor"] = has_ven_anchor
    changed_rows["has_us_primary_token"] = has_us_primary_token
    changed_rows["has_us_primary_pair"] = has_us_primary_pair
    changed_rows["has_us_primary"] = has_us_primary
    changed_rows["has_relation_secondary"] = has_relation_secondary
    changed_rows["filter_duplicate_decision"] = duplicate_decisions
    changed_rows["filter_length_decision"] = length_decisions
    changed_rows["filter_score_decision"] = score_decisions
    changed_rows["filter_anchor_decision"] = anchor_decisions
    changed_rows["filter_final_decision"] = final_decisions
    changed_rows["filter_reasons"] = reasons

    eval_cols = [
        "url_id",
        "in_filter_scope",
        "duplicate_text_hash",
        "duplicate_cluster_size",
        "text_word_count",
        "doc_relevance_score_num",
        "has_ven_anchor",
        "has_us_primary_token",
        "has_us_primary_pair",
        "has_us_primary",
        "has_relation_secondary",
        "filter_duplicate_decision",
        "filter_length_decision",
        "filter_score_decision",
        "filter_anchor_decision",
        "filter_final_decision",
        "filter_reasons",
    ]
    incoming_eval = changed_rows[eval_cols].rename(columns={"doc_relevance_score_num": "doc_relevance_score"})
    merged_eval = upsert_eval(existing_eval, incoming_eval)

    args.eval.parent.mkdir(parents=True, exist_ok=True)
    merged_eval.to_csv(args.eval, index=False)

    summary_rows = []
    for col in [
        "filter_duplicate_decision",
        "filter_length_decision",
        "filter_score_decision",
        "filter_anchor_decision",
        "filter_final_decision",
    ]:
        vc = merged_eval[col].value_counts(dropna=False)
        for decision, count in vc.items():
            summary_rows.append({"metric": col, "decision": decision, "count": int(count)})
    pd.DataFrame(summary_rows).to_csv(args.summary_output, index=False)

    write_weekly_samples(changed_rows, args.sample_dir, args.sample_size, args.seed)

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "preprocessing" / "plot_filter_stage_score_histograms.py"),
            "--eval-csv",
            str(args.eval),
            "--output",
            str(args.plot_output),
        ],
        check=True,
    )

    print(f"Changed url_ids evaluated: {len(changed_rows):,}")
    print(f"Evaluation table written: {args.eval}")
    print(f"Summary counts written: {args.summary_output}")
    print(f"Weekly samples written: {args.sample_dir}")
    print(f"Histogram written: {args.plot_output}")


if __name__ == "__main__":
    main()
