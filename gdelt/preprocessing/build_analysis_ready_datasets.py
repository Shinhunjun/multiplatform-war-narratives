from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_url_index import canonicalize_url
from evaluate_filter_strategy import load_filter_rules


REQUIRED_LOOKUP_COLS = {
    "url_id",
    "SourceURL",
    "SourceURL_Canonical",
    "Title",
    "Text",
    "Tokens",
    "Scrape_Status",
}
REQUIRED_EVAL_COLS = {
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
}
URL_EVENT_JOIN_COLS = [
    "url_id",
    "SourceURL_Canonical",
    "row_count",
    "doc_relevance_sum",
    "doc_relevance_matches",
    "doc_token_count",
    "doc_relevance_score",
    "in_filter_scope",
    "duplicate_cluster_size",
    "text_word_count",
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
    "filter_final_decision_effective",
    "analysis_include",
    "analysis_review_flag",
    "filter_reasons",
]
RAW_CONTENT_COLS = {"Title", "Text"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for analysis-ready parquet exports.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    base = Path(__file__).resolve().parent
    artifact_dir = base.parent / "data" / "preprocessing"
    output_dir = base.parent / "data" / "analysis_ready"
    parser = argparse.ArgumentParser(
        description=(
            "Build analysis-ready parquet exports: one event-level table preserving gdelt_scraped rows "
            "and one url-level content table with filtered Title/Text/Tokens."
        )
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=base.parent / "data" / "gdelt_scraped.csv",
        help="Path to gdelt_scraped.csv",
    )
    parser.add_argument(
        "--lookup",
        type=Path,
        default=artifact_dir / "url_lookup.csv",
        help="Path to url_lookup.csv",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=artifact_dir / "url_filter_eval.csv",
        help="Path to url_filter_eval.csv",
    )
    parser.add_argument(
        "--filter-rules",
        type=Path,
        default=base / "filter_rule_config.json",
        help="Path to filter_rule_config.json",
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        default=output_dir / "analysis_events.parquet",
        help="Output parquet path for event-level analysis table",
    )
    parser.add_argument(
        "--url-output",
        type=Path,
        default=output_dir / "analysis_url_content.parquet",
        help="Output parquet path for url-level content table",
    )
    parser.add_argument(
        "--keep-raw-content-cols",
        action="store_true",
        help="Keep raw Title/Text columns in the event-level table (off by default to save space)",
    )
    return parser.parse_args()


def parse_token_list(value: object) -> list[str]:
    """Parse serialized token arrays from CSV into Python string lists.

    Args:
        value (object): Serialized token array field.

    Returns:
        list[str]: Parsed token list.
    """
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a clear error when required columns are missing.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        required (set[str]): Required column names.
        name (str): Human-readable dataset label.

    Returns:
        None: No return value.
    """
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def validate_unique_key(df: pd.DataFrame, key: str, name: str) -> None:
    """Ensure a DataFrame has a unique non-null key column.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        key (str): Key column name.
        name (str): Human-readable dataset label.

    Returns:
        None: No return value.
    """
    if key not in df.columns:
        raise ValueError(f"{name} missing key column: {key}")
    if df[key].isna().any():
        raise ValueError(f"{name} contains null values in key column: {key}")
    if df[key].duplicated().any():
        dup_count = int(df[key].duplicated().sum())
        raise ValueError(f"{name} contains duplicate {key} values: {dup_count}")


def effective_filter_decision(raw_decision: object, review_handling: str) -> str:
    """Map raw filter decisions to effective analysis decisions.

    Args:
        raw_decision (object): Raw filter decision from url_filter_eval.
        review_handling (str): Review handling policy from filter_rule_config.json.

    Returns:
        str: Effective decision for downstream analysis.
    """
    decision = str(raw_decision).strip().lower()
    if decision != "review":
        return decision
    if review_handling == "include_with_flag":
        return "keep"
    if review_handling == "drop":
        return "drop"
    return "review"


def analysis_include_flag(effective_decision: str) -> object:
    """Convert effective decisions into an inclusion flag for downstream filtering.

    Args:
        effective_decision (str): Effective decision after review-handling policy.

    Returns:
        object: True/False or pd.NA when manual adjudication leaves the row unresolved.
    """
    if effective_decision == "keep":
        return True
    if effective_decision == "drop" or effective_decision == "out_of_scope":
        return False
    return pd.NA


def prepare_url_content_table(
    lookup_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    review_handling: str,
) -> pd.DataFrame:
    """Build the url-level content table with effective analysis decisions.

    Args:
        lookup_df (pd.DataFrame): url_lookup data.
        eval_df (pd.DataFrame): url_filter_eval data.
        review_handling (str): Review handling policy from filter_rule_config.json.

    Returns:
        pd.DataFrame: url-level content DataFrame ready for parquet export.
    """
    lookup = lookup_df.copy()
    eval_only = eval_df.copy()
    lookup_ids = set(pd.to_numeric(lookup["url_id"], errors="raise").astype("Int64").tolist())
    eval_ids = set(pd.to_numeric(eval_only["url_id"], errors="raise").astype("Int64").tolist())
    missing_eval_ids = sorted(lookup_ids - eval_ids)
    extra_eval_ids = sorted(eval_ids - lookup_ids)
    if missing_eval_ids:
        raise ValueError(f"url_filter_eval is missing {len(missing_eval_ids)} url_id values present in url_lookup")
    if extra_eval_ids:
        raise ValueError(f"url_filter_eval contains {len(extra_eval_ids)} url_id values not present in url_lookup")

    if "doc_relevance_score" in lookup.columns and "doc_relevance_score" in eval_only.columns:
        eval_only = eval_only.drop(columns=["doc_relevance_score"])

    url_df = lookup.merge(eval_only, on="url_id", how="left", validate="one_to_one")
    url_df["url_id"] = pd.to_numeric(url_df["url_id"], errors="raise").astype("Int64")
    url_df["Tokens"] = url_df["Tokens"].apply(parse_token_list)

    url_df["filter_final_decision"] = url_df["filter_final_decision"].fillna("out_of_scope").astype(str)
    url_df["filter_final_decision_effective"] = url_df["filter_final_decision"].map(
        lambda value: effective_filter_decision(value, review_handling)
    )
    url_df["analysis_review_flag"] = url_df["filter_final_decision"].eq("review")
    url_df["analysis_include"] = pd.Series(
        [analysis_include_flag(value) for value in url_df["filter_final_decision_effective"]],
        dtype="boolean",
    )
    return url_df


def resolve_event_url_ids(events_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve stable url_id values for event rows via canonical URL mapping.

    Args:
        events_df (pd.DataFrame): Raw event-level gdelt_scraped data.
        lookup_df (pd.DataFrame): url_lookup data with canonical URL mapping.

    Returns:
        pd.DataFrame: Event DataFrame with resolved url_id and SourceURL_Canonical columns.
    """
    events = events_df.copy()
    lookup_ids = lookup_df[["SourceURL_Canonical", "url_id"]].copy()
    lookup_ids["SourceURL_Canonical"] = lookup_ids["SourceURL_Canonical"].fillna("").astype(str)
    validate_unique_key(lookup_ids, "SourceURL_Canonical", "url_lookup canonical mapping")

    events["SourceURL"] = events["SourceURL"].fillna("").astype(str)
    if "SourceURL_Canonical" not in events.columns:
        events["SourceURL_Canonical"] = events["SourceURL"].map(canonicalize_url)
    else:
        events["SourceURL_Canonical"] = events["SourceURL_Canonical"].fillna("").astype(str)
        blank_mask = events["SourceURL_Canonical"].eq("")
        if blank_mask.any():
            events.loc[blank_mask, "SourceURL_Canonical"] = events.loc[blank_mask, "SourceURL"].map(canonicalize_url)

    resolved = events.merge(
        lookup_ids.rename(columns={"url_id": "url_id_lookup"}),
        on="SourceURL_Canonical",
        how="left",
        validate="many_to_one",
    )

    if "url_id" in resolved.columns:
        existing_ids = pd.to_numeric(resolved["url_id"], errors="coerce").astype("Int64")
        lookup_match = pd.to_numeric(resolved["url_id_lookup"], errors="coerce").astype("Int64")
        mismatch_mask = existing_ids.notna() & lookup_match.notna() & (existing_ids != lookup_match)
        if mismatch_mask.any():
            mismatch_count = int(mismatch_mask.sum())
            raise ValueError(
                f"Event dataset contains {mismatch_count} url_id values that conflict with lookup canonical mapping"
            )

    resolved["url_id"] = pd.to_numeric(resolved["url_id_lookup"], errors="coerce").astype("Int64")
    resolved = resolved.drop(columns=["url_id_lookup"])

    missing_mask = resolved["url_id"].isna()
    if missing_mask.any():
        missing_count = int(missing_mask.sum())
        raise ValueError(f"Unable to resolve url_id for {missing_count} event rows from SourceURL canonical mapping")
    return resolved


def prepare_event_table(
    events_df: pd.DataFrame,
    url_df: pd.DataFrame,
    keep_raw_content_cols: bool,
) -> pd.DataFrame:
    """Build the event-level analysis table while avoiding duplicated large content fields.

    Args:
        events_df (pd.DataFrame): Raw event-level gdelt_scraped data with resolved url_id.
        url_df (pd.DataFrame): url-level content table with filter metadata.
        keep_raw_content_cols (bool): Whether to keep raw Title/Text columns in the event table.

    Returns:
        pd.DataFrame: Event-level analysis DataFrame ready for parquet export.
    """
    event_df = events_df.copy()
    if not keep_raw_content_cols:
        drop_cols = [col for col in RAW_CONTENT_COLS if col in event_df.columns]
        if drop_cols:
            event_df = event_df.drop(columns=drop_cols)

    join_cols = [
        col
        for col in URL_EVENT_JOIN_COLS
        if col in url_df.columns and (col == "url_id" or col not in event_df.columns)
    ]
    event_df = event_df.merge(url_df[join_cols], on="url_id", how="left", validate="many_to_one")
    event_df["analysis_include"] = event_df["analysis_include"].astype("boolean")
    return event_df


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet with a helpful dependency error if needed.

    Args:
        df (pd.DataFrame): DataFrame to write.
        path (Path): Output parquet path.

    Returns:
        None: No return value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Writing parquet requires an engine such as pyarrow. Install preprocessing requirements with pyarrow."
        ) from exc


def main() -> None:
    """Build event-level and url-level analysis-ready parquet datasets.

    Returns:
        None: No return value.
    """
    args = parse_args()

    for required_path, label in (
        (args.events, "events input"),
        (args.lookup, "url lookup"),
        (args.eval, "filter eval"),
        (args.filter_rules, "filter rules"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"{label} file not found: {required_path}")

    print("Loading source datasets...", flush=True)
    events_df = pd.read_csv(args.events, low_memory=False)
    lookup_df = pd.read_csv(args.lookup, low_memory=False)
    eval_df = pd.read_csv(args.eval, low_memory=False)

    require_columns(events_df, {"SourceURL"}, "gdelt_scraped")
    require_columns(lookup_df, REQUIRED_LOOKUP_COLS, "url_lookup")
    require_columns(eval_df, REQUIRED_EVAL_COLS, "url_filter_eval")
    validate_unique_key(lookup_df, "url_id", "url_lookup")
    validate_unique_key(eval_df, "url_id", "url_filter_eval")

    print("Loading filter-rule config...", flush=True)
    rule_cfg = load_filter_rules(args.filter_rules)
    review_handling = rule_cfg["review_handling"]
    print(f"  review_handling = {review_handling}", flush=True)

    print("Resolving event-level url_id values...", flush=True)
    resolved_events = resolve_event_url_ids(events_df, lookup_df)

    print("Building url-level content table...", flush=True)
    url_df = prepare_url_content_table(lookup_df, eval_df, review_handling)

    print("Building event-level analysis table...", flush=True)
    event_df = prepare_event_table(resolved_events, url_df, args.keep_raw_content_cols)
    if len(event_df) != len(events_df):
        raise ValueError("Event-level analysis table row count changed unexpectedly")

    print("Writing parquet outputs...", flush=True)
    write_parquet(event_df, args.events_output)
    write_parquet(url_df, args.url_output)

    included_urls = int(url_df["analysis_include"].fillna(False).sum())
    review_urls = int(url_df["analysis_review_flag"].sum())
    print(f"Event rows written: {len(event_df):,}", flush=True)
    print(f"URL-content rows written: {len(url_df):,}", flush=True)
    print(f"Included URL rows under current policy: {included_urls:,}", flush=True)
    print(f"Raw review URL rows: {review_urls:,}", flush=True)
    print(f"Event parquet: {args.events_output}", flush=True)
    print(f"URL-content parquet: {args.url_output}", flush=True)


if __name__ == "__main__":
    main()
