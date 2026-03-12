from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


WORD_RE = re.compile(r"[A-Za-z0-9']+")
WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for full filtering-strategy evaluation.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate filtering strategy on url_lookup.csv, write url_filter_eval.csv, and "
            "export step-by-step sample CSVs for manual QA."
        )
    )
    parser.add_argument("--lookup", type=Path, default=base / "url_lookup.csv", help="Path to url_lookup.csv")
    parser.add_argument(
        "--anchors",
        type=Path,
        default=base / "anchor_token_sets.json",
        help="Path to anchor_token_sets.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "url_filter_eval.csv",
        help="Output path for filter evaluation table",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=base / "url_filter_summary_counts.csv",
        help="Output path for summary counts",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=base / "filter_samples",
        help="Directory for step sample CSV files",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Sample size per decision bucket in each sample file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    return parser.parse_args()


def parse_token_set(value: object) -> set[str]:
    """Parse a serialized token field into a lowercase token set.
    
    Args:
        value (object): Serialized token value from CSV.
    
    Returns:
        set[str]: Parsed token set.
    """
    if value is None or pd.isna(value):
        return set()
    s = str(value).strip()
    if not s:
        return set()
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return {str(x).strip().lower() for x in parsed if str(x).strip()}
    if "," in s:
        return {part.strip().lower() for part in s.split(",") if part.strip()}
    return {s.lower()}


def count_words(text: object) -> int:
    """Count lexical tokens in text using the configured word regex.
    
    Args:
        text (object): Raw text value.
    
    Returns:
        int: Word-count estimate used by the length filter.
    """
    if text is None or pd.isna(text):
        return 0
    return len(WORD_RE.findall(str(text)))


def normalize_text_for_hash(text: object) -> str:
    """Normalize text for stable duplicate hashing in filter evaluation.
    
    Args:
        text (object): Raw text value.
    
    Returns:
        str: Whitespace-normalized lowercase text.
    """
    if text is None or pd.isna(text):
        return ""
    value = str(text).replace("\r", " ").replace("\n", " ")
    return WS_RE.sub(" ", value).strip().lower()


def text_hash(text: str) -> str:
    """Compute SHA-256 hash for normalized text.
    
    Args:
        text (str): Normalized text string.
    
    Returns:
        str: Hex digest for duplicate clustering.
    """
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def decision_duplicate(cluster_size: int) -> str:
    """Assign duplicate-stage decision from duplicate cluster size.
    
    Args:
        cluster_size (int): Size of the duplicate text cluster for a row.
    
    Returns:
        str: Duplicate-stage decision label (drop/review/keep).
    """
    if cluster_size > 1:
        return "drop"
    return "keep"


def decision_length(word_count: int) -> str:
    """Assign length-stage decision from tokenized word count.
    
    Args:
        word_count (int): Word count for the article text.
    
    Returns:
        str: Length-stage decision label (drop/review/keep).
    """
    if word_count < 40:
        return "drop"
    if word_count < 80:
        return "review"
    return "keep"


def decision_score(score: float) -> str:
    """Assign score-stage decision from document relevance score.
    
    Args:
        score (float): Document relevance score.
    
    Returns:
        str: Score-stage decision label (drop/review/keep).
    """
    if score < 25:
        return "drop"
    if score < 40:
        return "review"
    return "keep"


def decision_anchor(has_ven: bool, has_us_primary: bool, has_relation_secondary: bool) -> str:
    """Assign anchor-stage decision from Venezuela/US/relation anchor signals.
    
    Args:
        has_ven (bool): Whether Venezuela-anchor terms are present.
        has_us_primary (bool): Whether US-primary anchor signal is present.
        has_relation_secondary (bool): Whether relation-context anchors are present.
    
    Returns:
        str: Anchor-stage decision label (drop/review/keep).
    """
    if not has_ven:
        return "drop"
    if has_us_primary:
        return "keep"
    if has_relation_secondary:
        return "review"
    return "drop"


def final_decision(dup_dec: str, length_dec: str, score_dec: str, anchor_dec: str) -> str:
    """Combine stage decisions into a single final filter decision.
    
    Args:
        dup_dec (str): Duplicate-stage decision.
        length_dec (str): Length-stage decision.
        score_dec (str): Score-stage decision.
        anchor_dec (str): Anchor-stage decision.
    
    Returns:
        str: Final decision label for the row.
    """
    decisions = [dup_dec, length_dec, score_dec, anchor_dec]
    if "drop" in decisions:
        return "drop"
    if "review" in decisions:
        return "review"
    return "keep"


def reasons_for_row(dup_dec: str, length_dec: str, score_dec: str, anchor_dec: str, in_scope: bool) -> str:
    """Build a pipe-delimited reason label describing why a row was dropped/reviewed/kept.
    
    Args:
        dup_dec (str): Duplicate-stage decision.
        length_dec (str): Length-stage decision.
        score_dec (str): Score-stage decision.
        anchor_dec (str): Anchor-stage decision.
        in_scope (bool): Whether the row is in filtering scope.
    
    Returns:
        str: Human-readable reason code string.
    """
    if not in_scope:
        return "out_of_scope"
    reasons: list[str] = []
    if dup_dec != "keep":
        reasons.append(f"duplicate_{dup_dec}")
    if length_dec != "keep":
        reasons.append(f"length_{length_dec}")
    if score_dec != "keep":
        reasons.append(f"score_{score_dec}")
    if anchor_dec != "keep":
        reasons.append(f"anchor_{anchor_dec}")
    return "|".join(reasons) if reasons else "pass_all"


def stratified_sample(df: pd.DataFrame, decision_col: str, sample_size: int, seed: int) -> pd.DataFrame:
    """Draw a stratified sample by decision label for manual QA outputs.
    
    Args:
        df (pd.DataFrame): Input DataFrame to sample from.
        decision_col (str): Column containing decision labels for stratification.
        sample_size (int): Maximum rows per decision bucket.
        seed (int): Random seed for reproducible sampling.
    
    Returns:
        pd.DataFrame: Sampled DataFrame preserving label balance up to the requested cap.
    """
    parts = []
    for idx, decision in enumerate(sorted(df[decision_col].dropna().unique().tolist())):
        group = df[df[decision_col] == decision]
        if group.empty:
            continue
        n = min(sample_size, len(group))
        parts.append(group.sample(n=n, random_state=seed + idx))
    if not parts:
        return df.head(0).copy()
    return pd.concat(parts, ignore_index=True)


def upsert_eval(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Upsert newly evaluated rows into an existing url_filter_eval table by url_id.
    
    Args:
        existing (pd.DataFrame): Existing evaluation DataFrame on disk.
        incoming (pd.DataFrame): Newly computed evaluation rows.
    
    Returns:
        pd.DataFrame: Merged evaluation DataFrame sorted by url_id.
    """
    if existing.empty:
        out = incoming.copy()
        out["url_id"] = pd.to_numeric(out["url_id"], errors="coerce").astype("Int64")
        return out.sort_values("url_id").reset_index(drop=True)

    out = existing.copy()
    inc = incoming.copy()

    out["url_id"] = pd.to_numeric(out["url_id"], errors="coerce").astype("Int64")
    inc["url_id"] = pd.to_numeric(inc["url_id"], errors="coerce").astype("Int64")

    for col in inc.columns:
        if col not in out.columns:
            out[col] = pd.NA

    out = out.drop_duplicates(subset=["url_id"], keep="last").set_index("url_id", drop=False)
    inc = inc.drop_duplicates(subset=["url_id"], keep="last").set_index("url_id", drop=False)

    overlap_ids = out.index.intersection(inc.index)
    if len(overlap_ids) > 0:
        for col in inc.columns:
            out.loc[overlap_ids, col] = inc.loc[overlap_ids, col].values

    new_ids = inc.index.difference(out.index)
    if len(new_ids) > 0:
        out = pd.concat([out, inc.loc[new_ids]], axis=0)

    out = out.sort_index().reset_index(drop=True)
    out["url_id"] = pd.to_numeric(out["url_id"], errors="coerce").astype("Int64")
    return out


def main() -> None:
    """Run end-to-end filter evaluation, write row-level outputs, summary counts, and QA samples.
    
    Returns:
        None: No return value.
    """
    args = parse_args()

    if not args.lookup.exists():
        raise FileNotFoundError(f"Lookup file not found: {args.lookup}")
    if not args.anchors.exists():
        raise FileNotFoundError(f"Anchor config file not found: {args.anchors}")

    print("Loading url_lookup...", flush=True)
    df = pd.read_csv(args.lookup, low_memory=False)
    print(f"  Rows loaded: {len(df):,}", flush=True)

    required_cols = {"url_id", "SourceURL", "Title", "Text", "Tokens", "Scrape_Status", "doc_relevance_score"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"url_lookup missing required columns: {sorted(missing_cols)}")

    print("Loading anchor token sets...", flush=True)
    anchor_cfg = json.loads(args.anchors.read_text(encoding="utf-8"))
    anchors = anchor_cfg.get("anchors", {})
    ven_primary = set(anchors.get("venezuela_primary", []))
    us_primary = set(anchors.get("us_primary", []))
    us_pairs = [tuple(pair) for pair in anchors.get("us_primary_token_pairs", []) if len(pair) >= 2]
    relation_secondary = set(anchors.get("relation_context_secondary", []))

    print("Preparing scope and feature columns...", flush=True)
    score = pd.to_numeric(df["doc_relevance_score"], errors="coerce")
    success_mask = df["Scrape_Status"].fillna("").astype(str).str.contains("success", case=False)
    nonempty_text_mask = df["Text"].fillna("").astype(str).str.strip() != ""
    nonempty_tokens_mask = df["Tokens"].fillna("").astype(str).str.strip() != ""
    in_scope = success_mask & nonempty_text_mask & nonempty_tokens_mask & score.notna()

    df["in_filter_scope"] = in_scope
    df["text_word_count"] = df["Text"].apply(count_words)
    df["doc_relevance_score_num"] = score

    token_sets = [set() for _ in range(len(df))]
    scope_idx = df.index[in_scope].tolist()
    print(f"  Parsing token arrays for scope rows: {len(scope_idx):,}", flush=True)
    for i in tqdm(scope_idx, total=len(scope_idx), desc="Parsing token arrays", unit="row", file=sys.stdout):
        token_sets[i] = parse_token_set(df.at[i, "Tokens"])

    print("Computing duplicate text clusters...", flush=True)
    df["duplicate_text_hash"] = ""
    df["duplicate_cluster_size"] = 0
    if scope_idx:
        norm_scope_text = df.loc[scope_idx, "Text"].map(normalize_text_for_hash)
        hash_scope = norm_scope_text.map(text_hash)
        cluster_sizes = hash_scope.map(hash_scope.value_counts()).fillna(0).astype(int)
        df.loc[scope_idx, "duplicate_text_hash"] = hash_scope
        df.loc[scope_idx, "duplicate_cluster_size"] = cluster_sizes

    print("Applying anchor logic...", flush=True)
    has_ven = [False] * len(df)
    has_us_primary_token = [False] * len(df)
    has_us_primary_pair = [False] * len(df)
    has_us_primary = [False] * len(df)
    has_relation = [False] * len(df)

    for i in tqdm(scope_idx, total=len(scope_idx), desc="Evaluating anchor signals", unit="row", file=sys.stdout):
        tok = token_sets[i]
        h_ven = bool(tok & ven_primary)
        h_us_token = bool(tok & us_primary)
        h_us_pair = any(all(t in tok for t in pair) for pair in us_pairs)
        h_us = h_us_token or h_us_pair
        h_rel = bool(tok & relation_secondary)

        has_ven[i] = h_ven
        has_us_primary_token[i] = h_us_token
        has_us_primary_pair[i] = h_us_pair
        has_us_primary[i] = h_us
        has_relation[i] = h_rel

    df["has_ven_anchor"] = has_ven
    df["has_us_primary_token"] = has_us_primary_token
    df["has_us_primary_pair"] = has_us_primary_pair
    df["has_us_primary"] = has_us_primary
    df["has_relation_secondary"] = has_relation

    print("Computing step decisions...", flush=True)
    df["filter_duplicate_decision"] = "out_of_scope"
    df["filter_length_decision"] = "out_of_scope"
    df["filter_score_decision"] = "out_of_scope"
    df["filter_anchor_decision"] = "out_of_scope"
    df["filter_final_decision"] = "out_of_scope"

    for i in tqdm(scope_idx, total=len(scope_idx), desc="Applying filter rules", unit="row", file=sys.stdout):
        duplicate_dec = decision_duplicate(int(df.at[i, "duplicate_cluster_size"]))
        length_dec = decision_length(int(df.at[i, "text_word_count"]))
        score_dec = decision_score(float(df.at[i, "doc_relevance_score_num"]))
        anchor_dec = decision_anchor(
            bool(df.at[i, "has_ven_anchor"]),
            bool(df.at[i, "has_us_primary"]),
            bool(df.at[i, "has_relation_secondary"]),
        )
        final_dec = final_decision(duplicate_dec, length_dec, score_dec, anchor_dec)

        df.at[i, "filter_duplicate_decision"] = duplicate_dec
        df.at[i, "filter_length_decision"] = length_dec
        df.at[i, "filter_score_decision"] = score_dec
        df.at[i, "filter_anchor_decision"] = anchor_dec
        df.at[i, "filter_final_decision"] = final_dec

    print("Building row-level filter reasons...", flush=True)
    reason_values: list[str] = []
    for i in tqdm(range(len(df)), total=len(df), desc="Building reason labels", unit="row", file=sys.stdout):
        reason_values.append(
            reasons_for_row(
                str(df.at[i, "filter_duplicate_decision"]),
                str(df.at[i, "filter_length_decision"]),
                str(df.at[i, "filter_score_decision"]),
                str(df.at[i, "filter_anchor_decision"]),
                bool(df.at[i, "in_filter_scope"]),
            )
        )
    df["filter_reasons"] = reason_values

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
    eval_df = df[eval_cols].rename(columns={"doc_relevance_score_num": "doc_relevance_score"})

    print("Writing evaluation outputs...", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing_eval = pd.read_csv(args.output, low_memory=False)
        eval_df = upsert_eval(existing_eval, eval_df)
    eval_df.to_csv(args.output, index=False)

    # Summary counts for quick audits.
    summary_rows = []
    for col in [
        "filter_duplicate_decision",
        "filter_length_decision",
        "filter_score_decision",
        "filter_anchor_decision",
        "filter_final_decision",
    ]:
        vc = eval_df[col].value_counts(dropna=False)
        for decision, count in vc.items():
            summary_rows.append({"metric": col, "decision": decision, "count": int(count)})
    summary_df = pd.DataFrame(summary_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)

    print("Writing step sample files...", flush=True)
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

    args.sample_dir.mkdir(parents=True, exist_ok=True)

    s0 = stratified_sample(step0_df, "filter_duplicate_decision", args.sample_size, args.seed)
    s1 = stratified_sample(step1_df, "filter_length_decision", args.sample_size, args.seed)
    s2 = stratified_sample(step2_df, "filter_score_decision", args.sample_size, args.seed)
    s3 = stratified_sample(step3_df, "filter_anchor_decision", args.sample_size, args.seed)
    sf = stratified_sample(scope_df, "filter_final_decision", args.sample_size, args.seed)

    def save_sample(sample_df: pd.DataFrame, out_name: str) -> None:
        """Write a sample DataFrame with review columns to the configured sample directory.
        
        Args:
            sample_df (pd.DataFrame): Sample rows to write.
            out_name (str): Output filename within the sample directory.
        
        Returns:
            None: No return value.
        """
        out = sample_df[sample_cols].rename(
            columns={"SourceURL": "url", "doc_relevance_score_num": "doc_relevance_score"}
        )
        out.to_csv(args.sample_dir / out_name, index=False)

    save_sample(s0, "sample_step0_duplicate.csv")
    save_sample(s1, "sample_step1_length.csv")
    save_sample(s2, "sample_step2_score.csv")
    save_sample(s3, "sample_step3_anchor.csv")
    save_sample(sf, "sample_final_decision.csv")

    print(f"Evaluation table: {args.output}", flush=True)
    print(f"Summary counts: {args.summary_output}", flush=True)
    print(f"Sample directory: {args.sample_dir}", flush=True)
    print("Decision counts (final):", flush=True)
    print(eval_df["filter_final_decision"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
