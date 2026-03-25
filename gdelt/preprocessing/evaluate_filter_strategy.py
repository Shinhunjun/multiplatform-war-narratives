from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


WORD_RE = re.compile(r"[A-Za-z0-9']+")
WS_RE = re.compile(r"\s+")
DECISION_LABELS = {"drop", "review", "keep"}
DEFAULT_DECISION_PRIORITY = ("drop", "review", "keep")
ALLOWED_REVIEW_HANDLING = {"drop", "manual_adjudication", "include_with_flag"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for full filtering-strategy evaluation.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    base = Path(__file__).resolve().parent
    artifact_dir = base.parent / "data" / "preprocessing"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate filtering strategy on url_lookup.csv, write url_filter_eval.csv, and "
            "export step-by-step sample CSVs for manual QA."
        )
    )
    parser.add_argument("--lookup", type=Path, default=artifact_dir / "url_lookup.csv", help="Path to url_lookup.csv")
    parser.add_argument(
        "--anchors",
        type=Path,
        default=base / "anchor_token_sets.json",
        help="Path to anchor_token_sets.json",
    )
    parser.add_argument(
        "--filter-rules",
        type=Path,
        default=base / "filter_rule_config.json",
        help="Path to filter_rule_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_dir / "url_filter_eval.csv",
        help="Output path for filter evaluation table",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=artifact_dir / "url_filter_summary_counts.csv",
        help="Output path for summary counts",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=artifact_dir / "filter_samples",
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


def _require_bool(section: str, key: str, value: object) -> bool:
    """Require a strict boolean config value and raise a clear error otherwise.
    
    Args:
        section (str): Parent config section label.
        key (str): Config key name.
        value (object): Raw config value.
    
    Returns:
        bool: Validated boolean value.
    """
    if isinstance(value, bool):
        return value
    raise ValueError(f"{section}.{key} must be a boolean")


def _require_int(section: str, key: str, value: object) -> int:
    """Require an integer config value and raise a clear error otherwise.
    
    Args:
        section (str): Parent config section label.
        key (str): Config key name.
        value (object): Raw config value.
    
    Returns:
        int: Validated integer value.
    """
    if isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"{section}.{key} must be an integer")


def _require_float(section: str, key: str, value: object) -> float:
    """Require a numeric config value and return it as float.
    
    Args:
        section (str): Parent config section label.
        key (str): Config key name.
        value (object): Raw config value.
    
    Returns:
        float: Validated numeric value as float.
    """
    if isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be numeric")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{section}.{key} must be numeric")


def load_filter_rules(path: Path) -> dict[str, Any]:
    """Load and validate filter-rule configuration from JSON.
    
    Args:
        path (Path): Path to JSON filter-rule config file.
    
    Returns:
        dict[str, Any]: Normalized config dictionary.
    """
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Filter rule config root must be an object")

    scope_raw = cfg.get("scope")
    if not isinstance(scope_raw, dict):
        raise ValueError("scope section is required and must be an object")

    thresholds_raw = cfg.get("thresholds")
    if not isinstance(thresholds_raw, dict):
        raise ValueError("thresholds section is required and must be an object")

    scope = {
        "require_success_status": _require_bool(
            "scope",
            "require_success_status",
            scope_raw.get("require_success_status"),
        ),
        "require_nonempty_text": _require_bool(
            "scope",
            "require_nonempty_text",
            scope_raw.get("require_nonempty_text"),
        ),
        "require_nonempty_tokens": _require_bool(
            "scope",
            "require_nonempty_tokens",
            scope_raw.get("require_nonempty_tokens"),
        ),
        "require_numeric_score": _require_bool(
            "scope",
            "require_numeric_score",
            scope_raw.get("require_numeric_score"),
        ),
    }

    thresholds = {
        "duplicate_drop_cluster_size_gt": _require_int(
            "thresholds",
            "duplicate_drop_cluster_size_gt",
            thresholds_raw.get("duplicate_drop_cluster_size_gt"),
        ),
        "length_drop_lt": _require_int(
            "thresholds",
            "length_drop_lt",
            thresholds_raw.get("length_drop_lt"),
        ),
        "length_review_lt": _require_int(
            "thresholds",
            "length_review_lt",
            thresholds_raw.get("length_review_lt"),
        ),
        "score_drop_lt": _require_float(
            "thresholds",
            "score_drop_lt",
            thresholds_raw.get("score_drop_lt"),
        ),
        "score_review_lt": _require_float(
            "thresholds",
            "score_review_lt",
            thresholds_raw.get("score_review_lt"),
        ),
    }

    if thresholds["duplicate_drop_cluster_size_gt"] < 0:
        raise ValueError("thresholds.duplicate_drop_cluster_size_gt must be >= 0")
    if thresholds["length_drop_lt"] < 0:
        raise ValueError("thresholds.length_drop_lt must be >= 0")
    if thresholds["length_review_lt"] <= thresholds["length_drop_lt"]:
        raise ValueError("thresholds.length_review_lt must be > thresholds.length_drop_lt")
    if thresholds["score_review_lt"] <= thresholds["score_drop_lt"]:
        raise ValueError("thresholds.score_review_lt must be > thresholds.score_drop_lt")

    priority_raw = cfg.get("final_decision_priority", list(DEFAULT_DECISION_PRIORITY))
    if not isinstance(priority_raw, list) or len(priority_raw) != 3:
        raise ValueError("final_decision_priority must be a 3-item list")
    priority = tuple(str(x).strip().lower() for x in priority_raw)
    if set(priority) != DECISION_LABELS:
        raise ValueError("final_decision_priority must contain exactly: drop, review, keep")

    review_handling = str(cfg.get("review_handling", "include_with_flag")).strip().lower()
    if review_handling not in ALLOWED_REVIEW_HANDLING:
        allowed = ", ".join(sorted(ALLOWED_REVIEW_HANDLING))
        raise ValueError(f"review_handling must be one of: {allowed}")

    return {
        "version": str(cfg.get("version", "")).strip(),
        "scope": scope,
        "thresholds": thresholds,
        "final_decision_priority": priority,
        "review_handling": review_handling,
    }


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


def decision_duplicate(cluster_size: int, drop_cluster_size_gt: int = 1) -> str:
    """Assign duplicate-stage decision from duplicate cluster size.
    
    Args:
        cluster_size (int): Size of the duplicate text cluster for a row.
        drop_cluster_size_gt (int): Drop when cluster size is greater than this value. Defaults to 1.
    
    Returns:
        str: Duplicate-stage decision label (drop/review/keep).
    """
    if cluster_size > drop_cluster_size_gt:
        return "drop"
    return "keep"


def decision_length(word_count: int, drop_lt: int = 40, review_lt: int = 80) -> str:
    """Assign length-stage decision from tokenized word count.
    
    Args:
        word_count (int): Word count for the article text.
        drop_lt (int): Drop threshold for word count. Defaults to 40.
        review_lt (int): Review threshold for word count. Defaults to 80.
    
    Returns:
        str: Length-stage decision label (drop/review/keep).
    """
    if word_count < drop_lt:
        return "drop"
    if word_count < review_lt:
        return "review"
    return "keep"


def decision_score(score: float, drop_lt: float = 25.0, review_lt: float = 40.0) -> str:
    """Assign score-stage decision from document relevance score.
    
    Args:
        score (float): Document relevance score.
        drop_lt (float): Drop threshold for relevance score. Defaults to 25.
        review_lt (float): Review threshold for relevance score. Defaults to 40.
    
    Returns:
        str: Score-stage decision label (drop/review/keep).
    """
    if score < drop_lt:
        return "drop"
    if score < review_lt:
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


def final_decision(
    dup_dec: str,
    length_dec: str,
    score_dec: str,
    anchor_dec: str,
    priority: tuple[str, str, str] = DEFAULT_DECISION_PRIORITY,
) -> str:
    """Combine stage decisions into a single final filter decision.
    
    Args:
        dup_dec (str): Duplicate-stage decision.
        length_dec (str): Length-stage decision.
        score_dec (str): Score-stage decision.
        anchor_dec (str): Anchor-stage decision.
        priority (tuple[str, str, str]): Ordered precedence for final decision labels.
    
    Returns:
        str: Final decision label for the row.
    """
    decisions = [dup_dec, length_dec, score_dec, anchor_dec]
    for label in priority:
        if label in decisions:
            return label
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
    if not args.filter_rules.exists():
        raise FileNotFoundError(f"Filter rule config file not found: {args.filter_rules}")

    print("Loading url_lookup...", flush=True)
    df = pd.read_csv(args.lookup, low_memory=False)
    print(f"  Rows loaded: {len(df):,}", flush=True)

    required_cols = {"url_id", "SourceURL", "Title", "Text", "Tokens", "Scrape_Status", "doc_relevance_score"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"url_lookup missing required columns: {sorted(missing_cols)}")

    print("Loading filter-rule config...", flush=True)
    rule_cfg = load_filter_rules(args.filter_rules)
    thresholds = rule_cfg["thresholds"]
    scope_cfg = rule_cfg["scope"]
    final_priority = rule_cfg["final_decision_priority"]

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
    in_scope = pd.Series(True, index=df.index, dtype=bool)
    if scope_cfg["require_success_status"]:
        in_scope &= success_mask
    if scope_cfg["require_nonempty_text"]:
        in_scope &= nonempty_text_mask
    if scope_cfg["require_nonempty_tokens"]:
        in_scope &= nonempty_tokens_mask
    if scope_cfg["require_numeric_score"]:
        in_scope &= score.notna()

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
        duplicate_dec = decision_duplicate(
            int(df.at[i, "duplicate_cluster_size"]),
            drop_cluster_size_gt=thresholds["duplicate_drop_cluster_size_gt"],
        )
        length_dec = decision_length(
            int(df.at[i, "text_word_count"]),
            drop_lt=thresholds["length_drop_lt"],
            review_lt=thresholds["length_review_lt"],
        )
        score_dec = decision_score(
            float(df.at[i, "doc_relevance_score_num"]),
            drop_lt=thresholds["score_drop_lt"],
            review_lt=thresholds["score_review_lt"],
        )
        anchor_dec = decision_anchor(
            bool(df.at[i, "has_ven_anchor"]),
            bool(df.at[i, "has_us_primary"]),
            bool(df.at[i, "has_relation_secondary"]),
        )
        final_dec = final_decision(
            duplicate_dec,
            length_dec,
            score_dec,
            anchor_dec,
            priority=final_priority,
        )

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
