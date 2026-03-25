from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the duplicate-filter evaluation builder.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    project_dir = Path(__file__).resolve().parents[1]
    artifact_dir = project_dir / "data" / "preprocessing"
    parser = argparse.ArgumentParser(
        description=(
            "Build early duplicate-filter evaluation from url_lookup.csv. "
            "Marks duplicate text clusters without removing rows."
        )
    )
    parser.add_argument("--lookup", type=Path, default=artifact_dir / "url_lookup.csv", help="Path to url_lookup.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_dir / "url_filter_eval.csv",
        help="Output path for url_filter_eval.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=artifact_dir / "url_filter_summary_counts.csv",
        help="Output path for duplicate-only summary counts",
    )
    return parser.parse_args()


def normalize_text_for_hash(text: object) -> str:
    """Normalize text so duplicate detection is robust to case and whitespace differences.
    
    Args:
        text (object): Raw text value from the lookup table.
    
    Returns:
        str: Normalized text used as hash input.
    """
    if text is None or pd.isna(text):
        return ""
    value = str(text).replace("\r", " ").replace("\n", " ")
    return WS_RE.sub(" ", value).strip().lower()


def text_hash(text: str) -> str:
    """Compute a stable SHA-256 hash for normalized text.
    
    Args:
        text (str): Normalized text to hash.
    
    Returns:
        str: Hex-encoded hash string, or an empty string for blank input.
    """
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def upsert_eval(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge new duplicate-evaluation rows into an existing evaluation table by url_id.
    
    Args:
        existing (pd.DataFrame): Existing evaluation DataFrame already on disk.
        incoming (pd.DataFrame): Newly computed evaluation rows for the current run.
    
    Returns:
        pd.DataFrame: Upserted evaluation DataFrame sorted by url_id.
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
    """Build duplicate-filter flags and token-training eligibility from url_lookup.csv.
    
    Returns:
        None: No return value.
    """
    args = parse_args()
    if not args.lookup.exists():
        raise FileNotFoundError(f"Lookup file not found: {args.lookup}")

    print("Loading url_lookup...", flush=True)
    df = pd.read_csv(args.lookup, low_memory=False)
    print(f"  Rows loaded: {len(df):,}", flush=True)

    required_cols = {"url_id", "Text", "Tokens", "Scrape_Status"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"url_lookup missing required columns: {sorted(missing_cols)}")

    success_mask = df["Scrape_Status"].fillna("").astype(str).str.contains("success", case=False)
    nonempty_text_mask = df["Text"].fillna("").astype(str).str.strip() != ""
    nonempty_tokens_mask = df["Tokens"].fillna("").astype(str).str.strip() != ""
    in_scope = success_mask & nonempty_text_mask & nonempty_tokens_mask

    df["in_filter_scope"] = in_scope
    df["duplicate_text_hash"] = ""
    df["duplicate_cluster_size"] = 0
    df["filter_duplicate_decision"] = "out_of_scope"
    df["used_for_token_training"] = False

    scope_idx = df.index[in_scope].tolist()
    print(f"Hashing in-scope texts for duplicate clustering: {len(scope_idx):,}", flush=True)
    if scope_idx:
        norm_text_values = df.loc[scope_idx, "Text"].map(normalize_text_for_hash).tolist()
        hash_values: list[str] = []
        for text in tqdm(norm_text_values, total=len(norm_text_values), desc="Hashing texts", unit="row", file=sys.stdout):
            hash_values.append(text_hash(text))

        hash_series = pd.Series(hash_values, index=scope_idx)
        cluster_sizes = hash_series.map(hash_series.value_counts()).astype(int)
        keep_mask = cluster_sizes == 1

        df.loc[scope_idx, "duplicate_text_hash"] = hash_series
        df.loc[scope_idx, "duplicate_cluster_size"] = cluster_sizes.values
        df.loc[scope_idx, "filter_duplicate_decision"] = np.where(keep_mask.values, "keep", "drop")
        df.loc[scope_idx, "used_for_token_training"] = keep_mask.values

    eval_cols = [
        "url_id",
        "in_filter_scope",
        "duplicate_text_hash",
        "duplicate_cluster_size",
        "filter_duplicate_decision",
        "used_for_token_training",
    ]
    eval_df = df[eval_cols].copy()

    print("Writing duplicate-filter evaluation outputs...", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing_eval = pd.read_csv(args.output, low_memory=False)
        eval_df = upsert_eval(existing_eval, eval_df)
    eval_df.to_csv(args.output, index=False)

    summary_rows = []
    vc_dup = eval_df["filter_duplicate_decision"].value_counts(dropna=False)
    for decision, count in vc_dup.items():
        summary_rows.append({"metric": "filter_duplicate_decision", "decision": decision, "count": int(count)})
    vc_train = eval_df["used_for_token_training"].value_counts(dropna=False)
    for decision, count in vc_train.items():
        summary_rows.append({"metric": "used_for_token_training", "decision": str(decision), "count": int(count)})
    summary_df = pd.DataFrame(summary_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)

    in_scope_count = int(eval_df["in_filter_scope"].sum())
    train_count = int((eval_df["used_for_token_training"] == True).sum())
    drop_count = int((eval_df["filter_duplicate_decision"] == "drop").sum())
    print(f"In-scope rows: {in_scope_count:,}", flush=True)
    print(f"Duplicate-drop rows: {drop_count:,}", flush=True)
    print(f"Rows used for token training: {train_count:,}", flush=True)
    print(f"Evaluation table: {args.output}", flush=True)
    print(f"Summary counts: {args.summary_output}", flush=True)


if __name__ == "__main__":
    main()
