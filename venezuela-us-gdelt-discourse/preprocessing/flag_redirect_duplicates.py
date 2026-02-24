from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Flag likely redirect/fallback content via duplicate normalized Text hashes."
    )
    parser.add_argument("--lookup", type=Path, default=base / "url_lookup.csv", help="Path to url_lookup.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to updated url_lookup output (default: overwrite --lookup)",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=base / "redirect_duplicate_clusters.csv",
        help="Path to cluster review CSV output",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum distinct URL count for a duplicate-text cluster to be flagged",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=200,
        help="Minimum normalized text length to consider for hashing/flagging",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_domain(url: object) -> str:
    if url is None or pd.isna(url):
        return ""
    raw = str(url).strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        if parsed.netloc:
            return parsed.netloc.lower()
        retry = urlsplit("http://" + raw)
        return retry.netloc.lower()
    except Exception:
        return ""


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output is not None else args.lookup

    if not args.lookup.exists():
        raise FileNotFoundError(f"Lookup file not found: {args.lookup}")

    df = pd.read_csv(args.lookup, low_memory=False)
    for col in ["Text", "SourceURL"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    text_norm = df["Text"].map(normalize_text)
    text_len = text_norm.str.len()
    valid_text_mask = text_len >= args.min_text_length

    df["text_len_norm"] = text_len
    df["text_hash"] = ""
    df.loc[valid_text_mask, "text_hash"] = text_norm[valid_text_mask].map(stable_text_hash)
    df["domain"] = df["SourceURL"].map(extract_domain)

    valid_df = df[valid_text_mask & (df["text_hash"] != "")].copy()

    # Distinct URL counts per hash cluster.
    hash_url_counts = (
        valid_df.groupby("text_hash")["SourceURL"]
        .nunique(dropna=True)
        .rename("text_hash_url_count")
    )

    # Distinct domain counts per hash cluster (helpful for diagnosing boilerplate vs redirect issues).
    hash_domain_counts = (
        valid_df.groupby("text_hash")["domain"]
        .nunique(dropna=True)
        .rename("text_hash_domain_count")
    )

    df = df.merge(hash_url_counts, on="text_hash", how="left")
    df = df.merge(hash_domain_counts, on="text_hash", how="left")
    df["text_hash_url_count"] = df["text_hash_url_count"].fillna(0).astype(int)
    df["text_hash_domain_count"] = df["text_hash_domain_count"].fillna(0).astype(int)

    df["suspect_redirect_content"] = (
        (df["text_len_norm"] >= args.min_text_length)
        & (df["text_hash_url_count"] >= args.min_cluster_size)
    )

    # Build a review table of suspicious clusters with representative rows.
    suspect = df[df["suspect_redirect_content"]].copy()
    cluster_summary = []
    if not suspect.empty:
        grouped = suspect.groupby("text_hash", as_index=False)
        for _, g in grouped:
            g = g.sort_values(by=["text_hash_url_count", "url_id"], ascending=[False, True])
            first = g.iloc[0]
            cluster_summary.append(
                {
                    "text_hash": first["text_hash"],
                    "cluster_url_count": int(first["text_hash_url_count"]),
                    "cluster_domain_count": int(first["text_hash_domain_count"]),
                    "sample_url": first.get("SourceURL", ""),
                    "sample_domain": first.get("domain", ""),
                    "sample_title": first.get("Title", ""),
                    "sample_text_prefix": str(first.get("Text", ""))[:300].replace("\n", " ").replace("\r", " "),
                }
            )

    review_df = pd.DataFrame(cluster_summary).sort_values(
        by=["cluster_url_count", "cluster_domain_count"], ascending=[False, False]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    review_df.to_csv(args.review_output, index=False)

    print(f"Rows processed: {len(df):,}")
    print(f"Rows with sufficient text length: {int(valid_text_mask.sum()):,}")
    print(f"Flagged suspect rows: {int(df['suspect_redirect_content'].sum()):,}")
    print(f"Flagged clusters: {len(review_df):,}")
    print(f"Updated lookup written: {output_path}")
    print(f"Review clusters written: {args.review_output}")


if __name__ == "__main__":
    main()
