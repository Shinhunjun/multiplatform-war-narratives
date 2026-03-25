from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for duplicate-article removal.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    base_dir = Path(__file__).resolve().parents[1]
    default_lookup = base_dir / "data" / "preprocessing" / "url_lookup.csv"
    default_gdelt = base_dir / "data" / "gdelt_scraped.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Remove duplicate articles by normalized full Text, keep one URL per duplicate-text "
            "cluster, and overwrite url_lookup.csv and gdelt_scraped.csv."
        )
    )
    parser.add_argument("--lookup", type=Path, default=default_lookup, help="Path to url_lookup.csv")
    parser.add_argument("--gdelt", type=Path, default=default_gdelt, help="Path to gdelt_scraped.csv")
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=1,
        help="Minimum normalized text length to include in duplicate detection",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    """Normalize text for exact duplicate detection across files.
    
    Args:
        value (object): Raw text field value.
    
    Returns:
        str: Lowercased, whitespace-normalized text.
    """
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


def text_hash(text: str) -> str:
    """Compute SHA-256 hash for normalized text.
    
    Args:
        text (str): Normalized article text.
    
    Returns:
        str: Hex digest used to identify duplicate clusters.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    """Remove rows tied to duplicate text clusters from url_lookup and gdelt_scraped outputs.
    
    Returns:
        None: No return value.
    """
    args = parse_args()
    tqdm.pandas(desc="Normalizing Text")

    if not args.lookup.exists():
        raise FileNotFoundError(f"Lookup file not found: {args.lookup}")
    if not args.gdelt.exists():
        raise FileNotFoundError(f"GDELT file not found: {args.gdelt}")

    print("Reading files...")
    lookup = pd.read_csv(args.lookup, low_memory=False)
    gdelt = pd.read_csv(args.gdelt, low_memory=False)
    
    required_lookup = {"url_id", "Text"}
    missing_lookup = required_lookup - set(lookup.columns)
    if missing_lookup:
        raise ValueError(f"url_lookup.csv missing required columns: {sorted(missing_lookup)}")

    if "url_id" not in gdelt.columns:
        raise ValueError("gdelt_scraped.csv missing required column: 'url_id'")

    lookup["url_id"] = pd.to_numeric(lookup["url_id"], errors="coerce").astype("Int64")
    gdelt["url_id"] = pd.to_numeric(gdelt["url_id"], errors="coerce").astype("Int64")

    # Normalize text and build hash for duplicate detection.
    lookup["text_norm"] = lookup["Text"].progress_apply(normalize_text)
    lookup["text_len_norm"] = lookup["text_norm"].str.len()
    valid_text_mask = lookup["text_len_norm"] >= args.min_text_length

    lookup["text_hash"] = ""
    lookup.loc[valid_text_mask, "text_hash"] = lookup.loc[valid_text_mask, "text_norm"].progress_apply(text_hash)

    # Drop entire duplicate-text clusters (no representative URL kept).
    hash_counts = lookup.loc[lookup["text_hash"].ne(""), "text_hash"].value_counts()
    dup_hashes = set(hash_counts[hash_counts > 1].index.tolist())
    to_drop = lookup[lookup["text_hash"].isin(dup_hashes)].copy()
    keep_lookup = lookup[~lookup["text_hash"].isin(dup_hashes)].copy()

    dropped_url_ids = set(to_drop["url_id"].dropna().astype(int).tolist())

    # Remove all gdelt rows tied to dropped duplicate url_ids.
    gdelt_keep = gdelt[~gdelt["url_id"].isin(dropped_url_ids)].copy()

    # Cleanup helper columns before write.
    keep_lookup = keep_lookup.drop(columns=["text_norm"], errors="ignore")
    if "text_len_norm" not in keep_lookup.columns:
        keep_lookup["text_len_norm"] = keep_lookup["Text"].map(normalize_text).str.len()

    duplicate_clusters = len(dup_hashes)
    print(f"Duplicate text clusters found: {duplicate_clusters:,}")
    print(f"url_lookup rows removed: {len(to_drop):,}")
    print(f"url_lookup rows kept: {len(keep_lookup):,}")
    print(f"gdelt rows removed: {len(gdelt) - len(gdelt_keep):,}")
    print(f"gdelt rows kept: {len(gdelt_keep):,}")

    print("Saving updated files...")    
    keep_lookup.to_csv(args.lookup, index=False)
    gdelt_keep.to_csv(args.gdelt, index=False)
    print(f"Overwrote lookup: {args.lookup}")
    print(f"Overwrote gdelt: {args.gdelt}")


if __name__ == "__main__":
    main()
