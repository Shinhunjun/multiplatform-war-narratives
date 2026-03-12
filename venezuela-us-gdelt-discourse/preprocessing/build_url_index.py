from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from tqdm import tqdm


TRACKING_QUERY_PREFIXES = (
    "utm_",
    "mc_",
)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "igshid",
    "mkt_tok",
    "ref",
    "ref_src",
    "source",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for URL indexing and lookup generation.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    base_dir = Path(__file__).resolve().parents[1]
    default_input = base_dir / "data" / "gdelt_scraped.csv"
    default_lookup = Path(__file__).resolve().parent / "url_lookup.csv"

    parser = argparse.ArgumentParser(
        description="Assign stable url_id values and build/update url_lookup.csv."
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Path to gdelt_scraped.csv")
    parser.add_argument("--lookup", type=Path, default=default_lookup, help="Path to url_lookup.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write updated gdelt_scraped with url_id (optional).",
    )
    parser.add_argument(
        "--write-input",
        action="store_true",
        help="Overwrite --input with url_id (ignored when --output is provided).",
    )
    parser.add_argument(
        "--canonical-col",
        action="store_true",
        help="Keep SourceURL_Canonical in updated gdelt_scraped output",
    )
    return parser.parse_args()


def _is_blank(value: object) -> bool:
    """Return whether a value should be treated as blank for CSV normalization purposes.
    
    Args:
        value (object): Input value to check.
    
    Returns:
        bool: True when the value is null/NaN/empty; otherwise False.
    """
    if value is None:
        return True
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def canonicalize_url(url: object) -> str:
    """Canonicalize a URL by normalizing scheme/host/path and stripping tracking query parameters.
    
    Args:
        url (object): Raw SourceURL value.
    
    Returns:
        str: Canonical URL string used for deduplicated URL identity.
    """
    if _is_blank(url):
        return ""

    raw = str(url).strip()
    try:
        parsed = urlsplit(raw)
    except Exception:
        return raw

    scheme = (parsed.scheme or "http").lower()
    netloc = parsed.netloc.lower().strip()
    path = parsed.path or ""

    if not netloc and path:
        # Handles values like "example.com/path"
        parsed_retry = urlsplit(f"http://{raw}")
        scheme = "http"
        netloc = parsed_retry.netloc.lower().strip()
        path = parsed_retry.path or ""
        parsed = parsed_retry

    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    if path != "/":
        path = path.rstrip("/")

    query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
    filtered_pairs: list[tuple[str, str]] = []
    for key, value in query_pairs:
        key_l = key.lower()
        if key_l in TRACKING_QUERY_KEYS:
            continue
        if any(key_l.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        filtered_pairs.append((key, value))

    query = urlencode(sorted(filtered_pairs))
    return urlunsplit((scheme, netloc, path, query, ""))


def choose_representative_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Select one representative row per canonical URL using content-richness and status heuristics.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing canonical URLs and scrape fields.
    
    Returns:
        pd.DataFrame: Representative-row DataFrame with row_count metadata.
    """
    work = df.copy()
    work["Text_len"] = work["Text"].fillna("").astype(str).str.len()
    work["Title_len"] = work["Title"].fillna("").astype(str).str.len()
    work["Success_flag"] = work["Scrape_Status"].fillna("").str.contains("success", case=False).astype(int)
    work["Has_text"] = (work["Text_len"] > 0).astype(int)
    work["Has_title"] = (work["Title_len"] > 0).astype(int)

    work = work.sort_values(
        by=["SourceURL_Canonical", "Has_text", "Text_len", "Has_title", "Title_len", "Success_flag"],
        ascending=[True, False, False, False, False, False],
    )
    rep = work.drop_duplicates(subset=["SourceURL_Canonical"], keep="first").copy()
    row_counts = work.groupby("SourceURL_Canonical").size().rename("row_count")
    rep = rep.merge(row_counts, on="SourceURL_Canonical", how="left")
    return rep


def load_existing_lookup(lookup_path: Path) -> pd.DataFrame:
    """Load and normalize an existing url_lookup.csv table, adding expected columns when missing.
    
    Args:
        lookup_path (Path): Path to an existing lookup CSV file.
    
    Returns:
        pd.DataFrame: Existing lookup DataFrame, or an empty schema-compatible DataFrame if absent.
    """
    if not lookup_path.exists():
        return pd.DataFrame(
            columns=[
                "url_id",
                "SourceURL",
                "SourceURL_Canonical",
                "Title",
                "Text",
                "Tokens",
                "Scrape_Status",
                "row_count",
            ]
        )

    existing = pd.read_csv(lookup_path, low_memory=False)
    required = {"url_id", "SourceURL_Canonical"}
    missing = required - set(existing.columns)
    if missing:
        raise ValueError(f"Existing lookup missing required columns: {sorted(missing)}")

    if "Tokens" not in existing.columns:
        existing["Tokens"] = ""
    if "row_count" not in existing.columns:
        existing["row_count"] = pd.NA
    if "SourceURL" not in existing.columns:
        existing["SourceURL"] = ""
    if "Title" not in existing.columns:
        existing["Title"] = ""
    if "Text" not in existing.columns:
        existing["Text"] = ""
    if "Scrape_Status" not in existing.columns:
        existing["Scrape_Status"] = ""
    return existing


def upsert_lookup(existing_lookup: pd.DataFrame, incoming_lookup: pd.DataFrame) -> pd.DataFrame:
    """Upsert incoming representative rows into an existing lookup table by url_id.
    
    Args:
        existing_lookup (pd.DataFrame): Current lookup DataFrame from disk.
        incoming_lookup (pd.DataFrame): New lookup rows computed in this run.
    
    Returns:
        pd.DataFrame: Merged lookup DataFrame sorted by url_id.
    """
    if existing_lookup.empty:
        out = incoming_lookup.sort_values("url_id").reset_index(drop=True)
        out["url_id"] = out["url_id"].astype("Int64")
        return out

    out = existing_lookup.copy()
    incoming = incoming_lookup.copy()

    out["url_id"] = pd.to_numeric(out["url_id"], errors="coerce").astype("Int64")
    incoming["url_id"] = pd.to_numeric(incoming["url_id"], errors="coerce").astype("Int64")

    for col in incoming.columns:
        if col not in out.columns:
            out[col] = pd.NA

    out = out.set_index("url_id", drop=False)
    incoming = incoming.set_index("url_id", drop=False)

    overlap_ids = out.index.intersection(incoming.index)
    update_cols = [
        "SourceURL",
        "SourceURL_Canonical",
        "Title",
        "Text",
        "Scrape_Status",
        "row_count",
    ]
    for col in update_cols:
        if col in out.columns and col in incoming.columns and len(overlap_ids) > 0:
            out.loc[overlap_ids, col] = incoming.loc[overlap_ids, col].values

    if "Tokens" in out.columns and "Tokens" in incoming.columns and len(overlap_ids) > 0:
        for uid in overlap_ids:
            if _is_blank(out.at[uid, "Tokens"]) and not _is_blank(incoming.at[uid, "Tokens"]):
                out.at[uid, "Tokens"] = incoming.at[uid, "Tokens"]

    new_ids = incoming.index.difference(out.index)
    if len(new_ids) > 0:
        new_rows = incoming.loc[new_ids].copy()
        new_rows = new_rows.reindex(columns=out.columns)
        out = pd.concat([out, new_rows], axis=0)

    out = out.sort_index().reset_index(drop=True)
    out["url_id"] = pd.to_numeric(out["url_id"], errors="coerce").astype("Int64")
    return out


def main() -> None:
    """Assign stable url_id values and write/update url_lookup and optional input outputs.
    
    Returns:
        None: No return value.
    """
    args = parse_args()
    write_dataset = args.output is not None or args.write_input
    output_path = args.output if args.output is not None else args.input

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print("Loading input CSV...", flush=True)
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  Rows loaded: {len(df):,}", flush=True)
    required_cols = {"SourceURL", "Title", "Text", "Scrape_Status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    print("Canonicalizing SourceURL values...", flush=True)
    df["SourceURL"] = df["SourceURL"].fillna("").astype(str)
    tqdm.pandas(desc="Canonicalizing URLs", file=sys.stdout)
    df["SourceURL_Canonical"] = df["SourceURL"].progress_map(canonicalize_url)

    print("Selecting representative row per canonical URL...", flush=True)
    reps = choose_representative_rows(df)
    # If the input already has url_id, remove it here and rebuild from lookup mapping.
    # This avoids merge suffixes (url_id_x/url_id_y) and keeps IDs stable from lookup.
    reps = reps.drop(columns=["url_id"], errors="ignore")

    print("Loading existing url_lookup mapping...", flush=True)
    existing_lookup = load_existing_lookup(args.lookup)
    existing_lookup["SourceURL_Canonical"] = existing_lookup["SourceURL_Canonical"].fillna("").astype(str)

    existing_ids = (
        existing_lookup[["SourceURL_Canonical", "url_id"]]
        .dropna(subset=["SourceURL_Canonical", "url_id"])
        .drop_duplicates(subset=["SourceURL_Canonical"], keep="first")
    )
    existing_ids["url_id"] = pd.to_numeric(existing_ids["url_id"], errors="coerce").astype("Int64")

    print("Assigning/rehydrating url_id values...", flush=True)
    reps = reps.merge(existing_ids, on="SourceURL_Canonical", how="left")
    max_existing_id = int(existing_ids["url_id"].max()) if not existing_ids["url_id"].dropna().empty else 0

    new_mask = reps["url_id"].isna()
    new_count = int(new_mask.sum())
    if new_count:
        reps.loc[new_mask, "url_id"] = range(max_existing_id + 1, max_existing_id + 1 + new_count)
    reps["url_id"] = reps["url_id"].astype("Int64")

    lookup_cols = [
        "url_id",
        "SourceURL",
        "SourceURL_Canonical",
        "Title",
        "Text",
        "Scrape_Status",
        "row_count",
    ]
    if "Tokens" in reps.columns:
        lookup_cols.insert(5, "Tokens")
    else:
        reps["Tokens"] = ""
        lookup_cols.insert(5, "Tokens")

    incoming_lookup = reps[lookup_cols].copy()
    new_lookup = upsert_lookup(existing_lookup, incoming_lookup)

    url_to_id = new_lookup[["SourceURL_Canonical", "url_id"]].drop_duplicates(subset=["SourceURL_Canonical"])
    df = df.drop(columns=["url_id"], errors="ignore")
    df = df.merge(url_to_id, on="SourceURL_Canonical", how="left")
    df["url_id"] = df["url_id"].astype("Int64")

    if not args.canonical_col and write_dataset:
        df = df.drop(columns=["SourceURL_Canonical"])

    args.lookup.parent.mkdir(parents=True, exist_ok=True)
    if write_dataset:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Writing output files...", flush=True)
    new_lookup.to_csv(args.lookup, index=False)
    if write_dataset:
        df.to_csv(output_path, index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Unique canonical URLs: {new_lookup['SourceURL_Canonical'].nunique():,}")
    print(f"New url_id assigned this run: {new_count:,}")
    print(f"Lookup written: {args.lookup}")
    if write_dataset:
        print(f"Updated dataset written: {output_path}")
    else:
        print("Input dataset not written (use --write-input or --output to write gdelt_scraped with url_id).")


if __name__ == "__main__":
    main()
