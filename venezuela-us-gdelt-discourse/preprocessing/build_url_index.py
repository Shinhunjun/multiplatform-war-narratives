from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd


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
        help="Path to write updated gdelt_scraped with url_id (default: overwrite --input)",
    )
    parser.add_argument(
        "--canonical-col",
        action="store_true",
        help="Keep SourceURL_Canonical in updated gdelt_scraped output",
    )
    return parser.parse_args()


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def canonicalize_url(url: object) -> str:
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


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output is not None else args.input

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    required_cols = {"SourceURL", "Title", "Text", "Scrape_Status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    df["SourceURL"] = df["SourceURL"].fillna("").astype(str)
    df["SourceURL_Canonical"] = df["SourceURL"].map(canonicalize_url)

    reps = choose_representative_rows(df)
    existing_lookup = load_existing_lookup(args.lookup)
    existing_lookup["SourceURL_Canonical"] = existing_lookup["SourceURL_Canonical"].fillna("").astype(str)

    existing_ids = (
        existing_lookup[["SourceURL_Canonical", "url_id"]]
        .dropna(subset=["SourceURL_Canonical", "url_id"])
        .drop_duplicates(subset=["SourceURL_Canonical"], keep="first")
    )
    existing_ids["url_id"] = pd.to_numeric(existing_ids["url_id"], errors="coerce").astype("Int64")

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

    new_lookup = reps[lookup_cols].copy()

    if not existing_lookup.empty:
        existing_keep = existing_lookup[
            [
                "url_id",
                "SourceURL_Canonical",
                "Tokens",
            ]
        ].copy()
        new_lookup = new_lookup.merge(
            existing_keep.rename(columns={"Tokens": "Tokens_existing"}),
            on=["url_id", "SourceURL_Canonical"],
            how="left",
        )
        # Preserve existing non-empty token values (e.g., manual edits or precomputed tokens).
        new_lookup["Tokens"] = new_lookup.apply(
            lambda r: r["Tokens_existing"] if not _is_blank(r["Tokens_existing"]) else r["Tokens"], axis=1
        )
        new_lookup = new_lookup.drop(columns=["Tokens_existing"])

    new_lookup = new_lookup.sort_values("url_id").reset_index(drop=True)
    new_lookup["url_id"] = new_lookup["url_id"].astype("Int64")

    url_to_id = new_lookup[["SourceURL_Canonical", "url_id"]].drop_duplicates(subset=["SourceURL_Canonical"])
    df = df.merge(url_to_id, on="SourceURL_Canonical", how="left")
    df["url_id"] = df["url_id"].astype("Int64")

    if not args.canonical_col:
        df = df.drop(columns=["SourceURL_Canonical"])

    args.lookup.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_lookup.to_csv(args.lookup, index=False)
    df.to_csv(output_path, index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Unique canonical URLs: {new_lookup['SourceURL_Canonical'].nunique():,}")
    print(f"New url_id assigned this run: {new_count:,}")
    print(f"Lookup written: {args.lookup}")
    print(f"Updated dataset written: {output_path}")


if __name__ == "__main__":
    main()
