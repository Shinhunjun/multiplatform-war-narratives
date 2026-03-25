from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import LOOKUP_PATH, bootstrap_project_paths, ensure_empty_csv


bootstrap_project_paths()
from build_url_index import canonicalize_url, choose_representative_rows, load_existing_lookup, upsert_lookup


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for incremental lookup updates."""
    parser = argparse.ArgumentParser(
        description=(
            "Update data/preprocessing/url_lookup.csv from newly appended weekly rows while preserving "
            "stable url_id values and emitting a changed_url_ids worklist."
        )
    )
    parser.add_argument("--events", type=Path, required=True, help="CSV of rows actually appended this run")
    parser.add_argument("--lookup", type=Path, default=LOOKUP_PATH, help="Path to data/preprocessing/url_lookup.csv")
    parser.add_argument("--changed-url-ids", type=Path, required=True, help="Output CSV path for changed url_ids")
    return parser.parse_args()


def _int_series(series: pd.Series) -> pd.Series:
    """Coerce a numeric series to pandas Int64."""
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("Int64")


def build_existing_representatives(existing_lookup: pd.DataFrame) -> pd.DataFrame:
    """Build an in-memory frame of existing representative URL rows."""
    if existing_lookup.empty:
        return pd.DataFrame(
            columns=["SourceURL", "SourceURL_Canonical", "Title", "Text", "Scrape_Status", "row_count"]
        )
    frame = existing_lookup.copy()
    if "SourceURL_Canonical" not in frame.columns:
        frame["SourceURL_Canonical"] = frame["SourceURL"].map(canonicalize_url)
    if "row_count" not in frame.columns:
        frame["row_count"] = 0
    return frame[
        ["SourceURL", "SourceURL_Canonical", "Title", "Text", "Scrape_Status", "row_count"]
    ].copy()


def changed_reasons(existing_lookup: pd.DataFrame, merged_lookup: pd.DataFrame) -> pd.DataFrame:
    """Return the changed-url worklist by comparing merged rows against the previous lookup."""
    existing = existing_lookup.copy()
    merged = merged_lookup.copy()

    if existing.empty:
        out = merged[["url_id"]].copy()
        out["reason"] = "new_url"
        return out

    existing["url_id"] = pd.to_numeric(existing["url_id"], errors="coerce").astype("Int64")
    merged["url_id"] = pd.to_numeric(merged["url_id"], errors="coerce").astype("Int64")
    existing = existing.drop_duplicates(subset=["url_id"], keep="last").set_index("url_id", drop=False)

    rows: list[dict[str, object]] = []
    for record in merged.to_dict(orient="records"):
        url_id = pd.to_numeric(record["url_id"], errors="coerce")
        if pd.isna(url_id):
            continue
        url_id_int = int(url_id)
        if url_id_int not in existing.index:
            rows.append({"url_id": url_id_int, "reason": "new_url"})
            continue
        prior = existing.loc[url_id_int]
        reason_parts: list[str] = []
        for column, label in [
            ("SourceURL", "sourceurl_changed"),
            ("Title", "title_changed"),
            ("Text", "text_changed"),
            ("Scrape_Status", "status_changed"),
        ]:
            if str(prior.get(column, "") or "") != str(record.get(column, "") or ""):
                reason_parts.append(label)
        if str(prior.get("Tokens", "") or "").strip() == "":
            reason_parts.append("tokens_missing")
        if reason_parts:
            rows.append({"url_id": url_id_int, "reason": "|".join(reason_parts)})

    if not rows:
        return pd.DataFrame(columns=["url_id", "reason"])
    out = pd.DataFrame(rows)
    out["url_id"] = pd.to_numeric(out["url_id"], errors="coerce").astype("Int64")
    return out.drop_duplicates(subset=["url_id"], keep="last").sort_values("url_id").reset_index(drop=True)


def main() -> None:
    """Update url_lookup.csv incrementally and emit changed_url_ids.csv."""
    args = parse_args()
    events_df = pd.read_csv(args.events, low_memory=False)
    existing_lookup = load_existing_lookup(args.lookup)
    if "row_count" not in existing_lookup.columns:
        existing_lookup["row_count"] = 0

    if events_df.empty:
        ensure_empty_csv(args.changed_url_ids, ["url_id", "reason"])
        if not args.lookup.exists():
            existing_lookup.to_csv(args.lookup, index=False)
        print(f"No appended weekly rows to merge into url_lookup. Output written: {args.changed_url_ids}")
        return

    events_df["SourceURL"] = events_df["SourceURL"].fillna("").astype(str)
    events_df["SourceURL_Canonical"] = events_df["SourceURL"].map(canonicalize_url)
    weekly_counts = events_df.groupby("SourceURL_Canonical").size().rename("weekly_row_count").astype("Int64")

    comparison_df = pd.concat(
        [
            build_existing_representatives(existing_lookup),
            events_df[["SourceURL", "SourceURL_Canonical", "Title", "Text", "Scrape_Status"]].copy(),
        ],
        ignore_index=True,
    )
    representative_df = choose_representative_rows(comparison_df)

    existing_canonical_to_id = {}
    if not existing_lookup.empty:
        existing_canonical_to_id = dict(
            zip(
                existing_lookup["SourceURL_Canonical"].fillna("").astype(str),
                pd.to_numeric(existing_lookup["url_id"], errors="coerce").astype("Int64"),
            )
        )

    next_id = 1
    if not existing_lookup.empty:
        next_id = int(pd.to_numeric(existing_lookup["url_id"], errors="coerce").max()) + 1

    url_ids: list[int] = []
    total_row_counts: list[int] = []
    existing_row_counts = {}
    if not existing_lookup.empty:
        existing_row_counts = dict(
            zip(
                existing_lookup["SourceURL_Canonical"].fillna("").astype(str),
                _int_series(existing_lookup["row_count"]),
            )
        )

    for canonical in representative_df["SourceURL_Canonical"].fillna("").astype(str):
        if canonical in existing_canonical_to_id and pd.notna(existing_canonical_to_id[canonical]):
            url_id_value = int(existing_canonical_to_id[canonical])
        else:
            url_id_value = next_id
            existing_canonical_to_id[canonical] = url_id_value
            next_id += 1
        url_ids.append(url_id_value)
        prior_count = int(existing_row_counts.get(canonical, 0))
        weekly_count = int(weekly_counts.get(canonical, 0))
        total_row_counts.append(prior_count + weekly_count)

    representative_df = representative_df.copy()
    representative_df["url_id"] = pd.Series(url_ids, dtype="Int64")
    representative_df["row_count"] = pd.Series(total_row_counts, dtype="Int64")
    representative_df["Tokens"] = ""
    incoming_lookup = representative_df[
        ["url_id", "SourceURL", "SourceURL_Canonical", "Title", "Text", "Tokens", "Scrape_Status", "row_count"]
    ].copy()

    merged_lookup = upsert_lookup(existing_lookup, incoming_lookup)
    changed_df = changed_reasons(existing_lookup, merged_lookup)

    args.lookup.parent.mkdir(parents=True, exist_ok=True)
    merged_lookup.to_csv(args.lookup, index=False)
    if changed_df.empty:
        ensure_empty_csv(args.changed_url_ids, ["url_id", "reason"])
    else:
        args.changed_url_ids.parent.mkdir(parents=True, exist_ok=True)
        changed_df.to_csv(args.changed_url_ids, index=False)

    print(f"Lookup rows after merge: {len(merged_lookup):,}")
    print(f"Changed url_ids emitted: {len(changed_df):,}")
    print(f"Lookup written: {args.lookup}")
    print(f"Changed ids written: {args.changed_url_ids}")


if __name__ == "__main__":
    main()
