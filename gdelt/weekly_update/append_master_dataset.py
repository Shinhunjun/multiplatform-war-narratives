from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    MASTER_DATASET_PATH,
    MASTER_EVENT_COLS,
    append_rows_to_csv,
    atomic_write_path,
    csv_row_count,
    dataframe_event_keys,
    ensure_empty_csv,
    recent_master_subset,
    write_audit_rows,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for appending weekly rows to the master dataset."""
    parser = argparse.ArgumentParser(
        description=(
            "Append only genuinely new weekly rows into data/gdelt_scraped.csv while safely "
            "ignoring same-date overlap that was refetched from GDELT."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Weekly scraped-event CSV path")
    parser.add_argument("--master", type=Path, default=MASTER_DATASET_PATH, help="Path to data/gdelt_scraped.csv")
    parser.add_argument("--appended-output", type=Path, required=True, help="CSV path for rows actually appended")
    parser.add_argument("--audit-output", type=Path, required=True, help="Audit CSV path")
    return parser.parse_args()


def rows_to_append(incoming_df: pd.DataFrame, recent_existing_df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of incoming rows that are new beyond the recent overlap window."""
    if incoming_df.empty:
        return incoming_df.head(0).copy()

    existing_counts = Counter(dataframe_event_keys(recent_existing_df)) if not recent_existing_df.empty else Counter()
    seen_counts: Counter[tuple[str, ...]] = Counter()
    keep_indices: list[int] = []

    for idx, key in zip(incoming_df.index.tolist(), dataframe_event_keys(incoming_df)):
        if seen_counts[key] >= existing_counts[key]:
            keep_indices.append(idx)
        seen_counts[key] += 1

    return incoming_df.loc[keep_indices].copy()


def main() -> None:
    """Append new weekly rows into the canonical master dataset and write an append audit."""
    args = parse_args()
    print("Loading incoming weekly scraped rows...")
    incoming_df = pd.read_csv(args.input, low_memory=False)
    if incoming_df.empty:
        ensure_empty_csv(args.appended_output, MASTER_EVENT_COLS)
        write_audit_rows(
            args.audit_output,
            [
                {"metric": "input_rows", "value": 0},
                {"metric": "recent_overlap_rows", "value": 0},
                {"metric": "rows_appended", "value": 0},
                {"metric": "rows_skipped_as_overlap", "value": 0},
            ],
        )
        print(f"No weekly rows to append. Output written: {args.appended_output}")
        return

    canonical_df = incoming_df.reindex(columns=MASTER_EVENT_COLS).copy()
    min_incoming_date = canonical_df["Date"].astype(str).min()
    print(f"Loading recent master rows on or after {min_incoming_date} for overlap check...")
    recent_existing_df = recent_master_subset(args.master, min_date=min_incoming_date) if args.master.exists() else pd.DataFrame(columns=MASTER_EVENT_COLS)
    print(f"Deduplicating {len(canonical_df):,} incoming rows against {len(recent_existing_df):,} recent master rows...")
    appended_df = rows_to_append(canonical_df, recent_existing_df)

    if appended_df.empty:
        ensure_empty_csv(args.appended_output, MASTER_EVENT_COLS)
    else:
        appended_df = appended_df.reset_index(drop=True)
        print(f"Writing {len(appended_df):,} new rows to appended output...")
        with atomic_write_path(args.appended_output) as tmp:
            appended_df.to_csv(tmp, index=False)

    if not args.master.exists():
        args.master.parent.mkdir(parents=True, exist_ok=True)
        ensure_empty_csv(args.master, MASTER_EVENT_COLS)

    if not appended_df.empty:
        print(f"Appending {len(appended_df):,} rows to master dataset (staging for crash safety)...")
        append_rows_to_csv(args.master, appended_df, MASTER_EVENT_COLS)

    skipped = len(canonical_df) - len(appended_df)
    audit_rows = [
        {"metric": "input_rows", "value": int(len(canonical_df))},
        {"metric": "recent_overlap_rows", "value": int(len(recent_existing_df))},
        {"metric": "rows_appended", "value": int(len(appended_df))},
        {"metric": "rows_skipped_as_overlap", "value": int(skipped)},
        {"metric": "master_rows_after_append", "value": int(csv_row_count(args.master))},
        {"metric": "min_incoming_date", "value": str(min_incoming_date)},
        {"metric": "max_incoming_date", "value": str(canonical_df['Date'].astype(str).max())},
    ]
    write_audit_rows(args.audit_output, audit_rows)

    print(f"Incoming weekly rows: {len(canonical_df):,}")
    print(f"Rows appended to master: {len(appended_df):,}")
    print(f"Rows skipped as overlap: {skipped:,}")
    print(f"Appended rows written: {args.appended_output}")
    print(f"Audit written: {args.audit_output}")


if __name__ == "__main__":
    main()
