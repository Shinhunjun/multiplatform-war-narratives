from __future__ import annotations

import csv
import json
import os
import sys
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
PREPROCESSING_DIR = PROJECT_DIR / "preprocessing"
WEEKLY_RUNS_DIR = DATA_DIR / "weekly_runs"
MASTER_DATASET_PATH = DATA_DIR / "gdelt_scraped.csv"
LOOKUP_PATH = DATA_DIR / "preprocessing" / "url_lookup.csv"
RELEVANCE_PATH = DATA_DIR / "preprocessing" / "text_relevance_tokens.csv"
FILTER_EVAL_PATH = DATA_DIR / "preprocessing" / "url_filter_eval.csv"
FILTER_SUMMARY_PATH = DATA_DIR / "preprocessing" / "url_filter_summary_counts.csv"
FILTER_PLOT_PATH = DATA_DIR / "preprocessing" / "filter_stage_score_histograms.png"
FILTER_RULES_PATH = PREPROCESSING_DIR / "filter_rule_config.json"
ANCHORS_PATH = PREPROCESSING_DIR / "anchor_token_sets.json"
ANALYSIS_READY_DIR = DATA_DIR / "analysis_ready"
ANALYSIS_EVENTS_PATH = ANALYSIS_READY_DIR / "analysis_events.parquet"
ANALYSIS_URL_CONTENT_PATH = ANALYSIS_READY_DIR / "analysis_url_content.parquet"

RAW_EVENT_COLS = [
    "Date",
    "Year",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor2Name",
    "Actor2CountryCode",
    "EventCode",
    "QuadClass",
    "GoldsteinScale",
    "AvgTone",
    "SourceURL",
]
MASTER_EVENT_COLS = RAW_EVENT_COLS + ["Title", "Text", "Scrape_Status", "Error_Details"]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    manifest_path: Path
    weekly_events_raw: Path
    weekly_scraped: Path
    weekly_appended: Path
    append_audit: Path
    changed_url_ids: Path
    weekly_score_summary: Path
    filter_samples_dir: Path


def bootstrap_project_paths() -> None:
    """Ensure the project root and preprocessing directory are importable."""
    for path in (PROJECT_DIR, PREPROCESSING_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


def make_run_id(now: datetime | None = None) -> str:
    """Build a UTC run identifier suitable for weekly run directories."""
    value = now or utc_now()
    return value.strftime("%Y%m%dT%H%M%SZ")


def ensure_run_paths(run_id: str) -> RunPaths:
    """Create the per-run output directory structure and return its paths."""
    run_dir = WEEKLY_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        weekly_events_raw=run_dir / "weekly_events_raw.csv",
        weekly_scraped=run_dir / "weekly_scraped.csv",
        weekly_appended=run_dir / "weekly_appended.csv",
        append_audit=run_dir / "append_audit.csv",
        changed_url_ids=run_dir / "changed_url_ids.csv",
        weekly_score_summary=run_dir / "weekly_score_summary.csv",
        filter_samples_dir=run_dir / "filter_samples",
    )


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read a JSON file when present, otherwise return the provided default."""
    if not path.exists():
        return {} if default is None else default.copy()
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def atomic_write_path(target: Path) -> Generator[Path, None, None]:
    """Yield a temporary path; on success rename it atomically to *target*.

    Usage::

        with atomic_write_path(output_path) as tmp:
            df.to_csv(tmp, index=False)

    If the body raises, the temporary file is removed and the target is left
    untouched.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        yield tmp
        os.replace(tmp, target)
    except BaseException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def update_manifest(manifest_path: Path, **updates: Any) -> dict[str, Any]:
    """Update a run manifest in place and return the merged payload."""
    payload = read_json(manifest_path, default={})
    payload.update(updates)
    write_json(manifest_path, payload)
    return payload


def parse_yyyymmdd(value: str | int) -> datetime:
    """Parse a YYYYMMDD date string into a UTC-naive datetime value."""
    return datetime.strptime(str(value), "%Y%m%d")


def format_yyyymmdd(value: datetime) -> str:
    """Format a datetime as YYYYMMDD."""
    return value.strftime("%Y%m%d")


def latest_event_date(master_path: Path = MASTER_DATASET_PATH) -> str:
    """Return the latest event date present in the master dataset."""
    if not master_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_path}")

    max_date: str | None = None
    for chunk in pd.read_csv(master_path, usecols=["Date"], chunksize=250_000, low_memory=False):
        values = chunk["Date"].dropna().astype(str)
        if values.empty:
            continue
        chunk_max = values.max()
        if max_date is None or chunk_max > max_date:
            max_date = chunk_max

    if max_date is None:
        raise ValueError(f"Unable to determine latest Date from {master_path}")
    return max_date


def csv_row_count(path: Path) -> int:
    """Count rows in a CSV file excluding the header when the file exists."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def ensure_empty_csv(path: Path, columns: list[str]) -> None:
    """Create an empty CSV file with the requested columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def normalize_key_value(value: object) -> str:
    """Normalize a raw key field into a deterministic string representation."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def event_identity_tuple(row: pd.Series | dict[str, Any]) -> tuple[str, ...]:
    """Build the event identity tuple used for weekly overlap deduplication."""
    return tuple(normalize_key_value(row[col]) for col in RAW_EVENT_COLS)


def dataframe_event_keys(df: pd.DataFrame) -> list[tuple[str, ...]]:
    """Build event identity tuples for an in-memory event DataFrame."""
    records = df[RAW_EVENT_COLS].to_dict(orient="records")
    return [event_identity_tuple(record) for record in records]


def recent_master_subset(master_path: Path, min_date: str) -> pd.DataFrame:
    """Load only the recent tail of the master dataset needed for overlap checks."""
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(master_path, chunksize=200_000, low_memory=False):
        dates = chunk["Date"].astype(str)
        mask = dates >= str(min_date)
        if mask.any():
            parts.append(chunk.loc[mask, MASTER_EVENT_COLS].copy())
    if not parts:
        return pd.DataFrame(columns=MASTER_EVENT_COLS)
    return pd.concat(parts, ignore_index=True)


def append_rows_to_csv(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    """Append rows to a CSV using a staging file for crash safety.

    Rows are first written to ``<path>.staging``.  Only after the staging file
    is fully written are its bytes appended to the canonical *path*.  If the
    process is interrupted during the final append the staging file remains on
    disk and can be inspected or re-applied manually.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    ordered = frame[columns].copy()
    staging = path.with_suffix(".staging")
    ordered.to_csv(staging, mode="w", header=not file_exists, index=False)
    with staging.open("rb") as src, path.open("ab") as dst:
        dst.write(src.read())
    staging.unlink()


def write_audit_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write an audit CSV from an iterable of dictionaries."""
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        ensure_empty_csv(path, ["metric", "value"])
        return
    pd.DataFrame(materialized).to_csv(path, index=False)


def load_changed_url_ids(path: Path) -> pd.DataFrame:
    """Load a changed-url-id worklist, returning an empty frame when absent."""
    if not path.exists():
        return pd.DataFrame(columns=["url_id", "reason"])
    df = pd.read_csv(path, low_memory=False)
    if "url_id" in df.columns:
        df["url_id"] = pd.to_numeric(df["url_id"], errors="coerce").astype("Int64")
    return df


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    """Load a small CSV into a list of row dictionaries."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
