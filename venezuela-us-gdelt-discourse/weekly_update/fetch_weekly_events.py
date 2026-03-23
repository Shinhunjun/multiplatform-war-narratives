from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from common import (
    MASTER_DATASET_PATH,
    RAW_EVENT_COLS,
    USER_AGENT,
    format_yyyymmdd,
    latest_event_date,
)


MASTERFILE_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
EXPORT_URL_RE = re.compile(r"http://data\.gdeltproject\.org/gdeltv2/(\d{14})\.export\.CSV\.zip$")
TAIL_BYTES_INITIAL = 2_000_000
TAIL_BYTES_MAX = 128_000_000

IDX_GLOBALEVENTID = 0
IDX_DATE = 1
IDX_YEAR = 3
IDX_ACTOR1_NAME = 6
IDX_ACTOR1_COUNTRY = 7
IDX_ACTOR2_NAME = 16
IDX_ACTOR2_COUNTRY = 17
IDX_EVENT_CODE = 26
IDX_QUAD_CLASS = 29
IDX_GOLDSTEIN = 30
IDX_AVG_TONE = 34
IDX_SOURCE_URL = 60

OUTPUT_COLS = ["GLOBALEVENTID", "GDELTExportTimestamp"] + RAW_EVENT_COLS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for weekly event fetching."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch new GDELT 2 event rows for the Venezuela-USA dyad, starting from the latest "
            "event date already present in the local master dataset unless overridden."
        )
    )
    parser.add_argument("--master", type=Path, default=MASTER_DATASET_PATH, help="Path to data/gdelt_scraped.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path for weekly raw events")
    parser.add_argument("--from-date", default=None, help="Inclusive start date in YYYYMMDD format")
    parser.add_argument("--to-date", default=None, help="Inclusive end date in YYYYMMDD format")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for downloaded export files")
    return parser.parse_args()


def extract_timestamp_from_url(url: str) -> str:
    """Extract the 14-digit export timestamp from a GDELT export URL."""
    match = EXPORT_URL_RE.search(url.strip())
    if not match:
        raise ValueError(f"Unrecognized GDELT export URL: {url}")
    return match.group(1)


def parse_masterfile_lines(text: str, start_date: str, end_date: str) -> list[str]:
    """Parse masterfile text and return matching export URLs for the requested date window."""
    urls: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        url = parts[-1]
        match = EXPORT_URL_RE.search(url)
        if not match:
            continue
        timestamp = match.group(1)
        date_part = timestamp[:8]
        if start_date <= date_part <= end_date:
            urls.append(url)
    return urls


def latest_available_export_timestamp(session: requests.Session, timeout: int) -> str:
    """Read the latest available GDELT export timestamp from lastupdate.txt."""
    response = session.get(LASTUPDATE_URL, timeout=timeout)
    response.raise_for_status()
    for line in response.text.splitlines():
        parts = line.split()
        if not parts:
            continue
        url = parts[-1]
        if ".export.CSV.zip" in url:
            return extract_timestamp_from_url(url)
    raise ValueError("Unable to parse latest export timestamp from lastupdate.txt")


def masterfile_export_urls(
    session: requests.Session,
    start_date: str,
    end_date: str,
    timeout: int,
) -> list[str]:
    """Fetch matching export URLs from masterfilelist using ranged tail reads with fallback."""
    head = session.head(MASTERFILE_URL, timeout=timeout)
    head.raise_for_status()
    total_bytes = int(head.headers.get("Content-Length", "0") or "0")
    if total_bytes <= 0:
        response = session.get(MASTERFILE_URL, timeout=timeout)
        response.raise_for_status()
        urls = parse_masterfile_lines(response.text, start_date, end_date)
        return sorted(set(urls))

    tail_bytes = min(TAIL_BYTES_INITIAL, total_bytes)
    urls: list[str] = []
    while True:
        start = max(0, total_bytes - tail_bytes)
        headers = {"Range": f"bytes={start}-{total_bytes - 1}"}
        response = session.get(MASTERFILE_URL, headers=headers, timeout=timeout)
        response.raise_for_status()
        text = response.text
        urls = parse_masterfile_lines(text, start_date, end_date)
        parsed_timestamps = [extract_timestamp_from_url(url) for url in urls] if urls else []
        earliest = min(parsed_timestamps)[:8] if parsed_timestamps else None
        if earliest is not None and earliest <= start_date:
            return sorted(set(urls))
        if tail_bytes >= min(TAIL_BYTES_MAX, total_bytes):
            break
        tail_bytes = min(tail_bytes * 2, TAIL_BYTES_MAX, total_bytes)

    response = session.get(MASTERFILE_URL, timeout=timeout)
    response.raise_for_status()
    urls = parse_masterfile_lines(response.text, start_date, end_date)
    return sorted(set(urls))


def is_venezuela_us_dyad(actor1_country: str, actor2_country: str) -> bool:
    """Return whether an event row belongs to the VEN/USA dyad in either direction."""
    return (actor1_country == "VEN" and actor2_country == "USA") or (
        actor1_country == "USA" and actor2_country == "VEN"
    )


def map_export_row(row: list[str], export_timestamp: str) -> dict[str, str] | None:
    """Map a raw GDELT event-export row into the repo's weekly raw-event schema."""
    if len(row) <= IDX_SOURCE_URL:
        return None

    actor1_country = row[IDX_ACTOR1_COUNTRY].strip()
    actor2_country = row[IDX_ACTOR2_COUNTRY].strip()
    source_url = row[IDX_SOURCE_URL].strip()
    if not is_venezuela_us_dyad(actor1_country, actor2_country):
        return None
    if not source_url or source_url.lower() == "unspecified":
        return None

    return {
        "GLOBALEVENTID": row[IDX_GLOBALEVENTID].strip(),
        "GDELTExportTimestamp": export_timestamp,
        "Date": row[IDX_DATE].strip(),
        "Year": row[IDX_YEAR].strip(),
        "Actor1Name": row[IDX_ACTOR1_NAME].strip(),
        "Actor1CountryCode": actor1_country,
        "Actor2Name": row[IDX_ACTOR2_NAME].strip(),
        "Actor2CountryCode": actor2_country,
        "EventCode": row[IDX_EVENT_CODE].strip(),
        "QuadClass": row[IDX_QUAD_CLASS].strip(),
        "GoldsteinScale": row[IDX_GOLDSTEIN].strip(),
        "AvgTone": row[IDX_AVG_TONE].strip(),
        "SourceURL": source_url,
    }


def fetch_rows_from_export(
    session: requests.Session,
    url: str,
    start_date: str,
    end_date: str,
    timeout: int,
) -> list[dict[str, str]]:
    """Download one GDELT export ZIP and return filtered weekly-event rows."""
    export_timestamp = extract_timestamp_from_url(url)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            wrapper = io.TextIOWrapper(handle, encoding="utf-8", errors="replace", newline="")
            reader = csv.reader(wrapper, delimiter="\t")
            for raw_row in reader:
                if len(raw_row) <= IDX_SOURCE_URL:
                    continue
                date_value = raw_row[IDX_DATE].strip()
                if not (start_date <= date_value <= end_date):
                    continue
                mapped = map_export_row(raw_row, export_timestamp)
                if mapped is not None:
                    rows.append(mapped)
    return rows


def main() -> None:
    """Fetch weekly GDELT event rows and persist them as a staging CSV."""
    args = parse_args()
    start_date = args.from_date or latest_event_date(args.master)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    latest_export_ts = latest_available_export_timestamp(session, timeout=args.timeout)
    latest_export_date = latest_export_ts[:8]
    end_date = args.to_date or latest_export_date

    if start_date > end_date:
        raise ValueError(f"Start date {start_date} is after end date {end_date}")

    export_urls = masterfile_export_urls(session, start_date=start_date, end_date=end_date, timeout=args.timeout)
    if args.max_files is not None:
        export_urls = export_urls[: args.max_files]

    all_rows: list[dict[str, str]] = []
    for url in export_urls:
        all_rows.extend(fetch_rows_from_export(session, url, start_date=start_date, end_date=end_date, timeout=args.timeout))

    output_df = pd.DataFrame(all_rows, columns=OUTPUT_COLS)
    if not output_df.empty:
        output_df = output_df.drop_duplicates(subset=["GLOBALEVENTID", "SourceURL"], keep="last")
        output_df = output_df.sort_values(by=["Date", "GDELTExportTimestamp", "GLOBALEVENTID"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    print(f"Latest dataset date: {start_date}")
    print(f"Latest available export date: {latest_export_date}")
    print(f"Export files scanned: {len(export_urls):,}")
    print(f"Weekly event rows written: {len(output_df):,}")
    print(f"Output written: {args.output}")


if __name__ == "__main__":
    main()
