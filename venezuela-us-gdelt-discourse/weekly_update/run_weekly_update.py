from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    PROJECT_DIR,
    csv_row_count,
    ensure_run_paths,
    latest_event_date,
    make_run_id,
    update_manifest,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the weekly orchestrator."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the weekly Venezuela-USA GDELT update pipeline: fetch, scrape, append, update lookup, "
            "apply frozen relevance/static filters, and refresh downstream exports."
        )
    )
    parser.add_argument("--run-id", default=None, help="Explicit run id (default: current UTC timestamp)")
    parser.add_argument("--from-date", default=None, help="Inclusive start date in YYYYMMDD format")
    parser.add_argument("--to-date", default=None, help="Inclusive end date in YYYYMMDD format")
    parser.add_argument("--run-eda", action="store_true", help="Also rerun eda/run_eda.py")
    parser.add_argument("--run-analysis", action="store_true", help="Also rerun python -m analysis.main --all")
    parser.add_argument("--max-fetch-files", type=int, default=None, help="Optional cap for fetched GDELT export files")
    return parser.parse_args()


def run_step(label: str, command: list[str]) -> None:
    """Run one weekly-update stage from the project root."""
    print("")
    print("=" * 72)
    print(label)
    print("=" * 72)
    print("Command:", " ".join(command))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, cwd=PROJECT_DIR, check=True, env=env)


def main() -> None:
    """Run the full weekly update workflow and record a manifest."""
    args = parse_args()
    run_id = args.run_id or make_run_id()
    paths = ensure_run_paths(run_id)
    started_at = utc_now().isoformat()

    update_manifest(
        paths.manifest_path,
        run_id=run_id,
        status="running",
        started_at=started_at,
        from_date=args.from_date,
        to_date=args.to_date,
        run_eda=bool(args.run_eda),
        run_analysis=bool(args.run_analysis),
        latest_existing_date=latest_event_date(),
    )

    try:
        fetch_cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "fetch_weekly_events.py"),
            "--output",
            str(paths.weekly_events_raw),
        ]
        if args.from_date:
            fetch_cmd += ["--from-date", args.from_date]
        if args.to_date:
            fetch_cmd += ["--to-date", args.to_date]
        if args.max_fetch_files is not None:
            fetch_cmd += ["--max-files", str(args.max_fetch_files)]
        run_step("Step 1/7: Fetch Weekly Events", fetch_cmd)

        run_step(
            "Step 2/7: Scrape Weekly URLs",
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "scrape_weekly_urls.py"),
                "--input",
                str(paths.weekly_events_raw),
                "--output",
                str(paths.weekly_scraped),
            ],
        )

        run_step(
            "Step 3/7: Append Master Dataset",
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "append_master_dataset.py"),
                "--input",
                str(paths.weekly_scraped),
                "--appended-output",
                str(paths.weekly_appended),
                "--audit-output",
                str(paths.append_audit),
            ],
        )

        if csv_row_count(paths.weekly_appended) > 0:
            run_step(
                "Step 4/7: Update Lookup Incrementally",
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "update_lookup_incremental.py"),
                    "--events",
                    str(paths.weekly_appended),
                    "--changed-url-ids",
                    str(paths.changed_url_ids),
                ],
            )

            run_step(
                "Step 5/7: Apply Frozen Relevance",
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "apply_frozen_relevance.py"),
                    "--changed-url-ids",
                    str(paths.changed_url_ids),
                    "--summary-output",
                    str(paths.weekly_score_summary),
                ],
            )

            run_step(
                "Step 6/7: Apply Static Filters",
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "apply_static_filters.py"),
                    "--changed-url-ids",
                    str(paths.changed_url_ids),
                    "--sample-dir",
                    str(paths.filter_samples_dir),
                ],
            )

            refresh_cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "refresh_analysis_exports.py"),
            ]
            if args.run_eda:
                refresh_cmd.append("--run-eda")
            if args.run_analysis:
                refresh_cmd.append("--run-analysis")
            run_step("Step 7/7: Refresh Analysis Exports", refresh_cmd)
        else:
            update_manifest(paths.manifest_path, note="No appended rows; downstream refresh skipped.")

        update_manifest(
            paths.manifest_path,
            status="completed",
            finished_at=utc_now().isoformat(),
            outputs={
                "run_dir": str(paths.run_dir),
                "weekly_events_raw": str(paths.weekly_events_raw),
                "weekly_scraped": str(paths.weekly_scraped),
                "weekly_appended": str(paths.weekly_appended),
                "append_audit": str(paths.append_audit),
                "changed_url_ids": str(paths.changed_url_ids),
                "weekly_score_summary": str(paths.weekly_score_summary),
                "filter_samples_dir": str(paths.filter_samples_dir),
            },
        )
        print("")
        print(f"Weekly update complete. Run dir: {paths.run_dir}")
    except Exception as exc:
        update_manifest(
            paths.manifest_path,
            status="failed",
            finished_at=utc_now().isoformat(),
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
