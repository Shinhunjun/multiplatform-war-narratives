from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import (
    ANALYSIS_EVENTS_PATH,
    ANALYSIS_URL_CONTENT_PATH,
    FILTER_EVAL_PATH,
    FILTER_RULES_PATH,
    LOOKUP_PATH,
    MASTER_DATASET_PATH,
    PROJECT_DIR,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for refreshing downstream weekly outputs."""
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild analysis-ready parquet exports and optionally rerun EDA and the downstream "
            "analysis pipeline."
        )
    )
    parser.add_argument("--events", type=Path, default=MASTER_DATASET_PATH, help="Path to data/gdelt_scraped.csv")
    parser.add_argument("--lookup", type=Path, default=LOOKUP_PATH, help="Path to data/preprocessing/url_lookup.csv")
    parser.add_argument("--eval", type=Path, default=FILTER_EVAL_PATH, help="Path to data/preprocessing/url_filter_eval.csv")
    parser.add_argument("--filter-rules", type=Path, default=FILTER_RULES_PATH, help="Path to filter_rule_config.json")
    parser.add_argument(
        "--events-output",
        type=Path,
        default=ANALYSIS_EVENTS_PATH,
        help="Output parquet path for analysis_events.parquet",
    )
    parser.add_argument(
        "--url-output",
        type=Path,
        default=ANALYSIS_URL_CONTENT_PATH,
        help="Output parquet path for analysis_url_content.parquet",
    )
    parser.add_argument("--run-eda", action="store_true", help="Also rerun eda/run_eda.py")
    parser.add_argument("--run-analysis", action="store_true", help="Also rerun python -m analysis.main --all")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    """Run a subprocess from the project root with unbuffered output."""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, cwd=PROJECT_DIR, check=True, env=env)


def main() -> None:
    """Rebuild downstream analysis exports and optional weekly outputs."""
    args = parse_args()

    run_command(
        [
            sys.executable,
            str(PROJECT_DIR / "preprocessing" / "build_analysis_ready_datasets.py"),
            "--events",
            str(args.events),
            "--lookup",
            str(args.lookup),
            "--eval",
            str(args.eval),
            "--filter-rules",
            str(args.filter_rules),
            "--events-output",
            str(args.events_output),
            "--url-output",
            str(args.url_output),
        ]
    )

    if args.run_eda:
        run_command([sys.executable, str(PROJECT_DIR / "eda" / "run_eda.py")])

    if args.run_analysis:
        run_command([sys.executable, "-m", "analysis.main", "--all"])

    print(f"Analysis-ready events parquet: {args.events_output}")
    print(f"Analysis-ready URL-content parquet: {args.url_output}")
    print(f"EDA rerun: {args.run_eda}")
    print(f"Analysis rerun: {args.run_analysis}")


if __name__ == "__main__":
    main()
