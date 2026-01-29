"""
Main entry point for the India-Pakistan Reddit Data Collection Pipeline.

Usage:
    uv run main.py historical
    uv run main.py historical --start 2024-01-01 --end 2024-12-31
    uv run main.py crisis
    uv run main.py crisis --name operation_sindoor
    uv run main.py crisis --all
    uv run main.py comments
    uv run main.py comments --file india_20240101_20241231_monthly_filtered.json
    uv run main.py analyze
    uv run main.py export
"""

import argparse
import asyncio
import sys
from pathlib import Path


def run_historical(args: argparse.Namespace) -> None:
    """Run historical data collection."""
    from scripts.collectors import collect_historical
    from scripts.config import (
        ALL_SEARCH_QUERIES,
        ALL_SUBREDDITS,
        HISTORICAL_DEFAULT_END,
        HISTORICAL_DEFAULT_START,
        PipelineConfig,
        PRIORITY_QUERIES,
    )

    config = PipelineConfig(base_dir=Path(args.output))
    config.ensure_directories()

    start_date = args.start or HISTORICAL_DEFAULT_START
    end_date = args.end or HISTORICAL_DEFAULT_END
    subreddits = args.subreddits.split(
        ",") if args.subreddits else ALL_SUBREDDITS
    queries = PRIORITY_QUERIES if args.priority_queries else ALL_SEARCH_QUERIES

    stats = asyncio.run(
        collect_historical(
            config=config,
            subreddits=subreddits,
            start_date=start_date,
            end_date=end_date,
            granularity=args.granularity,
            queries=queries,
            use_priority_queries=args.priority_queries,
            resume=not getattr(args, 'no_resume', False),
        )
    )

    print("\nCollection Summary:")
    for sub, count in stats.items():
        print(f"  r/{sub}: {count:,}")


def run_crisis(args: argparse.Namespace) -> None:
    """Run crisis period data collection."""
    from scripts.collectors import collect_crisis
    from scripts.config import CRISIS_QUERIES, CRISIS_SUBREDDITS, FLASHPOINTS, PipelineConfig

    config = PipelineConfig(base_dir=Path(args.output))
    config.ensure_directories()

    subreddits = args.subreddits.split(
        ",") if args.subreddits else CRISIS_SUBREDDITS

    if args.all:
        print(f"\nCollecting ALL {len(FLASHPOINTS)} crisis periods...")
        for crisis_key, crisis_info in FLASHPOINTS.items():
            print(f"\n{'='*70}")
            print(f"Processing: {crisis_info['name']}")
            print(f"{'='*70}")

            asyncio.run(
                collect_crisis(
                    config=config,
                    start_datetime=crisis_info["start"],
                    end_datetime=crisis_info["end"],
                    subreddits=subreddits,
                    queries=CRISIS_QUERIES,
                    crisis_name=crisis_info["name"],
                )
            )
        print(f"\n{'='*70}")
        print("ALL CRISIS PERIODS COLLECTED")
        print(f"{'='*70}")
        return

    if args.name:
        if args.name not in FLASHPOINTS:
            print(f"Error: Unknown crisis '{args.name}'")
            print(f"Available: {', '.join(FLASHPOINTS.keys())}")
            sys.exit(1)
        crisis_info = FLASHPOINTS[args.name]
        start_datetime = crisis_info["start"]
        end_datetime = crisis_info["end"]
        crisis_name = crisis_info["name"]
    elif args.start and args.end:
        start_datetime = args.start
        end_datetime = args.end
        crisis_name = "Custom"
    else:
        crisis_key = "operation_sindoor"
        crisis_info = FLASHPOINTS[crisis_key]
        start_datetime = crisis_info["start"]
        end_datetime = crisis_info["end"]
        crisis_name = crisis_info["name"]
        print(f"\nUsing default crisis: {crisis_name}")

    result = asyncio.run(
        collect_crisis(
            config=config,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            subreddits=subreddits,
            queries=CRISIS_QUERIES,
            crisis_name=crisis_name,
        )
    )

    print(f"\nCollected {result['metadata']['total']:,} submissions")


def run_comments(args: argparse.Namespace) -> None:
    """Run comment collection for existing submissions."""
    from scripts.collectors import collect_comments, collect_comments_batch
    from scripts.config import PipelineConfig

    config = PipelineConfig(base_dir=Path(args.output))

    if args.file:
        submissions_file = Path(args.file)
        if not submissions_file.exists():
            submissions_file = config.submissions_dir / args.file

        if not submissions_file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        asyncio.run(
            collect_comments(
                config=config,
                submissions_file=submissions_file,
                min_comments=args.min_comments,
                max_submissions=args.max_submissions,
                resume=not getattr(args, 'no_resume', False),
            )
        )
    else:
        pattern = args.pattern or "*_filtered.json"
        asyncio.run(
            collect_comments_batch(
                config=config,
                file_pattern=pattern,
                min_comments=args.min_comments,
                max_submissions=args.max_submissions,
                resume=not getattr(args, 'no_resume', False),
            )
        )


def run_analyze(args: argparse.Namespace) -> None:
    """Run analysis on collected data."""
    from scripts.analyzers import generate_summary_report, quick_stats
    from scripts.config import PipelineConfig
    from scripts.processors import load_all_submissions
    from scripts.visualizers import generate_all_visualizations

    config = PipelineConfig(base_dir=Path(args.output))

    print("Loading submissions...")
    df = load_all_submissions(config.submissions_dir)

    if len(df) == 0:
        print("No data found. Run collection first.")
        sys.exit(1)

    print(f"Loaded {len(df):,} submissions")

    quick_stats(df)

    if not args.skip_visualizations:
        generate_all_visualizations(df, config)

    if not args.skip_report:
        report = generate_summary_report(df, config)
        print(report)


def run_export(args: argparse.Namespace) -> None:
    """Export data to Parquet format."""
    from scripts.config import PipelineConfig
    from scripts.processors import export_sub_to_parquet, export_comments_to_parquet

    config = PipelineConfig(base_dir=Path(args.output))
    config.exports_dir.mkdir(parents=True, exist_ok=True)

    # Export submissions
    submissions_output = config.exports_dir / "all_submissions.parquet"
    submissions_df = export_sub_to_parquet(config.submissions_dir, submissions_output)

    # Export comments
    comments_output = config.exports_dir / "all_comments.parquet"
    comments_df = export_comments_to_parquet(config.comments_dir, comments_output)


def list_flashpoints() -> None:
    """List all available flashpoints."""
    from scripts.config import FLASHPOINTS

    print("\nAvailable Crisis Periods (FLASHPOINTS):")
    print(f"{'='*70}")
    for key, info in FLASHPOINTS.items():
        print(f"\n  {key}:")
        print(f"    Name:     {info['name']}")
        print(f"    Start:    {info['start']}")
        print(f"    End:      {info['end']}")
        print(f"    Priority: {info['priority']}")
    print(f"\n{'='*70}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="India-Pakistan Reddit Data Collection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py historical                    # Collect 2019-2024 data (default)
  uv run main.py historical --start 2024-01-01 --end 2024-12-31
  uv run main.py crisis                        # Collect Operation Sindoor (default)
  uv run main.py crisis --name pulwama_attack  # Collect specific crisis
  uv run main.py crisis --all                  # Collect ALL crisis periods
  uv run main.py comments                      # Collect comments for all submissions
  uv run main.py comments --file india_2024.json
  uv run main.py analyze
  uv run main.py export
  uv run main.py list-flashpoints              # Show available crisis periods
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        default="./data",
        help="Base output directory (default: ./data)",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Historical collection
    hist_parser = subparsers.add_parser(
        "historical", help="Collect historical data (default: 2019-2024)"
    )
    hist_parser.add_argument(
        "--start", "-s", help="Start date YYYY-MM-DD (default: 2019-01-01)"
    )
    hist_parser.add_argument(
        "--end", "-e", help="End date YYYY-MM-DD (default: 2024-12-31)"
    )
    hist_parser.add_argument(
        "--granularity",
        "-g",
        default="monthly",
        choices=["monthly", "daily", "hourly"],
        help="Time window granularity (default: monthly)",
    )
    hist_parser.add_argument(
        "--subreddits",
        help="Comma-separated list of subreddits (default: all 14 configured)",
    )
    hist_parser.add_argument(
        "--priority-queries",
        action="store_true",
        default=True,
        help="Use priority queries only (default: True)",
    )
    hist_parser.add_argument(
        "--all-queries",
        action="store_false",
        dest="priority_queries",
        help="Use all search queries",
    )
    hist_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already collected subreddits (default: resume enabled)",
    )

    # Crisis collection
    crisis_parser = subparsers.add_parser(
        "crisis", help="Collect crisis period data (default: Operation Sindoor)"
    )
    crisis_parser.add_argument(
        "--name",
        "-n",
        help="Crisis name from FLASHPOINTS (e.g., pulwama_attack, operation_sindoor)",
    )
    crisis_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Collect ALL crisis periods from FLASHPOINTS",
    )
    crisis_parser.add_argument(
        "--start", "-s", help="Custom start datetime (ISO format)"
    )
    crisis_parser.add_argument(
        "--end", "-e", help="Custom end datetime (ISO format)"
    )
    crisis_parser.add_argument(
        "--subreddits",
        help="Comma-separated list of subreddits (default: crisis subreddits)",
    )

    # Comment collection
    comment_parser = subparsers.add_parser(
        "comments", help="Collect comments (default: all submission files)"
    )
    comment_parser.add_argument(
        "--file", "-f", help="Specific submissions JSON file"
    )
    comment_parser.add_argument(
        "--pattern",
        "-p",
        default="*_filtered.json",
        help="Glob pattern for submission files (default: *_filtered.json)",
    )
    comment_parser.add_argument(
        "--min-comments",
        type=int,
        default=5,
        help="Minimum comments threshold (default: 5)",
    )
    comment_parser.add_argument(
        "--max-submissions",
        type=int,
        default=10000,
        help="Maximum submissions to process per file (default: 10000)",
    )
    comment_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already processed posts (default: resume enabled)",
    )

    # Analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze collected data"
    )
    analyze_parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip generating visualizations",
    )
    analyze_parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip generating summary report",
    )

    # Export
    subparsers.add_parser("export", help="Export data to Parquet")

    # List flashpoints
    subparsers.add_parser("list-flashpoints",
                          help="List available crisis periods")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    command_handlers = {
        "historical": run_historical,
        "crisis": run_crisis,
        "comments": run_comments,
        "analyze": run_analyze,
        "export": run_export,
        "list-flashpoints": lambda _: list_flashpoints(),
    }

    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
