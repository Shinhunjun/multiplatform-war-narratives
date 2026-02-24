"""
Main entry point for the Venezuela TikTok Data Collection Pipeline.

Usage:
    python main.py videos
    python main.py videos --start 20190101 --end 20260214
    python main.py crisis --all
    python main.py crisis --name guaido_recognition_2019
    python main.py comments
    python main.py comments --file videos_20240720_20240818.json
    python main.py export
    python main.py list-flashpoints
"""

import argparse
import sys
from pathlib import Path


def run_videos(args: argparse.Namespace) -> None:
    """Run historical video collection."""
    from scripts.auth import create_client
    from scripts.collectors import collect_videos_historical
    from scripts.config import (
        ALL_HASHTAGS,
        ALL_SEARCH_QUERIES,
        HISTORICAL_DEFAULT_END,
        HISTORICAL_DEFAULT_START,
        PipelineConfig,
        PRIORITY_HASHTAGS,
        PRIORITY_QUERIES,
    )

    config = PipelineConfig(base_dir=Path(args.output))
    config.ensure_directories()
    api = create_client(config)

    start_date = args.start or HISTORICAL_DEFAULT_START
    end_date = args.end or HISTORICAL_DEFAULT_END
    keywords = PRIORITY_QUERIES if args.priority_queries else ALL_SEARCH_QUERIES
    hashtags = PRIORITY_HASHTAGS if args.priority_queries else ALL_HASHTAGS

    stats = collect_videos_historical(
        config=config,
        api=api,
        start_date=start_date,
        end_date=end_date,
        keywords=keywords,
        hashtags=hashtags,
        resume=not getattr(args, "no_resume", False),
    )

    print("\nCollection Summary:")
    total = sum(stats.values())
    windows_with_data = sum(1 for v in stats.values() if v > 0)
    print(f"  Total videos: {total:,}")
    print(f"  Windows with data: {windows_with_data} / {len(stats)}")


def run_crisis(args: argparse.Namespace) -> None:
    """Run crisis period video collection."""
    from scripts.auth import create_client
    from scripts.collectors import collect_videos_crisis
    from scripts.config import FLASHPOINTS, PipelineConfig

    config = PipelineConfig(base_dir=Path(args.output))
    config.ensure_directories()
    api = create_client(config)

    if args.all:
        stats = collect_videos_crisis(
            config=config, api=api, collect_all=True,
        )
    elif args.name:
        stats = collect_videos_crisis(
            config=config, api=api, crisis_key=args.name,
        )
    else:
        # Default: most recent critical crisis
        stats = collect_videos_crisis(
            config=config, api=api, crisis_key="election_2024",
        )
        print("(Used default crisis: election_2024)")

    total = sum(stats.values())
    print(f"\nTotal crisis videos: {total:,}")


def run_comments(args: argparse.Namespace) -> None:
    """Run comment collection for collected videos."""
    from scripts.auth import create_client
    from scripts.collectors import collect_comments
    from scripts.config import PipelineConfig

    config = PipelineConfig(base_dir=Path(args.output))
    api = create_client(config)

    videos_file = None
    if args.file:
        videos_file = Path(args.file)
        if not videos_file.exists():
            videos_file = config.videos_dir / args.file
        if not videos_file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

    stats = collect_comments(
        config=config,
        api=api,
        videos_file=videos_file,
        min_comments=args.min_comments,
        max_videos=args.max_videos,
        resume=not getattr(args, "no_resume", False),
    )

    total = sum(stats.values())
    print(f"\nTotal comments collected: {total:,}")


def run_export(args: argparse.Namespace) -> None:
    """Export collected data to Parquet format."""
    from scripts.config import PipelineConfig
    from scripts.processors import export_to_parquet

    config = PipelineConfig(base_dir=Path(args.output))

    videos_df, comments_df = export_to_parquet(config)

    if videos_df is not None:
        print(f"\nVideos: {len(videos_df):,} records")
        if "region_code" in videos_df.columns:
            print("\nTop regions:")
            print(videos_df["region_code"].value_counts().head(10).to_string())
    if comments_df is not None:
        print(f"\nComments: {len(comments_df):,} records")


def list_flashpoints() -> None:
    """List all available flashpoints."""
    from scripts.config import FLASHPOINTS

    print(f"\nAvailable Crisis Periods (FLASHPOINTS):")
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
        description="Venezuela TikTok Data Collection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py videos                          # Collect 2013-2026 (default)
  python main.py videos --start 20240101 --end 20241231
  python main.py crisis --all                    # Collect ALL crisis periods
  python main.py crisis --name election_2024     # Specific crisis
  python main.py comments                        # Comments for all videos
  python main.py comments --file videos_20240720_20240818.json
  python main.py export                          # Export to Parquet
  python main.py list-flashpoints                # Show crisis periods
        """,
    )

    parser.add_argument(
        "--output", "-o", default="./data",
        help="Base output directory (default: ./data)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Video collection
    vid_parser = subparsers.add_parser(
        "videos", help="Collect historical video data",
    )
    vid_parser.add_argument(
        "--start", "-s", help="Start date YYYYMMDD (default: 20130101)",
    )
    vid_parser.add_argument(
        "--end", "-e", help="End date YYYYMMDD (default: 20260214)",
    )
    vid_parser.add_argument(
        "--priority-queries", action="store_true", default=True,
        help="Use priority queries only (default: True)",
    )
    vid_parser.add_argument(
        "--all-queries", action="store_false", dest="priority_queries",
        help="Use all search queries",
    )
    vid_parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't skip already collected windows",
    )

    # Crisis collection
    crisis_parser = subparsers.add_parser(
        "crisis", help="Collect crisis period data",
    )
    crisis_parser.add_argument(
        "--name", "-n", help="Crisis key from FLASHPOINTS",
    )
    crisis_parser.add_argument(
        "--all", "-a", action="store_true",
        help="Collect ALL crisis periods",
    )

    # Comment collection
    comment_parser = subparsers.add_parser(
        "comments", help="Collect comments for videos",
    )
    comment_parser.add_argument(
        "--file", "-f", help="Specific videos JSON file",
    )
    comment_parser.add_argument(
        "--min-comments", type=int, default=5,
        help="Minimum comment_count threshold (default: 5)",
    )
    comment_parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Maximum videos to process",
    )
    comment_parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't skip already processed videos",
    )

    # Export
    subparsers.add_parser("export", help="Export data to Parquet")

    # List flashpoints
    subparsers.add_parser("list-flashpoints", help="List available crisis periods")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    command_handlers = {
        "videos": run_videos,
        "crisis": run_crisis,
        "comments": run_comments,
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
