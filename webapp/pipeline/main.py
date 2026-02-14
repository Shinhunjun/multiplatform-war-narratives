"""
Daily ETL Pipeline - Main Orchestrator

Runs as a Cloud Run Job triggered by Cloud Scheduler (daily).

Stages:
1. COLLECT  - Fetch new data from Reddit API and GDELT BigQuery
2. SCRAPE   - Extract text from GDELT news article URLs
3. PREPROCESS - Clean and filter raw data
4. ANALYZE  - Run sentiment, topic modeling on new data
5. UPDATE   - Merge results with existing analysis outputs

Usage:
    python -m webapp.pipeline.main                   # Run full pipeline
    python -m webapp.pipeline.main --stage collect    # Run specific stage
    python -m webapp.pipeline.main --date 2026-02-14  # Specific date
    python -m webapp.pipeline.main --reddit-only      # Skip GDELT
    python -m webapp.pipeline.main --gdelt-only       # Skip Reddit
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def stage_collect_reddit(config: PipelineConfig, run_date: str) -> dict:
    """Stage 1a: Collect Reddit data."""
    from .collectors.reddit import RedditCollector

    logger.info("=" * 50)
    logger.info("STAGE 1a: COLLECT REDDIT DATA")
    logger.info("=" * 50)

    collector = RedditCollector(config)
    return collector.run(run_date)


def stage_collect_gdelt(config: PipelineConfig, run_date: str) -> dict:
    """Stage 1b: Collect GDELT data."""
    from .collectors.gdelt import GDELTCollector

    logger.info("=" * 50)
    logger.info("STAGE 1b: COLLECT GDELT DATA")
    logger.info("=" * 50)

    collector = GDELTCollector(config)
    return collector.run(run_date)


def stage_scrape(config: PipelineConfig, article_urls: list, run_date: str) -> dict:
    """Stage 2: Scrape news articles."""
    from .collectors.scraper import ArticleScraper

    logger.info("=" * 50)
    logger.info("STAGE 2: SCRAPE NEWS ARTICLES")
    logger.info("=" * 50)

    if not article_urls:
        logger.info("No article URLs to scrape")
        return {"articles_count": 0}

    scraper = ArticleScraper(config)
    return scraper.run(article_urls, run_date)


def stage_preprocess(config: PipelineConfig, run_date: str) -> dict:
    """Stage 3: Preprocess raw data."""
    from .processing.preprocessor import Preprocessor

    logger.info("=" * 50)
    logger.info("STAGE 3: PREPROCESS DATA")
    logger.info("=" * 50)

    preprocessor = Preprocessor(config)
    return preprocessor.run(run_date)


def stage_analyze(config: PipelineConfig, run_date: str) -> dict:
    """Stage 4: Run analysis on preprocessed data."""
    from .processing.analyzer import IncrementalAnalyzer

    logger.info("=" * 50)
    logger.info("STAGE 4: RUN ANALYSIS")
    logger.info("=" * 50)

    analyzer = IncrementalAnalyzer(config)

    # Load preprocessed Reddit data
    reddit_path = config.processed_dir / "reddit" / f"reddit_{run_date}.parquet"
    if reddit_path.exists():
        import pandas as pd
        reddit_df = pd.read_parquet(reddit_path)
        results = analyzer.run_and_update(reddit_df)
    else:
        logger.info("No preprocessed Reddit data found")
        results = {}

    return results


def run_pipeline(
    config: PipelineConfig,
    run_date: str,
    stages: list = None,
    reddit_only: bool = False,
    gdelt_only: bool = False,
) -> dict:
    """Run the full ETL pipeline."""
    all_stages = stages or ["collect", "scrape", "preprocess", "analyze"]
    report = {"run_date": run_date, "stages": {}}

    article_urls = []

    # COLLECT
    if "collect" in all_stages:
        if not gdelt_only:
            try:
                reddit_result = stage_collect_reddit(config, run_date)
                report["stages"]["collect_reddit"] = reddit_result
            except Exception as e:
                logger.error(f"Reddit collection failed: {e}")
                report["stages"]["collect_reddit"] = {"error": str(e)}

        if not reddit_only:
            try:
                gdelt_result = stage_collect_gdelt(config, run_date)
                report["stages"]["collect_gdelt"] = gdelt_result
                article_urls = gdelt_result.get("article_urls", [])
            except Exception as e:
                logger.error(f"GDELT collection failed: {e}")
                report["stages"]["collect_gdelt"] = {"error": str(e)}

    # SCRAPE
    if "scrape" in all_stages and not reddit_only:
        try:
            scrape_result = stage_scrape(config, article_urls, run_date)
            report["stages"]["scrape"] = scrape_result
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            report["stages"]["scrape"] = {"error": str(e)}

    # PREPROCESS
    if "preprocess" in all_stages:
        try:
            preprocess_result = stage_preprocess(config, run_date)
            report["stages"]["preprocess"] = preprocess_result
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            report["stages"]["preprocess"] = {"error": str(e)}

    # ANALYZE
    if "analyze" in all_stages:
        try:
            analyze_result = stage_analyze(config, run_date)
            report["stages"]["analyze"] = analyze_result
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            report["stages"]["analyze"] = {"error": str(e)}

    return report


def main():
    parser = argparse.ArgumentParser(description="Daily ETL Pipeline")
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Run date (YYYY-MM-DD, default: today UTC)",
    )
    parser.add_argument(
        "--stage",
        choices=["collect", "scrape", "preprocess", "analyze"],
        help="Run a specific stage only",
    )
    parser.add_argument("--reddit-only", action="store_true", help="Skip GDELT")
    parser.add_argument("--gdelt-only", action="store_true", help="Skip Reddit")
    parser.add_argument(
        "--lookback",
        type=int,
        default=1,
        help="Days to look back for data collection (default: 1)",
    )

    args = parser.parse_args()

    config = PipelineConfig()
    config.lookback_days = args.lookback
    config.ensure_directories()

    stages = [args.stage] if args.stage else None

    logger.info(f"Pipeline starting for date: {args.date}")
    logger.info(f"Stages: {stages or 'all'}")
    logger.info(f"Lookback: {config.lookback_days} day(s)")

    report = run_pipeline(
        config,
        run_date=args.date,
        stages=stages,
        reddit_only=args.reddit_only,
        gdelt_only=args.gdelt_only,
    )

    # Save report
    report_dir = config.data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"pipeline_report_{args.date}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Pipeline complete. Report: {report_path}")
    logger.info(json.dumps(report, indent=2, default=str))

    # Return non-zero if any stage had errors
    has_errors = any(
        "error" in v for v in report.get("stages", {}).values() if isinstance(v, dict)
    )
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
