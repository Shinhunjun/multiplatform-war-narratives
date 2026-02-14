"""
Pipeline configuration for daily ETL jobs.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PipelineConfig:
    """Configuration for the daily ETL pipeline."""

    # --- Paths ---
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "pipeline_data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "venezuela-us-reddit-discourse" / "analysis" / "outputs"

    # --- GCS (for Cloud Run deployment) ---
    gcs_bucket: str = field(
        default_factory=lambda: os.environ.get("GCS_BUCKET", "")
    )
    gcs_data_prefix: str = "pipeline_data"

    # --- Arctic Shift (Reddit) ---
    arctic_sleep_sec: float = 1.0
    arctic_backoff_sec: float = 10.0
    arctic_max_retries: int = 5
    arctic_timeout: int = 60
    arctic_task_num: int = 1

    # Subreddits to monitor
    subreddits: List[str] = field(
        default_factory=lambda: [
            "venezuela", "vzla",
            "politics", "news", "worldnews",
            "Conservative", "Libertarian",
            "neoliberal", "socialism",
            "LatinAmerica", "geopolitics",
        ]
    )

    # Search queries for Arctic Shift
    reddit_queries: List[str] = field(
        default_factory=lambda: [
            "Venezuela",
            "Maduro",
            "Venezuela US",
            "Venezuela sanctions",
            "Guaido",
            "Venezuelan crisis",
            "Venezuela oil",
            "Caracas",
            "Venezuela election",
            "Venezuela humanitarian",
        ]
    )

    # Venezuela-specific subs (no keyword filtering needed)
    venezuela_subreddits: List[str] = field(
        default_factory=lambda: ["venezuela", "vzla"]
    )

    # --- GDELT BigQuery ---
    gcp_project: str = field(
        default_factory=lambda: os.environ.get("GCP_PROJECT", "")
    )

    gdelt_keywords: List[str] = field(
        default_factory=lambda: [
            "venezuela", "maduro", "caracas",
            "guaido", "pdvsa", "citgo",
        ]
    )
    gdelt_max_articles: int = 500

    # --- News Scraper ---
    scraper_timeout: int = 15  # seconds per article
    scraper_max_concurrent: int = 5

    # --- Analysis Models ---
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    min_cluster_size: int = 50
    min_samples: int = 10
    n_topics: int = 15

    # --- Pipeline Settings ---
    incremental: bool = True  # append to existing data instead of reprocessing all
    lookback_days: int = 1    # how many days back to fetch

    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        for d in [
            self.raw_dir / "reddit" / "submissions",
            self.raw_dir / "reddit" / "comments",
            self.raw_dir / "gdelt",
            self.raw_dir / "news",
            self.processed_dir / "reddit",
            self.processed_dir / "gdelt",
        ]:
            d.mkdir(parents=True, exist_ok=True)
