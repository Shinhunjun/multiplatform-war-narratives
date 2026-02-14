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

    # --- Reddit API (PRAW) ---
    reddit_client_id: str = field(
        default_factory=lambda: os.environ.get("REDDIT_CLIENT_ID", "")
    )
    reddit_client_secret: str = field(
        default_factory=lambda: os.environ.get("REDDIT_CLIENT_SECRET", "")
    )
    reddit_user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "REDDIT_USER_AGENT",
            "venezuela-narrative-analysis/1.0 (capstone research project)",
        )
    )

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

    # Keywords for filtering Reddit posts
    reddit_keywords: List[str] = field(
        default_factory=lambda: [
            "venezuela", "maduro", "guaidó", "guaido", "caracas",
            "pdvsa", "citgo", "petro", "bolivar",
            "sanctions", "oil embargo",
        ]
    )

    # Max posts to fetch per subreddit per run
    reddit_posts_per_sub: int = 100
    reddit_comments_per_post: int = 50

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
