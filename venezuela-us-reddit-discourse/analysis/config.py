"""
Configuration for the analysis pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AnalysisConfig:
    """Central configuration for analysis pipeline."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data-collection" / "data"

    @property
    def submissions_dir(self) -> Path:
        return self.data_dir / "submissions"

    @property
    def comments_dir(self) -> Path:
        return self.data_dir / "comments"

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "analysis" / "outputs"

    # Model settings
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Processing settings
    batch_size: int = 64  # Larger batch for GPU
    max_seq_length: int = 512

    # Clustering settings
    min_cluster_size: int = 50
    min_samples: int = 10

    # Topic modeling
    n_topics: int = 15  # Auto if None

    # Temporal analysis
    time_granularity: str = "month"  # month, quarter, year

    # Sampling for summarization
    samples_per_cluster: int = 20

    def ensure_directories(self) -> None:
        """Create all output directories."""
        for subdir in ["sentiment", "topics", "clusters", "visualizations"]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)


# Crisis periods for temporal analysis
CRISIS_PERIODS = {
    "maduro_inauguration_2013": ("2013-04-14", "2013-04-30"),
    "protests_2014": ("2014-02-01", "2014-05-31"),
    "oil_crash_2014": ("2014-11-01", "2015-02-28"),
    "trump_sanctions_2017": ("2017-08-01", "2017-09-30"),
    "maduro_reelection_2018": ("2018-05-15", "2018-05-31"),
    "guaido_recognition_2019": ("2019-01-20", "2019-02-28"),
    "failed_uprising_2019": ("2019-04-28", "2019-05-05"),
    "biden_policy_2021": ("2021-01-20", "2021-03-31"),
    "election_2024": ("2024-07-20", "2024-08-15"),
    "gonzalez_exile_2024": ("2024-09-01", "2024-09-15"),
}

# Subreddit groupings
SUBREDDIT_GROUPS = {
    "venezuela": ["venezuela", "vzla"],
    "us_mainstream": ["politics", "news", "worldnews"],
    "us_conservative": ["Conservative", "Libertarian"],
    "us_progressive": ["neoliberal", "socialism"],
    "regional": ["LatinAmerica", "geopolitics"],
}
