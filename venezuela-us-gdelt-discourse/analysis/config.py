"""
Configuration for the GDELT discourse analysis pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class AnalysisConfig:
    """Central configuration for analysis pipeline."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        """Execute data_dir."""
        return self.base_dir / "data"

    @property
    def preprocessing_dir(self) -> Path:
        """Execute preprocessing_dir."""
        return self.base_dir / "preprocessing"

    @property
    def output_dir(self) -> Path:
        """Execute output_dir."""
        return self.base_dir / "analysis" / "outputs"

    @property
    def gdelt_csv_path(self) -> Path:
        """Execute gdelt_csv_path."""
        return self.data_dir / self.gdelt_file

    @property
    def url_lookup_path(self) -> Path:
        """Execute url_lookup_path."""
        return self.preprocessing_dir / self.url_lookup_file

    @property
    def relevance_tokens_path(self) -> Path:
        """Execute relevance_tokens_path."""
        return self.preprocessing_dir / self.relevance_tokens_file

    @property
    def relevant_terms_path(self) -> Path:
        """Execute relevant_terms_path."""
        return self.preprocessing_dir / self.relevant_terms_file

    @property
    def redirect_scores_path(self) -> Path:
        """Execute redirect_scores_path."""
        return self.preprocessing_dir / self.redirect_scores_file

    @property
    def redirect_clusters_path(self) -> Path:
        """Execute redirect_clusters_path."""
        return self.preprocessing_dir / self.redirect_clusters_file

    # Input filenames
    gdelt_file: str = "gdelt_scraped.csv"
    url_lookup_file: str = "url_lookup.csv"
    relevance_tokens_file: str = "text_relevance_tokens.csv"
    relevant_terms_file: str = "relevant_terms.csv"
    redirect_scores_file: str = "redirect_duplicate_cluster_scores.csv"
    redirect_clusters_file: str = "redirect_duplicate_clusters.csv"

    # Model settings
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Processing settings
    batch_size: int = 64
    max_seq_length: int = 512

    # Data filters
    require_successful_scrape: bool = True
    min_doc_relevance_score: Optional[float] = None
    exclude_suspect_redirect_content: bool = False

    # Main grouping for aggregations
    source_group_column: str = "source_domain"

    # Clustering settings
    min_cluster_size: int = 50
    min_samples: int = 10

    # Topic modeling
    n_topics: int = 15

    # Temporal analysis
    time_granularity: str = "month"  # month, quarter, year

    # Sampling for summarization
    samples_per_cluster: int = 20

    def ensure_directories(self) -> None:
        """Create all output directories."""
        for subdir in ["sentiment", "topics", "clusters", "visualizations"]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)


# Crisis periods for temporal analysis
CRISIS_PERIODS: Dict[str, Tuple[str, str]] = {
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

EVENT_CATEGORY_MAP = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict",
}
