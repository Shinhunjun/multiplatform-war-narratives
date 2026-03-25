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
        """Return the configured project data directory path.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.base_dir / "data"

    @property
    def analysis_ready_dir(self) -> Path:
        """Return the analysis-ready dataset directory path.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.data_dir / self.analysis_ready_subdir

    @property
    def preprocessing_dir(self) -> Path:
        """Return the preprocessing artifact directory path.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.data_dir / "preprocessing"

    @property
    def output_dir(self) -> Path:
        """Return the analysis output directory path.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.base_dir / "analysis" / "outputs"

    @property
    def gdelt_csv_path(self) -> Path:
        """Return the path to the primary scraped GDELT CSV file.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.data_dir / self.gdelt_file

    @property
    def analysis_events_path(self) -> Path:
        """Return the path to the analysis-ready event parquet file.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.analysis_ready_dir / self.analysis_events_file

    @property
    def analysis_url_content_path(self) -> Path:
        """Return the path to the analysis-ready URL-content parquet file.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.analysis_ready_dir / self.analysis_url_content_file

    @property
    def url_lookup_path(self) -> Path:
        """Return the path to the URL lookup table produced in preprocessing.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.preprocessing_dir / self.url_lookup_file

    @property
    def relevance_tokens_path(self) -> Path:
        """Return the path to token relevance scores used by analysis modules.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.preprocessing_dir / self.relevance_tokens_file

    @property
    def relevant_terms_path(self) -> Path:
        """Return the path to curated relevant terms used for interpretation.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.preprocessing_dir / self.relevant_terms_file

    @property
    def redirect_scores_path(self) -> Path:
        """Return the path to redirect-duplicate cluster score output.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.preprocessing_dir / self.redirect_scores_file

    @property
    def redirect_clusters_path(self) -> Path:
        """Return the path to redirect-duplicate cluster detail output.
        
        Returns:
            Path: Filesystem path value.
        """
        return self.preprocessing_dir / self.redirect_clusters_file

    # Input filenames
    analysis_ready_subdir: str = "analysis_ready"
    analysis_events_file: str = "analysis_events.parquet"
    analysis_url_content_file: str = "analysis_url_content.parquet"
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
    use_analysis_ready_parquet: bool = True
    require_successful_scrape: bool = True
    require_analysis_include: bool = True
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
        """Create all output directories.
        
        Returns:
            None: No return value.
        """
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
