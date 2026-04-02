"""
Venezuela-US GDELT discourse analysis pipeline.

Modules:
- sentiment: RoBERTa sentiment analysis on scraped article text
- topic: BERTopic topic modeling
- clustering: Embedding + HDBSCAN clustering with temporal tracking
"""

from .config import AnalysisConfig, CRISIS_PERIODS, EVENT_CATEGORY_MAP
from .data_loader import load_all_data, load_gdelt_events, load_url_lookup

__all__ = [
    "AnalysisConfig",
    "CRISIS_PERIODS",
    "EVENT_CATEGORY_MAP",
    "load_all_data",
    "load_gdelt_events",
    "load_url_lookup",
]
