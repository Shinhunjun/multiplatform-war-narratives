"""
BERTopic-based topic modeling with temporal analysis.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def create_bertopic_model(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_topics: Optional[int] = None,
    min_topic_size: int = 50,
    n_gram_range: Tuple[int, int] = (1, 2),
):
    """
    Create a BERTopic model with specified configuration.

    Args:
        embedding_model: Sentence transformer model name
        n_topics: Number of topics (None for auto)
        min_topic_size: Minimum documents per topic
        n_gram_range: N-gram range for topic representation

    Returns:
        Configured BERTopic model
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    # Embedding model
    sentence_model = SentenceTransformer(embedding_model)

    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Vectorizer for topic representation
    vectorizer_model = CountVectorizer(
        ngram_range=n_gram_range,
        stop_words="english",
        min_df=2,  # Lower threshold for smaller datasets
    )

    # Representation model
    representation_model = KeyBERTInspired()

    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=n_topics,
        top_n_words=10,
        verbose=True,
    )

    return topic_model


def fit_topics(
    df: pd.DataFrame,
    text_column: str = "text",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_topics: Optional[int] = None,
    min_topic_size: int = 50,
) -> Tuple[pd.DataFrame, "BERTopic", np.ndarray]:
    """
    Fit BERTopic model on DataFrame.

    Returns:
    - DataFrame with topic assignments
    - Fitted BERTopic model
    - Document embeddings
    """
    df = df.copy()
    texts = df[text_column].fillna("").tolist()

    print(f"Fitting BERTopic on {len(texts):,} documents...")

    # Create and fit model
    topic_model = create_bertopic_model(
        embedding_model=embedding_model,
        n_topics=n_topics,
        min_topic_size=min_topic_size,
    )

    topics, probs = topic_model.fit_transform(texts)

    # Get embeddings
    embeddings = topic_model._extract_embeddings(texts)

    # Add to DataFrame
    df["topic_id"] = topics
    df["topic_prob"] = probs

    # Add topic labels
    topic_info = topic_model.get_topic_info()
    topic_labels = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df["topic_label"] = df["topic_id"].map(topic_labels)

    print(f"Found {len(topic_info) - 1} topics (excluding outliers)")

    return df, topic_model, embeddings


def get_topic_info(topic_model) -> pd.DataFrame:
    """Get topic information including keywords."""
    topic_info = topic_model.get_topic_info()
    return topic_info


def get_topic_keywords(topic_model, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]:
    """Get keywords for a specific topic."""
    return topic_model.get_topic(topic_id)[:n_words]


def topics_over_time(
    topic_model,
    docs: List[str],
    timestamps: List,
    nr_bins: int = 20,
) -> pd.DataFrame:
    """
    Analyze how topics change over time.

    Returns DataFrame with topic distributions per time bin.
    """
    topics_over_time = topic_model.topics_over_time(
        docs,
        timestamps,
        nr_bins=nr_bins,
    )
    return topics_over_time


def get_representative_docs(
    topic_model,
    topic_id: int,
    df: pd.DataFrame,
    n_docs: int = 10,
) -> pd.DataFrame:
    """Get representative documents for a topic."""
    topic_docs = df[df["topic_id"] == topic_id]
    if len(topic_docs) == 0:
        return pd.DataFrame()

    # Sort by probability
    topic_docs = topic_docs.sort_values("topic_prob", ascending=False)
    return topic_docs.head(n_docs)


def aggregate_topics_by_group(
    df: pd.DataFrame,
    group_by: List[str] = ["subreddit", "year_month"],
) -> pd.DataFrame:
    """
    Aggregate topic distributions by groups.

    Returns DataFrame with topic counts per group.
    """
    # Count topics per group
    topic_counts = df.groupby(group_by + ["topic_id"]).size().reset_index(name="count")

    # Calculate proportions within each group
    group_totals = topic_counts.groupby(group_by)["count"].transform("sum")
    topic_counts["proportion"] = topic_counts["count"] / group_totals

    return topic_counts


def save_topic_model(topic_model, path: str) -> None:
    """Save BERTopic model to disk."""
    topic_model.save(path)


def load_topic_model(path: str):
    """Load BERTopic model from disk."""
    from bertopic import BERTopic
    return BERTopic.load(path)
