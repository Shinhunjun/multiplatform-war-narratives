"""
BERTopic-based topic modeling with temporal analysis.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def create_bertopic_model(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_topics: Optional[int] = None,
    min_topic_size: int = 50,
    n_gram_range: Tuple[int, int] = (1, 2),
) -> Any:
    """Create a BERTopic model with specified configuration.
    
    Args:
        embedding_model (str): Embedding model identifier. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        n_topics (Optional[int]): Requested number of topics. Defaults to None.
        min_topic_size (int): Minimum topic size used by BERTopic/HDBSCAN. Defaults to 50.
        n_gram_range (Tuple[int, int]): N-gram range for vectorization. Defaults to (1, 2).
    
    Returns:
        Any: Object returned by the underlying library or runtime path.
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
        min_df=1,
        max_df=0.95,
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
    """Fit BERTopic model on DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
        text_column (str): Column name containing article text. Defaults to 'text'.
        embedding_model (str): Embedding model identifier. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        n_topics (Optional[int]): Requested number of topics. Defaults to None.
        min_topic_size (int): Minimum topic size used by BERTopic/HDBSCAN. Defaults to 50.
    
    Returns:
        Tuple[pd.DataFrame, 'BERTopic', np.ndarray]: Processed pandas DataFrame.
    """
    df = df.copy()
    texts = df[text_column].fillna("").tolist()

    print(f"Fitting BERTopic on {len(texts):,} documents...")

    # Use a smaller effective cluster size for sample/testing runs.
    effective_min_topic_size = min_topic_size
    if len(texts) < 5000:
        effective_min_topic_size = max(10, min(min_topic_size, max(10, len(texts) // 20)))
        if effective_min_topic_size != min_topic_size:
            print(
                f"Adjusted min_topic_size from {min_topic_size} "
                f"to {effective_min_topic_size} for dataset size {len(texts):,}"
            )

    # Create and fit model
    topic_model = create_bertopic_model(
        embedding_model=embedding_model,
        n_topics=n_topics,
        min_topic_size=effective_min_topic_size,
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


def get_topic_info(topic_model: Any) -> pd.DataFrame:
    """Get topic information including keywords.
    
    Args:
        topic_model (Any): Fitted BERTopic model instance.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    topic_info = topic_model.get_topic_info()
    return topic_info


def get_topic_keywords(topic_model: Any, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]:
    """Get keywords for a specific topic.
    
    Args:
        topic_model (Any): Fitted BERTopic model instance.
        topic_id (int): Topic identifier.
        n_words (int): Number of words to return. Defaults to 10.
    
    Returns:
        List[Tuple[str, float]]: List result produced by this function.
    """
    return topic_model.get_topic(topic_id)[:n_words]


def topics_over_time(
    topic_model: Any,
    docs: List[str],
    timestamps: List[Any],
    nr_bins: int = 20,
) -> pd.DataFrame:
    """Analyze how topics change over time.
    
    Args:
        topic_model (Any): Fitted BERTopic model instance.
        docs (List[str]): Document list used for temporal topic analysis.
        timestamps (List[Any]): Timestamp sequence aligned to documents.
        nr_bins (int): Number of bins for temporal aggregation. Defaults to 20.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    topics_over_time = topic_model.topics_over_time(
        docs,
        timestamps,
        nr_bins=nr_bins,
    )
    return topics_over_time


def get_representative_docs(
    topic_model: Any,
    topic_id: int,
    df: pd.DataFrame,
    n_docs: int = 10,
) -> pd.DataFrame:
    """Get representative documents for a topic.
    
    Args:
        topic_model (Any): Fitted BERTopic model instance.
        topic_id (int): Topic identifier.
        df (pd.DataFrame): Input DataFrame to process.
        n_docs (int): Number of representative documents to return. Defaults to 10.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
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
    """Aggregate topic distributions by groups.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
        group_by (List[str]): Grouping columns used for aggregation. Defaults to ['subreddit', 'year_month'].
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    # Count topics per group
    topic_counts = df.groupby(group_by + ["topic_id"]).size().reset_index(name="count")

    # Calculate proportions within each group
    group_totals = topic_counts.groupby(group_by)["count"].transform("sum")
    topic_counts["proportion"] = topic_counts["count"] / group_totals

    return topic_counts


def save_topic_model(topic_model: Any, path: str) -> None:
    """Save BERTopic model to disk.
    
    Args:
        topic_model (Any): Fitted BERTopic model instance.
        path (str): Filesystem path value.
    
    Returns:
        None: No return value.
    """
    topic_model.save(path)


def load_topic_model(path: str) -> Any:
    """Load BERTopic model from disk.
    
    Args:
        path (str): Filesystem path value.
    
    Returns:
        Any: Object returned by the underlying library or runtime path.
    """
    from bertopic import BERTopic
    return BERTopic.load(path)
