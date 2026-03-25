"""
Fit BERTopic independently for each month.

Instead of fitting one global model and slicing by month, this script
fits a separate BERTopic model per month so each month gets its own
unique topics and keywords — revealing how discourse changes over time.

Uses pre-computed embeddings (document_embeddings.npy) for speed.

Output:
  outputs/topics/monthly_topics_fitted.parquet
  outputs_news/topics/monthly_topics_fitted.parquet

Schema: year_month, topic_id, keywords, count, proportion
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Stopwords (same as global model)
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _sw
    STOPWORDS = list(set(_sw.words("english")) | set(_sw.words("spanish")))
except Exception:
    STOPWORDS = None


_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_embedding_model_cache = None


def _get_embedding_model():
    """Lazily load and cache the sentence transformer model."""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model_cache = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedding_model_cache


def _create_model(n_docs: int):
    """Create a BERTopic model with parameters scaled to month size."""
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    # Adaptive params from hyperparameter experiment (experiment_clustering.py)
    min_topic_size = max(10, n_docs // 400)
    n_neighbors = min(15, max(3, n_docs // 20))

    umap_model = UMAP(
        n_neighbors=min(n_neighbors, 10),
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=STOPWORDS,
        min_df=2,
    )
    representation_model = KeyBERTInspired()

    return BERTopic(
        embedding_model=_get_embedding_model(),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=False,
    )


def _extract_keywords(name: str) -> str:
    """Extract top keywords from BERTopic topic name."""
    if not isinstance(name, str):
        return ""
    parts = name.split("_")
    return ", ".join(parts[1:4]) if len(parts) > 1 else name


def fit_month(texts: list[str], embeddings: np.ndarray, min_docs: int = 30):
    """Fit BERTopic on a single month's data.

    Returns list of dicts with topic_id, keywords, count or None if too few docs.
    """
    if len(texts) < min_docs:
        return None

    model = _create_model(len(texts))

    try:
        topics, _ = model.fit_transform(texts, embeddings=embeddings)
    except Exception as e:
        logger.warning(f"  BERTopic fit failed: {e}")
        return None

    info = model.get_topic_info()
    # Filter outlier topic -1
    info = info[info["Topic"] >= 0]

    if info.empty:
        return None

    results = []
    total = sum(info["Count"])
    for _, row in info.iterrows():
        results.append({
            "topic_id": int(row["Topic"]),
            "keywords": _extract_keywords(row["Name"]),
            "count": int(row["Count"]),
            "proportion": round(row["Count"] / total, 6) if total > 0 else 0,
        })

    return results


def build_reddit(base: Path, months_filter: list[str] | None = None):
    """Fit monthly topics for Reddit data."""
    topics_dir = base / "outputs" / "topics"
    assignments_path = topics_dir / "topic_assignments.parquet"
    embeddings_path = topics_dir / "document_embeddings.npy"
    output_path = topics_dir / "monthly_topics_fitted.parquet"

    # Load text data from preprocessed parquets
    data_dir = base.parent / "data-collection" / "data" / "preprocessed"
    sub_path = data_dir / "submissions_clean.parquet"
    com_path = data_dir / "comments_clean.parquet"

    logger.info("[Reddit] Loading preprocessed data...")
    sub_df = pd.read_parquet(sub_path)
    com_df = pd.read_parquet(com_path)

    sub_df["text"] = sub_df["full_text"]
    com_df["text"] = com_df["body_clean"]

    # Build id → text mapping
    text_map = {}
    for _, row in sub_df.iterrows():
        text_map[row["id"]] = row["text"]
    for _, row in com_df.iterrows():
        text_map[row["id"]] = row["text"]

    logger.info(f"  Text map: {len(text_map):,} documents")

    # Load assignments (for id → year_month mapping + row order)
    logger.info("[Reddit] Loading assignments and embeddings...")
    assignments = pd.read_parquet(assignments_path)
    embeddings = np.load(embeddings_path)

    assert len(assignments) == len(embeddings), (
        f"Mismatch: assignments={len(assignments)}, embeddings={len(embeddings)}"
    )

    # Add text column
    assignments = assignments.copy()
    assignments["text"] = assignments["id"].map(text_map)
    # Drop rows without text
    mask = assignments["text"].notna() & (assignments["text"].str.len() > 0)
    assignments = assignments[mask]
    embeddings = embeddings[mask.values]

    logger.info(f"  {len(assignments):,} docs with text + embeddings")

    months = sorted(assignments["year_month"].unique())
    if months_filter:
        months = [m for m in months if m in months_filter]

    logger.info(f"  Fitting {len(months)} months...")

    all_results = []
    for i, month in enumerate(months):
        month_mask = assignments["year_month"] == month
        month_texts = assignments.loc[month_mask, "text"].tolist()
        month_embeddings = embeddings[month_mask.values]

        result = fit_month(month_texts, month_embeddings)
        n_topics = len(result) if result else 0
        logger.info(
            f"  [{i+1}/{len(months)}] {month}: {len(month_texts):,} docs → {n_topics} topics"
        )

        if result:
            for r in result:
                r["year_month"] = month
            all_results.extend(result)

    if not all_results:
        logger.warning("[Reddit] No monthly topics produced")
        return

    df = pd.DataFrame(all_results)
    df = df[["year_month", "topic_id", "keywords", "count", "proportion"]]
    df = df.sort_values(["year_month", "count"], ascending=[True, False]).reset_index(drop=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"[Reddit] Saved {len(df):,} rows to {output_path}")


def build_news(base: Path, months_filter: list[str] | None = None):
    """Fit monthly topics for GDELT news data."""
    news_topics_dir = base / "outputs_news" / "topics"
    assignments_path = news_topics_dir / "topic_assignments.parquet"
    embeddings_path = news_topics_dir / "document_embeddings.npy"
    output_path = news_topics_dir / "monthly_topics_fitted.parquet"

    if not assignments_path.exists() or not embeddings_path.exists():
        logger.warning("[News] Missing assignments or embeddings, skipping")
        return

    # Load GDELT text data
    gdelt_csv = base.parent.parent / "data" / "gdelt" / "gdelt_scraped_updated.csv"
    if not gdelt_csv.exists():
        logger.warning(f"[News] {gdelt_csv} not found, skipping")
        return

    logger.info("[News] Loading GDELT scraped data...")
    gdelt_df = pd.read_csv(gdelt_csv, low_memory=False)
    # Build id → text mapping (gdelt_0, gdelt_1, ...)
    # Filter successful scrapes (case-insensitive) with text >= 50 chars
    gdelt_df = gdelt_df[gdelt_df["Scrape_Status"].str.lower().str.contains("success", na=False)]
    gdelt_df["Text"] = gdelt_df["Text"].astype(str)
    gdelt_df = gdelt_df[gdelt_df["Text"].str.len() >= 50]
    gdelt_df = gdelt_df.reset_index(drop=True)
    text_map = {f"gdelt_{i}": row["Text"] for i, row in gdelt_df.iterrows()}

    logger.info(f"  Text map: {len(text_map):,} articles")

    # Load assignments + embeddings
    logger.info("[News] Loading assignments and embeddings...")
    assignments = pd.read_parquet(assignments_path)
    embeddings = np.load(embeddings_path)

    assert len(assignments) == len(embeddings), (
        f"Mismatch: assignments={len(assignments)}, embeddings={len(embeddings)}"
    )

    assignments = assignments.copy()
    assignments["text"] = assignments["id"].map(text_map)
    mask = assignments["text"].notna() & (assignments["text"].str.len() > 0)
    assignments = assignments[mask]
    embeddings = embeddings[mask.values]

    logger.info(f"  {len(assignments):,} docs with text + embeddings")

    months = sorted(assignments["year_month"].unique())
    if months_filter:
        months = [m for m in months if m in months_filter]

    logger.info(f"  Fitting {len(months)} months...")

    all_results = []
    for i, month in enumerate(months):
        month_mask = assignments["year_month"] == month
        month_texts = assignments.loc[month_mask, "text"].tolist()
        month_embeddings = embeddings[month_mask.values]

        result = fit_month(month_texts, month_embeddings)
        n_topics = len(result) if result else 0
        logger.info(
            f"  [{i+1}/{len(months)}] {month}: {len(month_texts):,} docs → {n_topics} topics"
        )

        if result:
            for r in result:
                r["year_month"] = month
            all_results.extend(result)

    if not all_results:
        logger.warning("[News] No monthly topics produced")
        return

    df = pd.DataFrame(all_results)
    df = df[["year_month", "topic_id", "keywords", "count", "proportion"]]
    df = df.sort_values(["year_month", "count"], ascending=[True, False]).reset_index(drop=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"[News] Saved {len(df):,} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit BERTopic per month")
    parser.add_argument("--reddit-only", action="store_true")
    parser.add_argument("--news-only", action="store_true")
    parser.add_argument(
        "--months",
        nargs="*",
        help="Specific months to fit (e.g. 2024-01 2024-02). Default: all.",
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    months = args.months if args.months else None

    if not args.news_only:
        build_reddit(base, months)
    if not args.reddit_only:
        build_news(base, months)


if __name__ == "__main__":
    main()
