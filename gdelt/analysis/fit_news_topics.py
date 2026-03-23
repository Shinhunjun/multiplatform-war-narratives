"""
Fit an independent BERTopic model on GDELT news data.

Unlike the previous approach (cosine similarity to Reddit centroids),
this trains a fresh BERTopic model directly on news texts for more
granular and news-specific topics.

Usage:
    python gdelt/analysis/fit_news_topics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
GDELT_CSV = PROJECT_ROOT / "data" / "gdelt" / "gdelt_scraped_updated.csv"
OUTPUT_DIR = PROJECT_ROOT / "reddit" / "analysis" / "outputs_news" / "topics"

# Add reddit analysis to path for stopwords
sys.path.insert(0, str(PROJECT_ROOT / "reddit" / "analysis"))


def load_data() -> pd.DataFrame:
    """Load and preprocess GDELT data."""
    from urllib.parse import urlparse

    print("Loading GDELT data...")
    df = pd.read_csv(GDELT_CSV, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    df = df[df["Scrape_Status"].isin(["Success", "Success (Archived)"])].copy()
    df = df.dropna(subset=["Text"])
    df = df[df["Text"].str.len() >= 50].copy()
    print(f"  After filter: {len(df):,}")

    df["Date"] = df["Date"].astype(str)
    df["year_month"] = df["Date"].str[:4] + "-" + df["Date"].str[4:6]

    def extract_domain(url):
        try:
            parsed = urlparse(str(url))
            domain = parsed.netloc
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"

    df["source"] = df["SourceURL"].apply(extract_domain)
    df = df.reset_index(drop=True)
    df["id"] = [f"gdelt_{i}" for i in range(len(df))]

    print(f"  Date range: {df['year_month'].min()} — {df['year_month'].max()}")
    return df


def fit_bertopic(texts: list[str]):
    """Fit BERTopic with news-appropriate parameters."""
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    en_stops = set(stopwords.words("english"))
    es_stops = set(stopwords.words("spanish"))
    combined_stops = list(en_stops | es_stops)

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Adaptive params from hyperparameter experiment: news optimal mcs=10, nn=10
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=combined_stops,
        min_df=3,
    )

    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=None,  # auto — let HDBSCAN decide
        top_n_words=10,
        verbose=True,
    )

    # Truncate long texts for embedding (keep first 512 chars)
    texts_trunc = [t[:512] for t in texts]

    print(f"Fitting BERTopic on {len(texts_trunc):,} documents...")
    topics, probs = topic_model.fit_transform(texts_trunc)

    embeddings = topic_model._extract_embeddings(texts_trunc)

    return topic_model, topics, probs, embeddings


def save_outputs(df: pd.DataFrame, topic_model, topics, probs, embeddings):
    """Save all topic outputs matching the webapp schema."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # topic_info.csv
    info = topic_model.get_topic_info()
    info.to_csv(OUTPUT_DIR / "topic_info.csv", index=False)
    n_topics = len(info[info["Topic"] >= 0])
    print(f"  topic_info.csv: {n_topics} topics (excl. outlier)")

    # topic_assignments (for monthly precompute later)
    df = df.copy()
    df["topic_id"] = topics
    df["topic_prob"] = probs

    topic_labels = dict(zip(info["Topic"], info["Name"]))
    df["topic_label"] = df["topic_id"].map(topic_labels)

    assign_cols = ["id", "source", "year_month", "topic_id", "topic_prob", "topic_label"]
    df[assign_cols].to_parquet(OUTPUT_DIR / "topic_assignments.parquet", index=False)
    print(f"  topic_assignments.parquet: {len(df):,} rows")

    # topics_over_time.csv
    valid = df[df["topic_id"] >= 0]
    tot = valid.groupby(["topic_id", "year_month"]).size().reset_index(name="Frequency")
    tot.columns = ["Topic", "Timestamp", "Frequency"]
    tot.to_csv(OUTPUT_DIR / "topics_over_time.csv", index=False)
    print(f"  topics_over_time.csv: {len(tot)} rows")

    # topics_by_source.csv
    tbs = df.groupby(["source", "topic_id"]).size().reset_index(name="count")
    tbs.to_csv(OUTPUT_DIR / "topics_by_source.csv", index=False)
    print(f"  topics_by_source.csv: {len(tbs)} rows")

    # document_embeddings.npy (for future use)
    np.save(OUTPUT_DIR / "document_embeddings.npy", embeddings)
    print(f"  document_embeddings.npy: {embeddings.shape}")

    # Save model
    model_path = OUTPUT_DIR / "bertopic_model"
    topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
    print(f"  Model saved to {model_path}")

    return df


def build_monthly(df: pd.DataFrame, topic_model):
    """Build topics_monthly.parquet for the slider UI."""
    info = topic_model.get_topic_info()
    name_map = info[info["Topic"] >= 0].set_index("Topic")["Name"].to_dict()

    valid = df[df["topic_id"] >= 0]
    monthly = (
        valid.groupby(["year_month", "topic_id"])
        .size()
        .reset_index(name="count")
    )

    month_totals = monthly.groupby("year_month")["count"].transform("sum")
    monthly["proportion"] = (monthly["count"] / month_totals).round(6)
    monthly["name"] = monthly["topic_id"].map(name_map).fillna("")

    def extract_keywords(name: str) -> str:
        if not name:
            return ""
        parts = name.split("_")
        return ", ".join(parts[1:4]) if len(parts) > 1 else name

    monthly["keywords"] = monthly["name"].apply(extract_keywords)
    monthly = monthly.sort_values(
        ["year_month", "count"], ascending=[True, False]
    ).reset_index(drop=True)

    monthly.to_parquet(OUTPUT_DIR / "topics_monthly.parquet", index=False)
    print(f"  topics_monthly.parquet: {len(monthly):,} rows, {monthly['year_month'].nunique()} months")


def main():
    df = load_data()
    texts = df["Text"].tolist()

    topic_model, topics, probs, embeddings = fit_bertopic(texts)

    df = save_outputs(df, topic_model, topics, probs, embeddings)
    build_monthly(df, topic_model)

    # Summary
    info = topic_model.get_topic_info()
    n_topics = len(info[info["Topic"] >= 0])
    assigned = sum(1 for t in topics if t >= 0)
    print(f"\n=== News Topic Modeling Complete ===")
    print(f"  Topics found: {n_topics}")
    print(f"  Documents assigned: {assigned:,} / {len(df):,} ({assigned/len(df)*100:.1f}%)")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
