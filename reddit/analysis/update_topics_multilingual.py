"""
Recompute BERTopic topic representations with multilingual (EN+ES) stopwords.

Instead of loading the saved model (which has Python version pickle issues),
we reconstruct the topic model from saved embeddings + assignments, then
apply a new CountVectorizer with combined EN+ES stopwords.

Usage:
    python reddit/analysis/update_topics_multilingual.py
"""

import sys
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Paths
OUTPUTS_DIR = Path(__file__).parent / "outputs"
TOPICS_DIR = OUTPUTS_DIR / "topics"
SENTIMENT_DIR = OUTPUTS_DIR / "sentiment"

# Combined EN + ES stopwords
EN_STOPS = set(stopwords.words("english"))
ES_STOPS = set(stopwords.words("spanish"))
COMBINED_STOPS = list(EN_STOPS | ES_STOPS)
print(f"Combined stopwords: {len(COMBINED_STOPS)} (EN: {len(EN_STOPS)}, ES: {len(ES_STOPS)})")


def main():
    print("=" * 60)
    print("RECOMPUTING TOPICS (Multilingual EN+ES Stopwords)")
    print("=" * 60)

    # 1. Load documents with text
    print("\nLoading documents...")
    sentiment_path = SENTIMENT_DIR / "sentiment_full.parquet"
    if not sentiment_path.exists():
        print(f"ERROR: {sentiment_path} not found")
        sys.exit(1)

    full_df = pd.read_parquet(sentiment_path)
    docs = full_df["text"].fillna("").tolist()
    print(f"  {len(docs):,} documents loaded")

    # 2. Load existing embeddings
    print("Loading embeddings...")
    embeddings_path = TOPICS_DIR / "document_embeddings.npy"
    embeddings = np.load(embeddings_path)
    print(f"  Embeddings shape: {embeddings.shape}")

    # 3. Load existing topic assignments to preserve clustering
    print("Loading existing topic assignments...")
    assignments = pd.read_parquet(TOPICS_DIR / "topic_assignments.parquet")
    old_topics = assignments["topic_id"].tolist()
    n_topics_found = len(set(old_topics)) - (1 if -1 in old_topics else 0)
    print(f"  {n_topics_found} topics (excluding outliers)")

    # 4. Show old Topic 0 info
    old_info = pd.read_csv(TOPICS_DIR / "topic_info.csv")
    old_t0 = old_info[old_info["Topic"] == 0]
    if len(old_t0) > 0:
        print(f"\n  OLD Topic 0: {old_t0['Name'].values[0]}")

    # 5. Create new BERTopic with multilingual vectorizer
    print("\nCreating BERTopic with multilingual stopwords...")

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=COMBINED_STOPS,
        min_df=2,
    )

    # Minimal sub-models (won't be used for fitting, just for structure)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=50, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    representation_model = KeyBERTInspired()

    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=None,
        top_n_words=10,
        verbose=True,
    )

    # 6. Fit using precomputed embeddings (UMAP + HDBSCAN will run but we override topics)
    print("\nFitting BERTopic with precomputed embeddings...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # 7. Compare results
    new_info = topic_model.get_topic_info()
    print(f"\n  NEW Topic 0: {new_info[new_info['Topic'] == 0]['Name'].values[0] if len(new_info[new_info['Topic'] == 0]) > 0 else 'N/A'}")

    print(f"\n--- All Topics (new) ---")
    for _, row in new_info[new_info["Topic"] >= 0].sort_values("Count", ascending=False).iterrows():
        print(f"  Topic {row['Topic']:>2}: {row['Count']:>6}  {row['Name'][:70]}")

    # 8. Save updated outputs
    print("\nSaving topic_info.csv...")
    new_info.to_csv(TOPICS_DIR / "topic_info.csv", index=False)

    print("Saving updated topic_assignments.parquet...")
    assignments_new = full_df[["id", "type", "subreddit", "year_month"]].copy()
    assignments_new["topic_id"] = topics
    assignments_new["topic_prob"] = probs
    topic_labels = dict(zip(new_info["Topic"], new_info["Name"]))
    assignments_new["topic_label"] = assignments_new["topic_id"].map(topic_labels)
    assignments_new.to_parquet(TOPICS_DIR / "topic_assignments.parquet", index=False)

    # 9. Regenerate topics_over_time
    print("Regenerating topics_over_time...")
    if "created_datetime" in full_df.columns:
        timestamps = full_df["created_datetime"].tolist()
    else:
        timestamps = pd.to_datetime(full_df["year_month"] + "-01").tolist()

    tot = topic_model.topics_over_time(docs, timestamps, nr_bins=30)
    tot.to_csv(TOPICS_DIR / "topics_over_time.csv", index=False)
    print(f"  {len(tot)} rows")

    # 10. Regenerate topics_by_subreddit
    print("Regenerating topics_by_subreddit...")
    full_df["topic_id"] = topics
    topic_counts = full_df.groupby(["subreddit", "topic_id"]).size().reset_index(name="count")
    group_totals = topic_counts.groupby("subreddit")["count"].transform("sum")
    topic_counts["proportion"] = topic_counts["count"] / group_totals
    topic_counts.to_csv(TOPICS_DIR / "topics_by_subreddit.csv", index=False)
    print(f"  {len(topic_counts)} rows")

    # 11. Save model
    print("Saving BERTopic model...")
    topic_model.save(
        str(TOPICS_DIR / "bertopic_model"),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    print("\n" + "=" * 60)
    print("DONE — Topics recomputed with EN+ES stopwords")
    print("=" * 60)


if __name__ == "__main__":
    main()
