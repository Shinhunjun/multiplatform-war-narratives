"""
GDELT News Analysis Script
Processes gdelt_scraped.csv → sentiment + topic assignment → output CSVs
matching the same schema as the Reddit analysis pipeline.

Usage:
    python gdelt/analysis/analyze_gdelt.py
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
GDELT_CSV = PROJECT_ROOT / "data" / "gdelt" / "gdelt_scraped_updated.csv"
REDDIT_OUTPUTS = PROJECT_ROOT / "reddit" / "analysis" / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "reddit" / "analysis" / "outputs_news"

SENTIMENT_DIR = OUTPUT_DIR / "sentiment"
TOPICS_DIR = OUTPUT_DIR / "topics"


def load_and_preprocess() -> pd.DataFrame:
    """Load GDELT data and preprocess for analysis."""
    print("Loading GDELT data...")
    df = pd.read_csv(GDELT_CSV, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # Filter to successful scrapes only
    df = df[df["Scrape_Status"].isin(["Success", "Success (Archived)"])].copy()
    print(f"  After status filter: {len(df):,}")

    # Drop rows with missing or short text
    df = df.dropna(subset=["Text"])
    df = df[df["Text"].str.len() >= 50].copy()
    print(f"  After text filter (>=50 chars): {len(df):,}")

    # Convert Date (YYYYMMDD int) → year_month (YYYY-MM)
    df["Date"] = df["Date"].astype(str)
    df["year_month"] = df["Date"].str[:4] + "-" + df["Date"].str[4:6]

    # Extract domain from SourceURL as source
    def extract_domain(url):
        try:
            parsed = urlparse(str(url))
            domain = parsed.netloc
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"

    df["source"] = df["SourceURL"].apply(extract_domain)

    # Generate unique ID
    df = df.reset_index(drop=True)
    df["id"] = [f"gdelt_{i}" for i in range(len(df))]

    print(f"  Sources: {df['source'].nunique():,}")
    print(f"  Date range: {df['year_month'].min()} — {df['year_month'].max()}")
    return df


def run_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Run RoBERTa sentiment analysis on all texts."""
    print("\n--- Sentiment Analysis ---")

    # Import here so we can fail fast if dependencies missing
    sys.path.insert(0, str(PROJECT_ROOT / "reddit" / "analysis"))
    from sentiment.roberta_analyzer import analyze_sentiment_batch

    texts = df["Text"].tolist()
    print(f"Analyzing {len(texts):,} texts...")

    results = analyze_sentiment_batch(texts, batch_size=64)

    df = df.copy()
    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_confidence"] = [r["confidence"] for r in results]
    df["sentiment_score"] = [r["sentiment_score"] for r in results]

    print(f"  Positive: {(df['sentiment_label'] == 'positive').sum():,}")
    print(f"  Neutral:  {(df['sentiment_label'] == 'neutral').sum():,}")
    print(f"  Negative: {(df['sentiment_label'] == 'negative').sum():,}")
    return df


def aggregate_sentiment(df: pd.DataFrame) -> None:
    """Aggregate sentiment and save CSVs matching Reddit output format."""
    print("\nAggregating sentiment...")

    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

    def agg_group(group_df, group_cols):
        agg = group_df.groupby(group_cols).agg(
            mean_sentiment=("sentiment_score", "mean"),
            positive_count=("sentiment_label", lambda x: (x == "positive").sum()),
            negative_count=("sentiment_label", lambda x: (x == "negative").sum()),
            total_count=("sentiment_label", "count"),
        ).reset_index()
        agg["positive_ratio"] = agg["positive_count"] / agg["total_count"]
        agg["negative_ratio"] = agg["negative_count"] / agg["total_count"]
        return agg[group_cols + ["mean_sentiment", "positive_ratio", "negative_ratio", "total_count"]]

    # By month
    by_month = agg_group(df, ["year_month"])
    by_month.to_csv(SENTIMENT_DIR / "sentiment_by_month.csv", index=False)
    print(f"  sentiment_by_month.csv: {len(by_month)} rows")

    # By source (analogous to by_subreddit)
    by_source = agg_group(df, ["source"])
    by_source.to_csv(SENTIMENT_DIR / "sentiment_by_source.csv", index=False)
    print(f"  sentiment_by_source.csv: {len(by_source)} rows")

    # By source + month (analogous to by_subreddit_month)
    by_source_month = agg_group(df, ["source", "year_month"])
    by_source_month.to_csv(SENTIMENT_DIR / "sentiment_by_source_month.csv", index=False)
    print(f"  sentiment_by_source_month.csv: {len(by_source_month)} rows")


def assign_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Assign topics using cosine similarity to existing Reddit topic centroids."""
    print("\n--- Topic Assignment ---")

    from sentence_transformers import SentenceTransformer

    # Load existing topic data
    embeddings_path = REDDIT_OUTPUTS / "topics" / "document_embeddings.npy"
    assignments_path = REDDIT_OUTPUTS / "topics" / "topic_assignments.parquet"

    if not embeddings_path.exists() or not assignments_path.exists():
        print("WARNING: Reddit topic embeddings not found. Skipping topic assignment.")
        df["topic_id"] = -1
        return df

    print("Loading Reddit topic centroids...")
    reddit_embeddings = np.load(embeddings_path)
    reddit_assignments = pd.read_parquet(assignments_path)

    # Compute centroids per topic (excluding -1 outlier)
    topic_ids = reddit_assignments["topic_id"].values
    unique_topics = sorted(set(topic_ids[topic_ids >= 0]))
    centroids = {}
    for t in unique_topics:
        mask = topic_ids == t
        centroids[t] = reddit_embeddings[mask].mean(axis=0)

    centroid_matrix = np.array([centroids[t] for t in unique_topics])
    # Normalize centroids
    centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    centroid_matrix_normed = centroid_matrix / (centroid_norms + 1e-10)

    print(f"  {len(unique_topics)} topic centroids loaded")

    # Embed GDELT texts
    print("Embedding GDELT texts...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = df["Text"].tolist()

    # Truncate texts for embedding
    texts_truncated = [t[:512] for t in texts]

    batch_size = 256
    all_embeddings = model.encode(
        texts_truncated,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Cosine similarity → assign nearest topic
    print("Assigning topics via cosine similarity...")
    THRESHOLD = 0.25
    assigned_topics = []

    chunk_size = 5000
    for i in tqdm(range(0, len(all_embeddings), chunk_size), desc="Topic assignment"):
        chunk = all_embeddings[i:i + chunk_size]
        # cosine sim (already normalized)
        sims = chunk @ centroid_matrix_normed.T
        best_idx = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(chunk)), best_idx]

        for j in range(len(chunk)):
            if best_sim[j] >= THRESHOLD:
                assigned_topics.append(unique_topics[best_idx[j]])
            else:
                assigned_topics.append(-1)

    df = df.copy()
    df["topic_id"] = assigned_topics

    assigned_count = sum(1 for t in assigned_topics if t >= 0)
    print(f"  Assigned: {assigned_count:,} ({assigned_count / len(df) * 100:.1f}%)")
    print(f"  Outlier (-1): {len(df) - assigned_count:,}")

    return df


def aggregate_topics(df: pd.DataFrame) -> None:
    """Aggregate topics and save CSVs."""
    print("\nAggregating topics...")

    TOPICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load Reddit topic_info for names
    topic_info_path = REDDIT_OUTPUTS / "topics" / "topic_info.csv"
    if topic_info_path.exists():
        reddit_topic_info = pd.read_csv(topic_info_path)
        # Copy topic_info.csv as-is (same topics, different counts for news)
        topic_counts = df[df["topic_id"] >= 0].groupby("topic_id").size().reset_index(name="Count")
        merged = reddit_topic_info[["Topic", "Name", "Representation"]].merge(
            topic_counts, left_on="Topic", right_on="topic_id", how="left"
        ).drop(columns=["topic_id"])
        merged["Count"] = merged["Count"].fillna(0).astype(int)
        merged = merged[merged["Topic"] >= 0]
        merged.to_csv(TOPICS_DIR / "topic_info.csv", index=False)
        print(f"  topic_info.csv: {len(merged)} topics")

    # Topics over time
    topics_time = df[df["topic_id"] >= 0].groupby(["topic_id", "year_month"]).size().reset_index(name="Frequency")
    topics_time.columns = ["Topic", "Timestamp", "Frequency"]
    topics_time.to_csv(TOPICS_DIR / "topics_over_time.csv", index=False)
    print(f"  topics_over_time.csv: {len(topics_time)} rows")

    # Topics by source
    topics_source = df.groupby(["source", "topic_id"]).size().reset_index(name="count")
    topics_source.to_csv(TOPICS_DIR / "topics_by_source.csv", index=False)
    print(f"  topics_by_source.csv: {len(topics_source)} rows")


def save_overview(df: pd.DataFrame) -> None:
    """Save overview.json with summary statistics."""
    overview = {
        "total_documents": len(df),
        "sources": int(df["source"].nunique()),
        "date_range": {
            "start": df["year_month"].min(),
            "end": df["year_month"].max(),
        },
        "avg_sentiment": round(float(df["sentiment_score"].mean()), 4),
    }
    overview_path = OUTPUT_DIR / "overview.json"
    with open(overview_path, "w") as f:
        json.dump(overview, f, indent=2)
    print(f"\noverview.json saved: {overview}")


def main():
    # Ensure output dirs exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and preprocess
    df = load_and_preprocess()

    # Step 2: Sentiment analysis
    df = run_sentiment_analysis(df)
    aggregate_sentiment(df)

    # Step 3: Topic assignment
    df = assign_topics(df)
    aggregate_topics(df)

    # Step 4: Save overview
    save_overview(df)

    print("\n=== GDELT Analysis Complete ===")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
