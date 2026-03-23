"""
TikTok Analysis Pipeline — produces outputs_tiktok/ compatible with the webapp.

Generates the same output structure as Reddit/GDELT:
  outputs_tiktok/
  ├── overview.json
  ├── sentiment/
  │   ├── sentiment_by_month.csv
  │   ├── sentiment_by_source.csv
  │   └── sentiment_by_source_month.csv
  ├── topics/
  │   ├── topic_info.csv
  │   ├── topics_over_time.csv
  │   ├── monthly_topics_fitted.parquet
  │   ├── topic_assignments.parquet
  │   └── document_embeddings.npy
  └── tiktok_specific/
      ├── hashtag_trends.parquet
      ├── engagement_metrics.parquet
      └── region_distribution.parquet

Usage:
    # From reddit/analysis/ venv:
    python ../../tiktok/analysis/run_analysis.py
    python ../../tiktok/analysis/run_analysis.py --skip-sentiment
    python ../../tiktok/analysis/run_analysis.py --skip-topics
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # capstone/
TIKTOK_DATA_DIR = PROJECT_ROOT / "tiktok" / "data-collection" / "data"
OUTPUT_DIR = PROJECT_ROOT / "reddit" / "analysis" / "outputs_tiktok"

# Add reddit analysis to path for reusing sentiment/topic modules
REDDIT_ANALYSIS_DIR = PROJECT_ROOT / "reddit" / "analysis"
sys.path.insert(0, str(REDDIT_ANALYSIS_DIR))


# ============================================================================
# STEP 1: LOAD & PREPROCESS
# ============================================================================

def load_tiktok_data() -> pd.DataFrame:
    """Load videos + comments into a unified DataFrame."""
    videos_dir = TIKTOK_DATA_DIR / "videos"
    # Try merged comments first, then individual playwright file
    comments_file = TIKTOK_DATA_DIR / "comments" / "comments_all_merged.json"
    if not comments_file.exists():
        comments_file = TIKTOK_DATA_DIR / "comments" / "comments_playwright.json"

    # Load all video files
    all_videos = []
    for f in sorted(videos_dir.glob("videos_*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for v in data:
            v["_source_file"] = f.name
        all_videos.extend(data)

    logger.info(f"Loaded {len(all_videos):,} video records")

    # Build video DataFrame
    vdf = pd.DataFrame(all_videos)
    vdf["doc_id"] = "tiktok_v_" + vdf["id"].astype(str)
    vdf["type"] = "video"
    vdf["text"] = vdf["video_description"].fillna("")
    # Add voice_to_text if available
    if "voice_to_text" in vdf.columns:
        vtt = vdf["voice_to_text"].fillna("")
        mask = vtt.str.len() > 0
        vdf.loc[mask, "text"] = vdf.loc[mask, "text"] + " " + vtt[mask]
    vdf["text"] = vdf["text"].str.strip()
    vdf["source"] = vdf["username"].fillna("unknown")
    vdf["created_utc"] = pd.to_numeric(vdf["create_time"], errors="coerce")
    vdf["created_datetime"] = pd.to_datetime(vdf["created_utc"], unit="s", errors="coerce")
    vdf["year_month"] = vdf["created_datetime"].dt.strftime("%Y-%m")
    vdf["hashtag_names"] = vdf["hashtag_names"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Load comments
    comments = []
    if comments_file.exists():
        with open(comments_file, "r", encoding="utf-8") as fh:
            comments = json.load(fh)
    logger.info(f"Loaded {len(comments):,} comment records")

    cdf = pd.DataFrame(comments) if comments else pd.DataFrame()
    if not cdf.empty:
        cdf["doc_id"] = "tiktok_c_" + cdf["comment_id"].astype(str)
        cdf["type"] = "comment"
        cdf["text"] = cdf["text"].fillna("")
        cdf["source"] = cdf.get("video_username", pd.Series(dtype=str)).fillna("unknown")
        cdf["created_utc"] = pd.to_numeric(cdf["create_time"], errors="coerce")
        cdf["created_datetime"] = pd.to_datetime(cdf["created_utc"], unit="s", errors="coerce")
        cdf["year_month"] = cdf["created_datetime"].dt.strftime("%Y-%m")
        cdf["hashtag_names"] = [[] for _ in range(len(cdf))]

    # Combine
    keep_cols = ["doc_id", "type", "text", "source", "created_utc",
                 "created_datetime", "year_month", "hashtag_names"]

    vdf_clean = vdf[[c for c in keep_cols if c in vdf.columns]].copy()
    # Add video-specific columns for later
    vdf_clean["region_code"] = vdf["region_code"].fillna("unknown") if "region_code" in vdf.columns else "unknown"
    vdf_clean["view_count"] = pd.to_numeric(vdf.get("view_count", 0), errors="coerce").fillna(0).astype(int)
    vdf_clean["like_count"] = pd.to_numeric(vdf.get("like_count", 0), errors="coerce").fillna(0).astype(int)
    vdf_clean["share_count"] = pd.to_numeric(vdf.get("share_count", 0), errors="coerce").fillna(0).astype(int)
    vdf_clean["comment_count"] = pd.to_numeric(vdf.get("comment_count", 0), errors="coerce").fillna(0).astype(int)
    vdf_clean["video_duration"] = pd.to_numeric(vdf.get("video_duration", 0), errors="coerce").fillna(0)

    if not cdf.empty:
        cdf_clean = cdf[[c for c in keep_cols if c in cdf.columns]].copy()
        for col in ["region_code", "view_count", "like_count", "share_count", "comment_count", "video_duration"]:
            if col not in cdf_clean.columns:
                cdf_clean[col] = "unknown" if col == "region_code" else 0
        df = pd.concat([vdf_clean, cdf_clean], ignore_index=True)
    else:
        df = vdf_clean

    # Filter: need text
    df = df[df["text"].str.len() >= 3].copy()
    df = df.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)

    logger.info(f"Unified dataset: {len(df):,} documents ({(df['type']=='video').sum():,} videos, {(df['type']=='comment').sum():,} comments)")
    logger.info(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")

    return df


# ============================================================================
# STEP 2: SENTIMENT ANALYSIS
# ============================================================================

def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Run sentiment analysis and save outputs."""
    from sentiment.roberta_analyzer import analyze_dataframe, aggregate_sentiment

    out_dir = OUTPUT_DIR / "sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running sentiment analysis...")
    df = analyze_dataframe(df, text_column="text", batch_size=64)

    # Save full results
    df.to_parquet(out_dir / "sentiment_full.parquet", index=False)

    # Aggregate by month
    by_month = aggregate_sentiment(df, group_by=["year_month"])
    by_month.to_csv(out_dir / "sentiment_by_month.csv", index=False)
    logger.info(f"  sentiment_by_month.csv: {len(by_month)} rows")

    # Aggregate by source (username)
    by_source = aggregate_sentiment(df, group_by=["source"])
    by_source.to_csv(out_dir / "sentiment_by_source.csv", index=False)
    logger.info(f"  sentiment_by_source.csv: {len(by_source)} rows")

    # Aggregate by source + month
    by_source_month = aggregate_sentiment(df, group_by=["source", "year_month"])
    by_source_month.to_csv(out_dir / "sentiment_by_source_month.csv", index=False)
    logger.info(f"  sentiment_by_source_month.csv: {len(by_source_month)} rows")

    return df


# ============================================================================
# STEP 3: TOPIC MODELING
# ============================================================================

def run_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Run BERTopic and save outputs compatible with webapp."""
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as _sw
        stopwords = list(set(_sw.words("english")) | set(_sw.words("spanish")))
    except Exception:
        stopwords = "english"

    out_dir = OUTPUT_DIR / "topics"
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = df["text"].tolist()
    n_docs = len(texts)
    logger.info(f"Topic modeling on {n_docs:,} documents...")

    # Compute embeddings
    logger.info("  Computing embeddings...")
    embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=64)
    np.save(out_dir / "document_embeddings.npy", embeddings)
    logger.info(f"  Embeddings shape: {embeddings.shape}")

    # Adaptive params from hyperparameter experiment
    min_topic_size = max(10, n_docs // 400)
    n_neighbors = min(10, max(5, n_docs // 50))

    umap_model = UMAP(
        n_neighbors=n_neighbors, n_components=5,
        min_dist=0.0, metric="cosine", random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size, min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2), stop_words=stopwords, min_df=2,
    )
    representation_model = KeyBERTInspired()

    model = BERTopic(
        embedding_model=embed_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=True,
    )

    logger.info("  Fitting BERTopic...")
    topics, probs = model.fit_transform(texts, embeddings=embeddings)

    # Save model
    model.save(str(out_dir / "bertopic_model"), serialization="safetensors", save_ctfidf=True)

    # topic_info.csv
    topic_info = model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] >= 0]
    topic_info.to_csv(out_dir / "topic_info.csv", index=False)
    logger.info(f"  topic_info.csv: {len(topic_info)} topics")

    # topic_assignments.parquet
    df = df.copy()
    df["topic_id"] = topics
    df["topic_prob"] = [p.max() if hasattr(p, 'max') else 0.0 for p in (probs if probs is not None else [0.0]*len(topics))]

    # Build topic label map
    label_map = {}
    for _, row in topic_info.iterrows():
        parts = str(row["Name"]).split("_")
        label = ", ".join(parts[1:4]) if len(parts) > 1 else row["Name"]
        label_map[row["Topic"]] = label
    df["topic_label"] = df["topic_id"].map(label_map).fillna("noise")

    assignments = df[["doc_id", "type", "source", "year_month", "topic_id", "topic_prob", "topic_label"]].copy()
    assignments = assignments.rename(columns={"doc_id": "id"})
    assignments.to_parquet(out_dir / "topic_assignments.parquet", index=False)

    # topics_over_time.csv
    try:
        tot = model.topics_over_time(texts, df["created_datetime"].tolist(), nr_bins=20)
        tot.to_csv(out_dir / "topics_over_time.csv", index=False)
        logger.info(f"  topics_over_time.csv: {len(tot)} rows")
    except Exception as e:
        logger.warning(f"  topics_over_time failed: {e}")

    # monthly_topics_fitted.parquet (independent per-month models)
    logger.info("  Fitting per-month BERTopic models...")
    from fit_monthly_topics import fit_month

    months = sorted(df["year_month"].dropna().unique())
    all_monthly = []
    for month in months:
        mask = df["year_month"] == month
        month_texts = df.loc[mask, "text"].tolist()
        month_embeddings = embeddings[mask.values]
        result = fit_month(month_texts, month_embeddings, min_docs=20)
        if result:
            for r in result:
                r["year_month"] = month
            all_monthly.extend(result)
            logger.info(f"    {month}: {len(month_texts)} docs → {len(result)} topics")

    if all_monthly:
        mdf = pd.DataFrame(all_monthly)
        mdf = mdf[["year_month", "topic_id", "keywords", "count", "proportion"]]
        mdf = mdf.sort_values(["year_month", "count"], ascending=[True, False]).reset_index(drop=True)
        mdf.to_parquet(out_dir / "monthly_topics_fitted.parquet", index=False)
        logger.info(f"  monthly_topics_fitted.parquet: {len(mdf)} rows")

    return df


# ============================================================================
# STEP 4: TIKTOK-SPECIFIC ANALYSIS
# ============================================================================

def run_tiktok_specific(df: pd.DataFrame):
    """Generate TikTok-specific analysis outputs."""
    out_dir = OUTPUT_DIR / "tiktok_specific"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = df[df["type"] == "video"].copy()

    # --- Hashtag Trends ---
    logger.info("Computing hashtag trends...")
    rows = []
    for _, row in videos.iterrows():
        ym = row["year_month"]
        sentiment = row.get("sentiment_score", 0.0)
        for ht in row.get("hashtag_names", []):
            if isinstance(ht, str) and ht.strip():
                rows.append({
                    "year_month": ym,
                    "hashtag": ht.lower().strip(),
                    "sentiment_score": sentiment,
                })

    if rows:
        ht_df = pd.DataFrame(rows)
        ht_agg = ht_df.groupby(["year_month", "hashtag"]).agg(
            count=("hashtag", "count"),
            mean_sentiment=("sentiment_score", "mean"),
        ).reset_index()
        ht_agg = ht_agg.sort_values(["year_month", "count"], ascending=[True, False])
        ht_agg.to_parquet(out_dir / "hashtag_trends.parquet", index=False)
        logger.info(f"  hashtag_trends.parquet: {len(ht_agg)} rows, {ht_agg['hashtag'].nunique()} unique hashtags")

    # --- Engagement Metrics ---
    logger.info("Computing engagement metrics...")
    eng = videos.groupby("year_month").agg(
        video_count=("doc_id", "count"),
        total_views=("view_count", "sum"),
        total_likes=("like_count", "sum"),
        total_shares=("share_count", "sum"),
        total_comments=("comment_count", "sum"),
        avg_views=("view_count", "mean"),
        avg_likes=("like_count", "mean"),
        avg_duration=("video_duration", "mean"),
    ).reset_index()
    eng.to_parquet(out_dir / "engagement_metrics.parquet", index=False)
    logger.info(f"  engagement_metrics.parquet: {len(eng)} rows")

    # --- Region Distribution ---
    logger.info("Computing region distribution...")
    region = videos.groupby(["year_month", "region_code"]).agg(
        count=("doc_id", "count"),
        mean_sentiment=("sentiment_score", "mean") if "sentiment_score" in videos.columns else ("doc_id", "count"),
    ).reset_index()
    region.to_parquet(out_dir / "region_distribution.parquet", index=False)
    logger.info(f"  region_distribution.parquet: {len(region)} rows")


# ============================================================================
# STEP 5: OVERVIEW JSON
# ============================================================================

def generate_overview(df: pd.DataFrame):
    """Generate overview.json for the webapp."""
    months = sorted(df["year_month"].dropna().unique())
    sources = sorted(df["source"].unique())
    n_topics = 0
    ti_path = OUTPUT_DIR / "topics" / "topic_info.csv"
    if ti_path.exists():
        n_topics = len(pd.read_csv(ti_path))

    overview = {
        "platform": "tiktok",
        "total_documents": len(df),
        "total_videos": int((df["type"] == "video").sum()),
        "total_comments": int((df["type"] == "comment").sum()),
        "num_sources": len(sources),
        "num_topics": n_topics,
        "avg_sentiment": float(df["sentiment_score"].mean()) if "sentiment_score" in df.columns else 0.0,
        "date_range": {"start": months[0] if months else "", "end": months[-1] if months else ""},
        "all_months": months,
    }

    with open(OUTPUT_DIR / "overview.json", "w") as f:
        json.dump(overview, f, indent=2, default=str)
    logger.info(f"overview.json saved")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TikTok Analysis Pipeline")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--skip-topics", action="store_true")
    args = parser.parse_args()

    # Ensure output dirs
    for subdir in ["sentiment", "topics", "tiktok_specific"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    logger.info("=" * 70)
    logger.info("STEP 1: Loading TikTok data")
    logger.info("=" * 70)
    df = load_tiktok_data()

    # Step 2: Sentiment
    if not args.skip_sentiment:
        logger.info("=" * 70)
        logger.info("STEP 2: Sentiment Analysis")
        logger.info("=" * 70)
        df = run_sentiment(df)
    else:
        # Try loading existing sentiment
        sent_path = OUTPUT_DIR / "sentiment" / "sentiment_full.parquet"
        if sent_path.exists():
            logger.info("Loading existing sentiment results...")
            sent_df = pd.read_parquet(sent_path)
            for col in ["sentiment_label", "sentiment_confidence", "sentiment_score"]:
                if col in sent_df.columns:
                    df[col] = sent_df[col].values[:len(df)]

    # Step 3: Topics
    if not args.skip_topics:
        logger.info("=" * 70)
        logger.info("STEP 3: Topic Modeling")
        logger.info("=" * 70)
        df = run_topics(df)

    # Step 4: TikTok-specific
    logger.info("=" * 70)
    logger.info("STEP 4: TikTok-Specific Analysis")
    logger.info("=" * 70)
    run_tiktok_specific(df)

    # Step 5: Overview
    logger.info("=" * 70)
    logger.info("STEP 5: Overview")
    logger.info("=" * 70)
    generate_overview(df)

    logger.info("=" * 70)
    logger.info("TIKTOK ANALYSIS COMPLETE")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
