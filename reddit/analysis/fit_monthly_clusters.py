"""
Fit HDBSCAN clustering independently for each month.

Same approach as fit_monthly_topics.py but for clustering:
each month gets its own UMAP reduction + HDBSCAN clustering
with adaptive min_cluster_size = max(10, n // 400).

Uses pre-computed embeddings (document_embeddings.npy) for speed.

Output per platform:
  clusters/monthly_clusters_fitted.parquet

Schema: year_month, cluster_id, keywords, count, proportion
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Stopwords
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _sw
    STOPWORDS = list(set(_sw.words("english")) | set(_sw.words("spanish")))
except Exception:
    STOPWORDS = "english"


def cluster_month(texts: list[str], embeddings: np.ndarray, min_docs: int = 30):
    """Cluster a single month's documents.

    Returns list of dicts with cluster_id, keywords, count, proportion
    or None if too few docs.
    """
    from hdbscan import HDBSCAN
    from umap import UMAP

    n = len(texts)
    if n < min_docs:
        return None

    # Adaptive params
    mcs = max(10, n // 400)
    n_neighbors = min(10, max(3, n // 20))

    try:
        reducer = UMAP(
            n_components=5,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)

        clusterer = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced)
    except Exception as e:
        logger.warning(f"  Clustering failed: {e}")
        return None

    cluster_ids = sorted(set(labels) - {-1})
    if not cluster_ids:
        return None

    n_clustered = (labels >= 0).sum()

    results = []
    for cid in cluster_ids:
        mask = labels == cid
        count = int(mask.sum())
        # Extract keywords for this cluster
        cluster_texts = [texts[i][:300] for i in range(n) if mask[i]]
        try:
            vec = CountVectorizer(
                max_features=500, stop_words=STOPWORDS,
                min_df=max(2, len(cluster_texts) // 10),
                ngram_range=(1, 2),
            )
            X = vec.fit_transform(cluster_texts)
            freqs = X.sum(axis=0).A1
            top_idx = freqs.argsort()[-5:][::-1]
            keywords = ", ".join([vec.get_feature_names_out()[i] for i in top_idx])
        except Exception:
            keywords = ""

        results.append({
            "cluster_id": cid,
            "keywords": keywords,
            "count": count,
            "proportion": round(count / n_clustered, 6) if n_clustered > 0 else 0,
        })

    return results


def build_platform(
    name: str,
    assignments_path: Path,
    embeddings_path: Path,
    text_loader,
    output_path: Path,
    months_filter: list[str] | None = None,
):
    """Fit monthly clusters for a platform."""
    if not assignments_path.exists() or not embeddings_path.exists():
        logger.warning(f"[{name}] Missing data, skipping")
        return

    logger.info(f"[{name}] Loading assignments and embeddings...")
    assignments = pd.read_parquet(assignments_path)
    embeddings = np.load(embeddings_path)

    assert len(assignments) == len(embeddings), (
        f"Mismatch: assigns={len(assignments)}, embeds={len(embeddings)}"
    )

    # Load texts
    text_map = text_loader()
    assignments = assignments.copy()
    assignments["text"] = assignments["id"].map(text_map)
    mask = assignments["text"].notna() & (assignments["text"].str.len() > 0)
    assignments = assignments[mask]
    embeddings = embeddings[mask.values]
    logger.info(f"  {len(assignments):,} docs with text + embeddings")

    months = sorted(assignments["year_month"].unique())
    if months_filter:
        months = [m for m in months if m in months_filter]

    logger.info(f"  Clustering {len(months)} months...")

    all_results = []
    for i, month in enumerate(months):
        month_mask = assignments["year_month"] == month
        month_texts = assignments.loc[month_mask, "text"].tolist()
        month_embeddings = embeddings[month_mask.values]

        result = cluster_month(month_texts, month_embeddings)
        n_clusters = len(result) if result else 0
        logger.info(
            f"  [{i+1}/{len(months)}] {month}: {len(month_texts):,} docs → {n_clusters} clusters"
        )

        if result:
            for r in result:
                r["year_month"] = month
            all_results.extend(result)

    if not all_results:
        logger.warning(f"[{name}] No monthly clusters produced")
        return

    df = pd.DataFrame(all_results)
    df = df[["year_month", "cluster_id", "keywords", "count", "proportion"]]
    df = df.sort_values(["year_month", "count"], ascending=[True, False]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"[{name}] Saved {len(df):,} rows to {output_path}")


def build_reddit(base: Path, months_filter=None):
    topics_dir = base / "outputs" / "topics"
    output = base / "outputs" / "clusters" / "monthly_clusters_fitted.parquet"

    data_dir = base.parent / "data-collection" / "data" / "preprocessed"
    sub_path = data_dir / "submissions_clean.parquet"
    com_path = data_dir / "comments_clean.parquet"

    def load_texts():
        sub_df = pd.read_parquet(sub_path)
        com_df = pd.read_parquet(com_path)
        text_map = {}
        for _, row in sub_df.iterrows():
            text_map[row["id"]] = row["full_text"]
        for _, row in com_df.iterrows():
            text_map[row["id"]] = row["body_clean"]
        logger.info(f"  Reddit text map: {len(text_map):,}")
        return text_map

    build_platform(
        "Reddit",
        topics_dir / "topic_assignments.parquet",
        topics_dir / "document_embeddings.npy",
        load_texts,
        output,
        months_filter,
    )


def build_news(base: Path, months_filter=None):
    topics_dir = base / "outputs_news" / "topics"
    output = base / "outputs_news" / "clusters" / "monthly_clusters_fitted.parquet"

    gdelt_csv = base.parent.parent / "data" / "gdelt" / "gdelt_scraped_updated.csv"

    def load_texts():
        gdf = pd.read_csv(gdelt_csv, low_memory=False)
        gdf = gdf[gdf["Scrape_Status"].str.lower().str.contains("success", na=False)]
        gdf["Text"] = gdf["Text"].astype(str)
        gdf = gdf[gdf["Text"].str.len() >= 50].reset_index(drop=True)
        text_map = {f"gdelt_{i}": row["Text"] for i, row in gdf.iterrows()}
        logger.info(f"  News text map: {len(text_map):,}")
        return text_map

    build_platform(
        "News",
        topics_dir / "topic_assignments.parquet",
        topics_dir / "document_embeddings.npy",
        load_texts,
        output,
        months_filter,
    )


def build_tiktok(base: Path, months_filter=None):
    topics_dir = base / "outputs_tiktok" / "topics"
    output = base / "outputs_tiktok" / "clusters" / "monthly_clusters_fitted.parquet"

    import json
    videos_dir = base.parent.parent / "tiktok" / "data-collection" / "data" / "videos"

    def load_texts():
        assigns = pd.read_parquet(topics_dir / "topic_assignments.parquet")
        # TikTok ids are tiktok_v_<id>, texts come from video files
        all_texts = []
        for f in sorted(videos_dir.glob("videos_*.json")):
            data = json.load(open(f))
            for v in data:
                all_texts.append(v.get("video_description", ""))
        text_map = {}
        for i, aid in enumerate(assigns["id"]):
            if i < len(all_texts):
                text_map[aid] = all_texts[i]
        logger.info(f"  TikTok text map: {len(text_map):,}")
        return text_map

    build_platform(
        "TikTok",
        topics_dir / "topic_assignments.parquet",
        topics_dir / "document_embeddings.npy",
        load_texts,
        output,
        months_filter,
    )


def main():
    parser = argparse.ArgumentParser(description="Fit monthly HDBSCAN clusters")
    parser.add_argument("--reddit", action="store_true")
    parser.add_argument("--news", action="store_true")
    parser.add_argument("--tiktok", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--months", nargs="*", help="Filter months (YYYY-MM)")
    args = parser.parse_args()

    base = Path(__file__).parent
    months = args.months

    if args.all or not (args.reddit or args.news or args.tiktok):
        args.reddit = args.news = args.tiktok = True

    if args.reddit:
        build_reddit(base, months)
    if args.news:
        build_news(base, months)
    if args.tiktok:
        build_tiktok(base, months)


if __name__ == "__main__":
    main()
