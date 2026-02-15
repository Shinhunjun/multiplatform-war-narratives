"""
Incremental analysis runner.
Runs sentiment, topic modeling, and clustering on new data,
then merges results with existing analysis outputs.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import PipelineConfig

logger = logging.getLogger(__name__)


class IncrementalAnalyzer:
    """
    Runs analysis on new data and merges with existing outputs.

    Strategy:
    - Sentiment: Run on new data, append to existing aggregations
    - Topics: Use existing BERTopic model to transform new documents
    - Clusters: Assign new documents to nearest existing clusters
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _load_existing_sentiment(self) -> dict:
        """Load existing sentiment output CSVs."""
        sentiment_dir = self.config.outputs_dir / "sentiment"
        return {
            "by_month": pd.read_csv(sentiment_dir / "sentiment_by_month.csv")
            if (sentiment_dir / "sentiment_by_month.csv").exists()
            else pd.DataFrame(),
            "by_subreddit": pd.read_csv(sentiment_dir / "sentiment_by_subreddit.csv")
            if (sentiment_dir / "sentiment_by_subreddit.csv").exists()
            else pd.DataFrame(),
            "by_subreddit_month": pd.read_csv(sentiment_dir / "sentiment_by_subreddit_month.csv")
            if (sentiment_dir / "sentiment_by_subreddit_month.csv").exists()
            else pd.DataFrame(),
        }

    def run_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run sentiment analysis on new data."""
        from transformers import pipeline as hf_pipeline

        logger.info(f"Running sentiment analysis on {len(df)} documents...")

        sentiment_pipe = hf_pipeline(
            "sentiment-analysis",
            model=self.config.sentiment_model,
            device="mps",  # Apple Silicon; change to "cuda" or "cpu" as needed
            truncation=True,
            max_length=512,
        )

        texts = df["text"].tolist()
        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Truncate long texts
            batch = [t[:512] for t in batch]
            try:
                preds = sentiment_pipe(batch)
                results.extend(preds)
            except Exception as e:
                logger.error(f"Sentiment batch {i} failed: {e}")
                results.extend([{"label": "neutral", "score": 0.5}] * len(batch))

        # Map labels to scores
        label_map = {"positive": 1, "negative": -1, "neutral": 0}
        df["sentiment_label"] = [r["label"] for r in results]
        df["sentiment_confidence"] = [r["score"] for r in results]
        df["sentiment_score"] = [
            label_map.get(r["label"], 0) * r["score"] for r in results
        ]

        logger.info(f"Sentiment done. Mean score: {df['sentiment_score'].mean():.3f}")
        return df

    def update_sentiment_outputs(self, new_df: pd.DataFrame) -> None:
        """Merge new sentiment results with existing aggregations."""
        sentiment_dir = self.config.outputs_dir / "sentiment"
        sentiment_dir.mkdir(parents=True, exist_ok=True)

        existing = self._load_existing_sentiment()

        # Aggregate new data
        if "year_month" in new_df.columns:
            new_by_month = (
                new_df.groupby("year_month")
                .agg(
                    mean_sentiment=("sentiment_score", "mean"),
                    positive_ratio=("sentiment_label", lambda x: (x == "positive").mean()),
                    negative_ratio=("sentiment_label", lambda x: (x == "negative").mean()),
                    total_count=("id", "count"),
                )
                .reset_index()
            )

            # Merge with existing
            if not existing["by_month"].empty:
                combined = pd.concat([existing["by_month"], new_by_month], ignore_index=True)
                # Re-aggregate overlapping months
                combined = (
                    combined.groupby("year_month")
                    .agg({
                        "mean_sentiment": "mean",
                        "positive_ratio": "mean",
                        "negative_ratio": "mean",
                        "total_count": "sum",
                    })
                    .reset_index()
                    .sort_values("year_month")
                )
            else:
                combined = new_by_month

            combined.to_csv(sentiment_dir / "sentiment_by_month.csv", index=False)

        if "subreddit" in new_df.columns:
            new_by_sub = (
                new_df.groupby("subreddit")
                .agg(
                    mean_sentiment=("sentiment_score", "mean"),
                    positive_ratio=("sentiment_label", lambda x: (x == "positive").mean()),
                    negative_ratio=("sentiment_label", lambda x: (x == "negative").mean()),
                    total_count=("id", "count"),
                )
                .reset_index()
            )

            if not existing["by_subreddit"].empty:
                combined = pd.concat([existing["by_subreddit"], new_by_sub], ignore_index=True)
                combined = (
                    combined.groupby("subreddit")
                    .agg({
                        "mean_sentiment": "mean",
                        "positive_ratio": "mean",
                        "negative_ratio": "mean",
                        "total_count": "sum",
                    })
                    .reset_index()
                )
            else:
                combined = new_by_sub

            combined.to_csv(sentiment_dir / "sentiment_by_subreddit.csv", index=False)

        if "subreddit" in new_df.columns and "year_month" in new_df.columns:
            new_by_both = (
                new_df.groupby(["subreddit", "year_month"])
                .agg(
                    mean_sentiment=("sentiment_score", "mean"),
                    positive_ratio=("sentiment_label", lambda x: (x == "positive").mean()),
                    negative_ratio=("sentiment_label", lambda x: (x == "negative").mean()),
                    total_count=("id", "count"),
                )
                .reset_index()
            )

            if not existing["by_subreddit_month"].empty:
                combined = pd.concat(
                    [existing["by_subreddit_month"], new_by_both], ignore_index=True
                )
                combined = (
                    combined.groupby(["subreddit", "year_month"])
                    .agg({
                        "mean_sentiment": "mean",
                        "positive_ratio": "mean",
                        "negative_ratio": "mean",
                        "total_count": "sum",
                    })
                    .reset_index()
                    .sort_values(["subreddit", "year_month"])
                )
            else:
                combined = new_by_both

            combined.to_csv(
                sentiment_dir / "sentiment_by_subreddit_month.csv", index=False
            )

        logger.info("Updated sentiment output files")

    def run_topics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign topics to new documents.

        Strategy:
        1. Try loading existing BERTopic model
        2. If model incompatible, use embedding similarity to existing topic
           representative embeddings (cosine similarity fallback)
        """
        topics_dir = self.config.outputs_dir / "topics"
        model_path = topics_dir / "bertopic_model"

        # Try BERTopic model first
        if model_path.exists():
            try:
                from bertopic import BERTopic
                logger.info("Loading existing BERTopic model...")
                topic_model = BERTopic.load(str(model_path))
                texts = df["text"].tolist()
                topics, probs = topic_model.transform(texts)
                df["topic_id"] = topics
                df["topic_prob"] = probs
                topic_info = topic_model.get_topic_info()
                id_to_name = dict(zip(topic_info["Topic"], topic_info["Name"]))
                df["topic_label"] = df["topic_id"].map(id_to_name).fillna("Unknown")
                logger.info(f"Topic assignment done (BERTopic). {(df['topic_id'] >= 0).sum()} assigned")
                return df
            except Exception as e:
                logger.warning(f"BERTopic model load failed: {e}")
                logger.info("Falling back to embedding-based topic assignment...")

        # Fallback: embedding similarity using existing topic assignments
        return self._assign_topics_by_embedding(df, topics_dir)

    def _assign_topics_by_embedding(self, df: pd.DataFrame, topics_dir: Path) -> pd.DataFrame:
        """Assign topics using cosine similarity to topic centroids.

        Uses saved embeddings + topic assignments to compute topic centroids,
        then assigns new documents to nearest centroid.
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        clusters_dir = self.config.outputs_dir / "clusters"
        embeddings_path = topics_dir / "document_embeddings.npy"
        assignments_path = topics_dir / "topic_assignments.parquet"
        topic_info_path = topics_dir / "topic_info.csv"

        if not assignments_path.exists() or not topic_info_path.exists():
            logger.warning("No topic assignments or info found. Skipping topics.")
            return df

        # Load topic info for labels
        topic_info = pd.read_csv(topic_info_path)
        id_to_name = dict(zip(topic_info["Topic"], topic_info["Name"]))

        # Compute topic centroids from existing embeddings
        if embeddings_path.exists():
            logger.info("Computing topic centroids from existing embeddings...")
            existing_embeddings = np.load(embeddings_path)
            existing_assignments = pd.read_parquet(assignments_path)

            # Build centroids for each topic
            centroids = {}
            for topic_id in topic_info[topic_info["Topic"] >= 0]["Topic"]:
                mask = existing_assignments["topic_id"] == topic_id
                if mask.sum() > 0:
                    indices = existing_assignments.index[mask].tolist()
                    # Limit indices to embedding array size
                    valid_indices = [i for i in indices if i < len(existing_embeddings)]
                    if valid_indices:
                        centroids[topic_id] = existing_embeddings[valid_indices].mean(axis=0)

            if not centroids:
                logger.warning("Could not compute centroids. Skipping topics.")
                return df

            centroid_ids = list(centroids.keys())
            centroid_matrix = np.array([centroids[tid] for tid in centroid_ids])

            # Embed new documents
            logger.info(f"Embedding {len(df)} new documents...")
            model = SentenceTransformer(self.config.embedding_model)
            new_embeddings = model.encode(df["text"].tolist(), show_progress_bar=False, batch_size=self.config.batch_size)

            # Assign by cosine similarity
            sims = cosine_similarity(new_embeddings, centroid_matrix)
            best_idx = sims.argmax(axis=1)
            best_sim = sims.max(axis=1)

            # Assign topic if similarity > threshold, else -1 (outlier)
            threshold = 0.25
            df["topic_id"] = [
                centroid_ids[idx] if best_sim[i] > threshold else -1
                for i, idx in enumerate(best_idx)
            ]
            df["topic_prob"] = best_sim
            df["topic_label"] = df["topic_id"].map(id_to_name).fillna("Outlier")

            assigned = (df["topic_id"] >= 0).sum()
            logger.info(f"Topic assignment done (embedding fallback). {assigned}/{len(df)} assigned")
        else:
            logger.warning("No embeddings found. Skipping topics.")

        return df

    def update_topic_outputs(self, new_df: pd.DataFrame) -> None:
        """Update topic output CSVs with new data."""
        topics_dir = self.config.outputs_dir / "topics"

        if "topic_id" not in new_df.columns:
            return

        # Update topics over time
        tot_path = topics_dir / "topics_over_time.csv"
        if tot_path.exists() and "year_month" in new_df.columns:
            existing_tot = pd.read_csv(tot_path)

            new_tot = (
                new_df[new_df["topic_id"] >= 0]
                .groupby(["topic_id", "year_month"])
                .size()
                .reset_index(name="Frequency")
                .rename(columns={"topic_id": "Topic", "year_month": "Timestamp"})
            )

            if not new_tot.empty:
                combined = pd.concat([existing_tot, new_tot], ignore_index=True)
                combined = (
                    combined.groupby(["Topic", "Timestamp"])
                    .agg({"Frequency": "sum"})
                    .reset_index()
                    .sort_values(["Topic", "Timestamp"])
                )
                combined.to_csv(tot_path, index=False)

        # Update topics by subreddit
        tbs_path = topics_dir / "topics_by_subreddit.csv"
        if tbs_path.exists() and "subreddit" in new_df.columns:
            existing_tbs = pd.read_csv(tbs_path)

            new_tbs = (
                new_df[new_df["topic_id"] >= 0]
                .groupby(["subreddit", "topic_id"])
                .size()
                .reset_index(name="count")
            )

            if not new_tbs.empty:
                combined = pd.concat([existing_tbs, new_tbs], ignore_index=True)
                combined = (
                    combined.groupby(["subreddit", "topic_id"])
                    .agg({"count": "sum"})
                    .reset_index()
                )
                combined.to_csv(tbs_path, index=False)

        logger.info("Updated topic output files")

    def run_and_update(self, reddit_df: pd.DataFrame) -> dict:
        """Run full incremental analysis and update outputs."""
        results = {}

        if reddit_df.empty:
            logger.info("No new Reddit data to analyze")
            return results

        # Sentiment
        try:
            reddit_df = self.run_sentiment(reddit_df)
            self.update_sentiment_outputs(reddit_df)
            results["sentiment"] = {
                "mean_score": float(reddit_df["sentiment_score"].mean()),
                "count": len(reddit_df),
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            results["sentiment"] = {"error": str(e)}

        # Topics
        try:
            reddit_df = self.run_topics(reddit_df)
            self.update_topic_outputs(reddit_df)
            if "topic_id" in reddit_df.columns:
                results["topics"] = {
                    "assigned": int((reddit_df["topic_id"] >= 0).sum()),
                    "count": len(reddit_df),
                }
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            results["topics"] = {"error": str(e)}

        return results
