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
        """Transform new documents using existing BERTopic model."""
        topics_dir = self.config.outputs_dir / "topics"
        model_path = topics_dir / "bertopic_model"

        if not model_path.exists():
            logger.warning("No existing BERTopic model found. Skipping topics.")
            return df

        from bertopic import BERTopic

        logger.info("Loading existing BERTopic model...")
        topic_model = BERTopic.load(str(model_path))

        texts = df["text"].tolist()
        topics, probs = topic_model.transform(texts)

        df["topic_id"] = topics
        df["topic_prob"] = probs

        # Get topic labels
        topic_info = topic_model.get_topic_info()
        id_to_name = dict(zip(topic_info["Topic"], topic_info["Name"]))
        df["topic_label"] = df["topic_id"].map(id_to_name).fillna("Unknown")

        logger.info(f"Topic assignment done. {(df['topic_id'] >= 0).sum()} assigned to topics")
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
        reddit_df = self.run_sentiment(reddit_df)
        self.update_sentiment_outputs(reddit_df)
        results["sentiment"] = {
            "mean_score": float(reddit_df["sentiment_score"].mean()),
            "count": len(reddit_df),
        }

        # Topics
        reddit_df = self.run_topics(reddit_df)
        self.update_topic_outputs(reddit_df)
        if "topic_id" in reddit_df.columns:
            results["topics"] = {
                "assigned": int((reddit_df["topic_id"] >= 0).sum()),
                "count": len(reddit_df),
            }

        return results
