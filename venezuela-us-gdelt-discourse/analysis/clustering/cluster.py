"""
HDBSCAN clustering with ID tracking and temporal analysis.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


class TemporalClusterer:
    """
    Clusterer with temporal analysis support.
    """

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        cluster_selection_method: str = "eom",
        group_column: str = "source_domain",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.group_column = group_column

        self.labels: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None
        self.cluster_info: Optional[pd.DataFrame] = None

    def fit(
        self,
        embeddings: np.ndarray,
        reduce_first: bool = True,
        n_components: int = 50,
    ) -> np.ndarray:
        """
        Fit HDBSCAN clustering on embeddings.

        Returns:
            Cluster labels (-1 for noise)
        """
        from hdbscan import HDBSCAN

        data = embeddings

        # Optionally reduce dimensions for faster clustering.
        if reduce_first and embeddings.shape[1] > n_components:
            from umap import UMAP

            print(f"Reducing to {n_components} dimensions for clustering...")
            reducer = UMAP(
                n_components=n_components,
                metric="cosine",
                random_state=42,
            )
            data = reducer.fit_transform(embeddings)

        print(f"Clustering {len(data):,} documents...")

        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            metric="euclidean",
            prediction_data=True,
        )
        clusterer.fit(data)

        self.labels = clusterer.labels_
        self.probabilities = clusterer.probabilities_

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = int((self.labels == -1).sum())
        noise_pct = (n_noise / len(self.labels) * 100.0) if len(self.labels) else 0.0
        print(f"Found {n_clusters} clusters, {n_noise:,} noise points ({noise_pct:.1f}%)")

        return self.labels

    def add_clusters_to_df(
        self,
        df: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Add cluster assignments to DataFrame."""
        df = df.copy()
        labels = labels if labels is not None else self.labels

        if labels is None:
            raise ValueError("No cluster labels available. Run fit() first.")

        df["cluster_id"] = labels
        df["cluster_prob"] = self.probabilities if self.probabilities is not None else 1.0
        df["is_noise"] = df["cluster_id"] == -1
        return df

    def get_cluster_ids(self, cluster_id: int) -> List[int]:
        """Get indices of documents in a cluster."""
        if self.labels is None:
            return []
        return np.where(self.labels == cluster_id)[0].tolist()

    def get_cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.

        Returns DataFrame with:
        - cluster_id
        - count
        - top_group (dominant source group)
        - time range
        - sentiment_mean (if available)
        """
        if "cluster_id" not in df.columns:
            raise ValueError("DataFrame must have cluster_id column")

        df_valid = df[df["cluster_id"] != -1]
        if df_valid.empty:
            return pd.DataFrame(
                columns=[
                    "cluster_id",
                    "count",
                    "top_group",
                    "time_start",
                    "time_end",
                    "sentiment_mean",
                    "sentiment_std",
                ]
            )

        summaries = []
        for cluster_id in sorted(df_valid["cluster_id"].unique()):
            cluster_df = df_valid[df_valid["cluster_id"] == cluster_id]
            summary = {
                "cluster_id": cluster_id,
                "count": len(cluster_df),
            }

            if self.group_column in cluster_df.columns:
                group_counts = cluster_df[self.group_column].value_counts(dropna=True)
                summary["groups"] = group_counts.to_dict()
                summary["top_group"] = group_counts.index[0] if len(group_counts) else None
            else:
                summary["groups"] = {}
                summary["top_group"] = None

            if "created_datetime" in cluster_df.columns:
                summary["time_start"] = cluster_df["created_datetime"].min()
                summary["time_end"] = cluster_df["created_datetime"].max()

            if "sentiment_score" in cluster_df.columns:
                summary["sentiment_mean"] = cluster_df["sentiment_score"].mean()
                summary["sentiment_std"] = cluster_df["sentiment_score"].std()

            if "avg_tone" in cluster_df.columns:
                summary["gdelt_tone_mean"] = cluster_df["avg_tone"].mean()
                summary["gdelt_tone_std"] = cluster_df["avg_tone"].std()

            if "goldstein_scale" in cluster_df.columns:
                summary["gdelt_goldstein_mean"] = cluster_df["goldstein_scale"].mean()
                summary["gdelt_goldstein_std"] = cluster_df["goldstein_scale"].std()

            summaries.append(summary)

        return pd.DataFrame(summaries).sort_values("count", ascending=False).reset_index(drop=True)

    def get_temporal_clusters(
        self,
        df: pd.DataFrame,
        time_column: str = "year_month",
    ) -> pd.DataFrame:
        """
        Get cluster distributions over time.
        """
        if "cluster_id" not in df.columns:
            raise ValueError("DataFrame must have cluster_id column")

        df_valid = df[df["cluster_id"] != -1]
        temporal = df_valid.groupby([time_column, "cluster_id"]).size().reset_index(name="count")

        if temporal.empty:
            temporal["proportion"] = pd.Series(dtype=float)
            return temporal

        period_totals = temporal.groupby(time_column)["count"].transform("sum")
        temporal["proportion"] = temporal["count"] / period_totals
        return temporal

    def cluster_by_period(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        periods: List[str],
        time_column: str = "year_month",
    ) -> Dict[str, pd.DataFrame]:
        """
        Cluster separately for each time period.
        """
        results = {}

        for period in tqdm(periods, desc="Clustering by period"):
            mask = df[time_column] == period
            period_df = df[mask].copy()
            period_embeddings = embeddings[mask]

            if len(period_df) < self.min_cluster_size * 2:
                period_df["cluster_id"] = -1
                period_df["cluster_prob"] = 0.0
            else:
                from hdbscan import HDBSCAN

                clusterer = HDBSCAN(
                    min_cluster_size=max(10, self.min_cluster_size // 2),
                    min_samples=max(5, self.min_samples // 2),
                    cluster_selection_method=self.cluster_selection_method,
                    metric="euclidean",
                )
                labels = clusterer.fit_predict(period_embeddings)
                period_df["cluster_id"] = labels
                period_df["cluster_prob"] = clusterer.probabilities_

            period_df["period"] = period
            results[period] = period_df

        return results


def track_cluster_evolution(
    period_clusters: Dict[str, pd.DataFrame],
    embeddings: np.ndarray,
    df: pd.DataFrame,
    similarity_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Track how clusters evolve across time periods using centroid similarity.
    """
    periods = sorted(period_clusters.keys())
    evolution = []

    for i, period in enumerate(periods[:-1]):
        next_period = periods[i + 1]

        current_df = period_clusters[period]
        next_df = period_clusters[next_period]

        current_clusters = current_df[current_df["cluster_id"] != -1]["cluster_id"].unique()
        next_clusters = next_df[next_df["cluster_id"] != -1]["cluster_id"].unique()

        for curr_cluster in current_clusters:
            curr_mask = df["id"].isin(current_df[current_df["cluster_id"] == curr_cluster]["id"])
            curr_centroid = embeddings[curr_mask].mean(axis=0)

            best_match = None
            best_sim = 0.0

            for next_cluster in next_clusters:
                next_mask = df["id"].isin(next_df[next_df["cluster_id"] == next_cluster]["id"])
                next_centroid = embeddings[next_mask].mean(axis=0)

                sim = np.dot(curr_centroid, next_centroid) / (
                    np.linalg.norm(curr_centroid) * np.linalg.norm(next_centroid)
                )

                if sim > best_sim:
                    best_sim = sim
                    best_match = next_cluster

            if best_sim >= similarity_threshold:
                evolution.append(
                    {
                        "from_period": period,
                        "to_period": next_period,
                        "from_cluster": curr_cluster,
                        "to_cluster": best_match,
                        "similarity": best_sim,
                        "from_count": len(current_df[current_df["cluster_id"] == curr_cluster]),
                        "to_count": len(next_df[next_df["cluster_id"] == best_match]) if best_match else 0,
                    }
                )

    return pd.DataFrame(evolution)
