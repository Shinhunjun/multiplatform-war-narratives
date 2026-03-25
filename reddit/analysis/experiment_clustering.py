"""
Clustering Hyperparameter Experiment
=====================================
Tests UMAP + HDBSCAN parameter combinations across 3 representative periods
(low / medium / high document density) to derive an adaptive parameter strategy.

Usage:
    python experiment_clustering.py
"""

import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
from umap import UMAP

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
REDDIT_EMBEDS = BASE / "outputs" / "topics" / "document_embeddings.npy"
REDDIT_ASSIGNS = BASE / "outputs" / "topics" / "topic_assignments.parquet"
NEWS_EMBEDS = BASE / "outputs_news" / "topics" / "document_embeddings.npy"
NEWS_ASSIGNS = BASE / "outputs_news" / "topics" / "topic_assignments.parquet"
OUTPUT_DIR = BASE / "experiment_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Hyperparameter Grid ──────────────────────────────────────────────
UMAP_PARAMS = {
    "n_components": [5, 10, 15],
    "n_neighbors": [10, 15, 30],
    "min_dist": [0.0, 0.05, 0.1],
}

HDBSCAN_PARAMS = {
    "min_cluster_size": [10, 25, 50, 100],
    "min_samples": [5, 10, 20],
}

# ── Representative Periods ───────────────────────────────────────────
# Low density / Medium density / High density (crisis)
TARGET_MONTHS = {
    "low":    "2015-03",   # Early, sparse data
    "medium": "2020-06",   # Average activity
    "high":   "2019-02",   # Guaidó crisis peak
}


def load_month_data(assigns_path, embeds_path, year_month):
    """Load embeddings for a specific month."""
    df = pd.read_parquet(assigns_path, columns=["year_month"])
    mask = df["year_month"] == year_month
    indices = np.where(mask.values)[0]
    if len(indices) == 0:
        return None, 0
    embeds = np.load(embeds_path, mmap_mode="r")[indices]
    return np.array(embeds), len(indices)


def evaluate_clustering(embeddings, labels):
    """Compute clustering quality metrics."""
    n_total = len(labels)
    noise_mask = labels == -1
    n_noise = noise_mask.sum()
    n_clustered = n_total - n_noise
    n_clusters = len(set(labels) - {-1})

    noise_ratio = n_noise / n_total if n_total > 0 else 1.0

    # Silhouette on clustered points only (need >= 2 clusters, >= 2 points)
    sil = -1.0
    if n_clusters >= 2 and n_clustered >= 2:
        try:
            sil = silhouette_score(
                embeddings[~noise_mask], labels[~noise_mask],
                metric="euclidean", sample_size=min(5000, n_clustered),
                random_state=42,
            )
        except Exception:
            sil = -1.0

    # Average cluster size (excluding noise)
    avg_size = n_clustered / n_clusters if n_clusters > 0 else 0

    return {
        "n_clusters": n_clusters,
        "noise_ratio": round(noise_ratio, 4),
        "silhouette": round(sil, 4),
        "n_clustered": n_clustered,
        "avg_cluster_size": round(avg_size, 1),
    }


def run_experiment(embeddings, umap_params, hdbscan_params):
    """Run single UMAP + HDBSCAN experiment."""
    # UMAP reduction
    reducer = UMAP(
        n_components=umap_params["n_components"],
        n_neighbors=umap_params["n_neighbors"],
        min_dist=umap_params["min_dist"],
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    # HDBSCAN clustering
    clusterer = HDBSCAN(
        min_cluster_size=hdbscan_params["min_cluster_size"],
        min_samples=hdbscan_params["min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    metrics = evaluate_clustering(reduced, labels)
    return metrics


def run_all_experiments():
    """Run full grid search across representative periods and platforms."""
    print("=" * 70)
    print("CLUSTERING HYPERPARAMETER EXPERIMENT")
    print("=" * 70)

    # Build parameter combinations
    umap_combos = [
        dict(zip(UMAP_PARAMS.keys(), v))
        for v in itertools.product(*UMAP_PARAMS.values())
    ]
    hdb_combos = [
        dict(zip(HDBSCAN_PARAMS.keys(), v))
        for v in itertools.product(*HDBSCAN_PARAMS.values())
    ]
    total_combos = len(umap_combos) * len(hdb_combos)
    print(f"UMAP combos: {len(umap_combos)}, HDBSCAN combos: {len(hdb_combos)}")
    print(f"Total combos per period: {total_combos}")
    print()

    all_results = []

    for platform, assigns_path, embeds_path in [
        ("reddit", REDDIT_ASSIGNS, REDDIT_EMBEDS),
        ("news", NEWS_ASSIGNS, NEWS_EMBEDS),
    ]:
        if not assigns_path.exists() or not embeds_path.exists():
            print(f"[Skip] {platform}: data not found")
            continue

        for density, month in TARGET_MONTHS.items():
            embeds, n_docs = load_month_data(assigns_path, embeds_path, month)
            if embeds is None or n_docs < 30:
                print(f"[Skip] {platform}/{month} ({density}): {n_docs} docs (< 30)")
                continue

            # Cap at 10K docs for speed (random sample)
            if n_docs > 10000:
                rng = np.random.RandomState(42)
                sample_idx = rng.choice(n_docs, 10000, replace=False)
                embeds = embeds[sample_idx]
                n_docs_label = f"{n_docs} (sampled 10K)"
            else:
                n_docs_label = str(n_docs)

            print(f"\n{'─'*70}")
            print(f"Platform: {platform} | Month: {month} ({density}) | Docs: {n_docs_label}")
            print(f"{'─'*70}")

            # Pre-compute UMAP reductions (most expensive step)
            # Group experiments by UMAP params to avoid redundant reductions
            for ui, umap_p in enumerate(umap_combos):
                t0 = time.time()
                reducer = UMAP(
                    n_components=umap_p["n_components"],
                    n_neighbors=umap_p["n_neighbors"],
                    min_dist=umap_p["min_dist"],
                    metric="cosine",
                    random_state=42,
                )
                reduced = reducer.fit_transform(embeds)
                umap_time = time.time() - t0

                for hdb_p in hdb_combos:
                    # Skip if min_cluster_size > 30% of docs
                    if hdb_p["min_cluster_size"] > len(embeds) * 0.3:
                        continue

                    t1 = time.time()
                    clusterer = HDBSCAN(
                        min_cluster_size=hdb_p["min_cluster_size"],
                        min_samples=hdb_p["min_samples"],
                        metric="euclidean",
                        cluster_selection_method="eom",
                    )
                    labels = clusterer.fit_predict(reduced)
                    hdb_time = time.time() - t1

                    metrics = evaluate_clustering(reduced, labels)

                    result = {
                        "platform": platform,
                        "month": month,
                        "density": density,
                        "n_docs": len(embeds),
                        # UMAP params
                        "umap_n_components": umap_p["n_components"],
                        "umap_n_neighbors": umap_p["n_neighbors"],
                        "umap_min_dist": umap_p["min_dist"],
                        # HDBSCAN params
                        "hdb_min_cluster_size": hdb_p["min_cluster_size"],
                        "hdb_min_samples": hdb_p["min_samples"],
                        # Metrics
                        **metrics,
                        "time_umap": round(umap_time, 2),
                        "time_hdbscan": round(hdb_time, 2),
                    }
                    all_results.append(result)

                done = (ui + 1) * len(hdb_combos)
                print(f"  UMAP({umap_p['n_components']}D, nn={umap_p['n_neighbors']}, "
                      f"md={umap_p['min_dist']}) done [{done}/{total_combos}] "
                      f"({umap_time:.1f}s)", flush=True)

    # ── Save Results ──────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "clustering_experiment_results.csv", index=False)
    print(f"\n\nSaved {len(df)} results to {OUTPUT_DIR / 'clustering_experiment_results.csv'}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP CONFIGURATIONS PER PERIOD (by silhouette score)")
    print("=" * 70)

    for (platform, density), group in df.groupby(["platform", "density"]):
        valid = group[group["silhouette"] > 0].copy()
        if valid.empty:
            print(f"\n{platform}/{density}: No valid configurations")
            continue

        # Score = silhouette * (1 - noise_ratio) to balance quality and coverage
        valid["composite"] = valid["silhouette"] * (1 - valid["noise_ratio"])
        top = valid.nlargest(3, "composite")

        print(f"\n{platform}/{density} (month={top.iloc[0]['month']}, n={top.iloc[0]['n_docs']}):")
        for _, row in top.iterrows():
            print(f"  UMAP({row['umap_n_components']}D, nn={row['umap_n_neighbors']}, "
                  f"md={row['umap_min_dist']}) + "
                  f"HDBSCAN(mcs={row['hdb_min_cluster_size']}, ms={row['hdb_min_samples']}) "
                  f"→ sil={row['silhouette']:.3f}, noise={row['noise_ratio']:.1%}, "
                  f"clusters={row['n_clusters']}, composite={row['composite']:.3f}")

    # ── Derive Adaptive Rule ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ADAPTIVE RULE DERIVATION")
    print("=" * 70)

    for platform in df["platform"].unique():
        pdf = df[df["platform"] == platform]
        print(f"\n[{platform}] Best min_cluster_size by density:")
        for density in ["low", "medium", "high"]:
            ddf = pdf[(pdf["density"] == density) & (pdf["silhouette"] > 0)].copy()
            if ddf.empty:
                print(f"  {density}: insufficient data")
                continue
            ddf["composite"] = ddf["silhouette"] * (1 - ddf["noise_ratio"])
            best = ddf.nlargest(1, "composite").iloc[0]
            ratio = best["hdb_min_cluster_size"] / best["n_docs"]
            print(f"  {density} (n={best['n_docs']}): "
                  f"best mcs={best['hdb_min_cluster_size']} "
                  f"(ratio={ratio:.4f}, ~{ratio*100:.1f}% of n_docs)")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
