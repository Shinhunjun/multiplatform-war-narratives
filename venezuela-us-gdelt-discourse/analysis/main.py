"""
Main GDELT discourse analysis pipeline runner.

Usage:
    python -m analysis.main --all
    python -m analysis.main --sentiment
    python -m analysis.main --topics
    python -m analysis.main --clusters
    python -m analysis.main --visualize
    python -m analysis.main --summarize
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .data_loader import load_all_data


def _load_saved_cluster_assignments(config: AnalysisConfig) -> Optional[pd.DataFrame]:
    """Execute _load_saved_cluster_assignments."""
    path = config.output_dir / "clusters" / "cluster_assignments.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def run_sentiment_analysis(
    df: pd.DataFrame,
    config: AnalysisConfig,
    save: bool = True,
) -> pd.DataFrame:
    """Run sentiment analysis on all loaded rows."""
    from .sentiment import analyze_dataframe, aggregate_sentiment, get_sentiment_summary

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)

    df = analyze_dataframe(
        df,
        text_column="text",
        model_name=config.sentiment_model,
        batch_size=config.batch_size,
    )

    summary = get_sentiment_summary(df)
    print("\nOverall Summary:")
    print(f"  Mean sentiment: {summary['mean_sentiment']:.3f}")
    print(f"  Positive: {summary['positive_ratio'] * 100:.1f}%")
    print(f"  Negative: {summary['negative_ratio'] * 100:.1f}%")

    group_col = config.source_group_column
    agg_group = aggregate_sentiment(df, [group_col]) if group_col in df.columns else pd.DataFrame()
    agg_monthly = aggregate_sentiment(df, ["year_month"])
    agg_both = aggregate_sentiment(df, [group_col, "year_month"]) if group_col in df.columns else pd.DataFrame()
    agg_event_category = (
        aggregate_sentiment(df, ["event_category"]) if "event_category" in df.columns else pd.DataFrame()
    )

    if save:
        output_dir = config.output_dir / "sentiment"
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_dir / "sentiment_full.parquet", index=False)
        if not agg_group.empty:
            agg_group.to_csv(output_dir / "sentiment_by_source_domain.csv", index=False)
        agg_monthly.to_csv(output_dir / "sentiment_by_month.csv", index=False)
        if not agg_both.empty:
            agg_both.to_csv(output_dir / "sentiment_by_source_domain_month.csv", index=False)
        if not agg_event_category.empty:
            agg_event_category.to_csv(output_dir / "sentiment_by_event_category.csv", index=False)

        print(f"\nSaved sentiment results to {output_dir}")

    return df


def run_topic_modeling(
    df: pd.DataFrame,
    config: AnalysisConfig,
    save: bool = True,
) -> Tuple[pd.DataFrame, object, np.ndarray]:
    """Run BERTopic topic modeling."""
    from .topic import fit_topics, get_topic_info, topics_over_time, aggregate_topics_by_group

    print("\n" + "=" * 60)
    print("TOPIC MODELING (BERTopic)")
    print("=" * 60)

    df, topic_model, embeddings = fit_topics(
        df,
        text_column="text",
        embedding_model=config.embedding_model,
        n_topics=config.n_topics,
        min_topic_size=config.min_cluster_size,
    )

    topic_info = get_topic_info(topic_model)
    print("\nTop Topics:")
    for _, row in topic_info.head(10).iterrows():
        if row["Topic"] != -1:
            print(f"  Topic {row['Topic']}: {row['Name'][:60]}... ({row['Count']:,} docs)")

    texts = df["text"].tolist()
    timestamps = df["created_datetime"].tolist()
    tot = topics_over_time(topic_model, texts, timestamps, nr_bins=30)

    group_col = config.source_group_column
    topic_by_group = aggregate_topics_by_group(df, [group_col]) if group_col in df.columns else pd.DataFrame()

    if save:
        output_dir = config.output_dir / "topics"
        output_dir.mkdir(parents=True, exist_ok=True)

        topic_model.save(str(output_dir / "bertopic_model"))

        save_cols = [
            "id",
            "type",
            "year_month",
            "topic_id",
            "topic_label",
            "topic_prob",
        ]
        optional_cols = [
            "source_domain",
            "event_category",
            "actor_pair",
            "doc_relevance_score",
        ]
        save_cols.extend([c for c in optional_cols if c in df.columns])

        df[save_cols].to_parquet(output_dir / "topic_assignments.parquet", index=False)
        topic_info.to_csv(output_dir / "topic_info.csv", index=False)
        tot.to_csv(output_dir / "topics_over_time.csv", index=False)
        if not topic_by_group.empty:
            topic_by_group.to_csv(output_dir / "topics_by_source_domain.csv", index=False)

        np.save(output_dir / "document_embeddings.npy", embeddings)
        print(f"\nSaved topic results to {output_dir}")

    return df, topic_model, embeddings


def run_clustering(
    df: pd.DataFrame,
    embeddings: Optional[np.ndarray],
    config: AnalysisConfig,
    save: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run clustering analysis."""
    from .clustering import TextEmbedder, reduce_dimensions, TemporalClusterer

    print("\n" + "=" * 60)
    print("CLUSTERING ANALYSIS")
    print("=" * 60)

    if embeddings is None:
        embedder = TextEmbedder(model_name=config.embedding_model)
        embeddings, _ = embedder.embed_dataframe(df, text_column="text", id_column="id")
    else:
        print(f"Using provided embeddings: {embeddings.shape}")

    clusterer = TemporalClusterer(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        group_column=config.source_group_column,
    )
    labels = clusterer.fit(embeddings)
    df = clusterer.add_clusters_to_df(df, labels)

    cluster_summary = clusterer.get_cluster_summary(df)
    print("\nTop Clusters:")
    if cluster_summary.empty:
        print("  No non-noise clusters found.")
    else:
        for _, row in cluster_summary.head(10).iterrows():
            top_group = row.get("top_group", "unknown")
            print(f"  Cluster {row['cluster_id']}: {row['count']:,} docs, top source: {top_group}")

    temporal_clusters = clusterer.get_temporal_clusters(df)

    embeddings_2d = reduce_dimensions(embeddings, n_components=2)
    df["umap_1"] = embeddings_2d[:, 0]
    df["umap_2"] = embeddings_2d[:, 1]

    if save:
        output_dir = config.output_dir / "clusters"
        output_dir.mkdir(parents=True, exist_ok=True)

        save_cols = [
            "id",
            "type",
            "year_month",
            "cluster_id",
            "cluster_prob",
            "umap_1",
            "umap_2",
        ]
        optional_cols = [
            "source_domain",
            "event_category",
            "actor_pair",
            "doc_relevance_score",
            "avg_tone",
            "goldstein_scale",
        ]
        save_cols.extend([c for c in optional_cols if c in df.columns])

        df[save_cols].to_parquet(output_dir / "cluster_assignments.parquet", index=False)
        cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)
        temporal_clusters.to_csv(output_dir / "temporal_clusters.csv", index=False)

        np.save(output_dir / "embeddings.npy", embeddings)
        np.save(output_dir / "embeddings_2d.npy", embeddings_2d)

        print(f"\nSaved cluster results to {output_dir}")

    return df, embeddings, embeddings_2d


def run_visualizations(
    df: pd.DataFrame,
    embeddings_2d: Optional[np.ndarray],
    config: AnalysisConfig,
) -> None:
    """Generate all visualizations."""
    from .clustering import (
        create_umap_scatter,
        create_animated_umap,
        create_cluster_river_plot,
        create_cluster_heatmap,
        create_interactive_scatter,
    )

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    if embeddings_2d is None and {"umap_1", "umap_2"}.issubset(df.columns):
        embeddings_2d = df[["umap_1", "umap_2"]].to_numpy()

    if embeddings_2d is None:
        raise ValueError("No 2D embeddings available. Run clustering first.")

    output_dir = config.output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating UMAP scatter plot...")
    create_umap_scatter(
        embeddings_2d,
        df,
        color_column="cluster_id",
        output_path=output_dir / "umap_clusters.png",
        title="GDELT Document Clusters (UMAP)",
    )

    if config.source_group_column in df.columns:
        create_umap_scatter(
            embeddings_2d,
            df,
            color_column=config.source_group_column,
            output_path=output_dir / "umap_source_domains.png",
            title="Documents by Source Domain (UMAP)",
        )

    print("Creating animated UMAP (this may take a while)...")
    try:
        create_animated_umap(
            embeddings_2d,
            df,
            time_column="year_month",
            color_column="cluster_id",
            output_path=output_dir / "umap_animation.gif",
            fps=2,
        )
    except Exception as e:
        print(f"Could not create animation: {e}")

    print("Creating cluster river plot...")
    create_cluster_river_plot(
        df,
        time_column="year_month",
        output_path=output_dir / "cluster_river.png",
        top_n_clusters=10,
    )

    if config.source_group_column in df.columns:
        print("Creating cluster heatmap...")
        create_cluster_heatmap(
            df,
            group_column=config.source_group_column,
            output_path=output_dir / "cluster_heatmap.png",
        )

    print("Creating interactive visualization...")
    create_interactive_scatter(
        embeddings_2d,
        df,
        output_path=output_dir / "interactive_clusters.html",
    )

    print(f"\nSaved visualizations to {output_dir}")


def run_cluster_summarization(
    df: pd.DataFrame,
    config: AnalysisConfig,
    llm_provider: str = "anthropic",
) -> pd.DataFrame:
    """Generate keyword and LLM summaries for clusters."""
    from .clustering import summarize_all_clusters, generate_keyword_summary

    print("\n" + "=" * 60)
    print("CLUSTER SUMMARIZATION")
    print("=" * 60)

    if "cluster_id" not in df.columns:
        raise ValueError("DataFrame must include cluster_id before summarization.")

    output_dir = config.output_dir / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating keyword summaries...")
    cluster_ids = sorted(df[df["cluster_id"] != -1]["cluster_id"].unique())

    keyword_summaries = []
    for cluster_id in cluster_ids:
        keywords = generate_keyword_summary(df, cluster_id)
        keyword_summaries.append(
            {
                "cluster_id": cluster_id,
                "keywords": ", ".join(keywords[:10]),
            }
        )

    keywords_df = pd.DataFrame(keyword_summaries)
    keywords_df.to_csv(output_dir / "cluster_keywords.csv", index=False)

    try:
        print(f"Generating LLM summaries using {llm_provider}...")
        summaries_df = summarize_all_clusters(
            df,
            n_samples=config.samples_per_cluster,
            llm_provider=llm_provider,
        )
        summaries_df.to_csv(output_dir / "cluster_summaries.csv", index=False)
        print(f"Saved summaries to {output_dir / 'cluster_summaries.csv'}")
        return summaries_df
    except Exception as e:
        print(f"Could not generate LLM summaries: {e}")
        print("Keyword summaries are still available in cluster_keywords.csv")
        return keywords_df


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Venezuela-US GDELT Discourse Analysis Pipeline")

    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--sentiment", action="store_true", help="Run sentiment analysis")
    parser.add_argument("--topics", action="store_true", help="Run topic modeling")
    parser.add_argument("--clusters", action="store_true", help="Run clustering")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--summarize", action="store_true", help="Generate cluster summaries")
    parser.add_argument("--llm", default="anthropic", choices=["anthropic", "openai"], help="LLM provider")
    parser.add_argument("--sample", type=int, default=None, help="Sample N rows for testing")
    parser.add_argument("--min-relevance", type=float, default=None, help="Filter by doc_relevance_score >= value")
    parser.add_argument(
        "--exclude-suspect-redirect",
        action="store_true",
        help="Exclude rows flagged as suspect redirect duplicates in preprocessing",
    )
    parser.add_argument(
        "--include-non-success",
        action="store_true",
        help="Include non-success scrape statuses (if they have text)",
    )

    args = parser.parse_args()
    if not any([args.all, args.sentiment, args.topics, args.clusters, args.visualize, args.summarize]):
        args.all = True

    config = AnalysisConfig()
    if args.min_relevance is not None:
        config.min_doc_relevance_score = args.min_relevance
    if args.exclude_suspect_redirect:
        config.exclude_suspect_redirect_content = True
    if args.include_non_success:
        config.require_successful_scrape = False
    config.ensure_directories()

    print("Loading data...")
    df = load_all_data(config)

    if args.sample:
        print(f"Sampling {args.sample:,} rows for testing...")
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)

    embeddings = None
    embeddings_2d = None

    if args.all or args.sentiment:
        df = run_sentiment_analysis(df, config)

    if args.all or args.topics:
        df, _, embeddings = run_topic_modeling(df, config)

    if args.all or args.clusters:
        df, embeddings, embeddings_2d = run_clustering(df, embeddings, config)

    if args.all or args.visualize:
        if "cluster_id" not in df.columns or not {"umap_1", "umap_2"}.issubset(df.columns):
            saved = _load_saved_cluster_assignments(config)
            if saved is not None:
                print("Using saved cluster assignments for visualization.")
                df = saved
                if args.sample and len(df) > args.sample:
                    df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
                embeddings_2d = df[["umap_1", "umap_2"]].to_numpy()
            else:
                raise FileNotFoundError("No saved cluster assignments found. Run with --clusters first.")

        run_visualizations(df, embeddings_2d, config)

    if args.all or args.summarize:
        if "cluster_id" not in df.columns:
            saved = _load_saved_cluster_assignments(config)
            if saved is None:
                raise FileNotFoundError("No saved cluster assignments found. Run with --clusters first.")
            print("Merging saved cluster assignments for summarization.")
            merge_cols = ["id", "cluster_id", "cluster_prob"]
            merged = df.merge(saved[merge_cols], on="id", how="inner")
            if merged.empty:
                print("No overlap between sampled data and saved clusters. Reloading full data for summarization.")
                full_df = load_all_data(config)
                merged = full_df.merge(saved[merge_cols], on="id", how="inner")
            if args.sample and len(merged) > args.sample:
                merged = merged.sample(n=args.sample, random_state=42).reset_index(drop=True)
            df = merged

        run_cluster_summarization(df, config, args.llm)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
