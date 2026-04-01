"""
Main analysis pipeline runner.

Usage:
    python -m analysis.main --all              # Run full pipeline
    python -m analysis.main --sentiment        # Run sentiment only
    python -m analysis.main --topics           # Run topic modeling only
    python -m analysis.main --clusters         # Run clustering only
    python -m analysis.main --visualize        # Generate visualizations only
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import AnalysisConfig, CRISIS_PERIODS
from .data_loader import load_all_data, get_time_periods


def run_sentiment_analysis(
    df: pd.DataFrame,
    config: AnalysisConfig,
    save: bool = True,
) -> pd.DataFrame:
    """Run sentiment analysis on all data."""
    from .sentiment import analyze_dataframe, aggregate_sentiment, get_sentiment_summary

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)

    # Analyze
    df = analyze_dataframe(
        df,
        text_column="text",
        model_name=config.sentiment_model,
        batch_size=config.batch_size,
    )

    # Print summary
    summary = get_sentiment_summary(df)
    print(f"\nOverall Summary:")
    print(f"  Mean sentiment: {summary['mean_sentiment']:.3f}")
    print(f"  Positive: {summary['positive_ratio']*100:.1f}%")
    print(f"  Negative: {summary['negative_ratio']*100:.1f}%")

    # Aggregate by subreddit and month
    agg_subreddit = aggregate_sentiment(df, ["subreddit"])
    agg_monthly = aggregate_sentiment(df, ["year_month"])
    agg_both = aggregate_sentiment(df, ["subreddit", "year_month"])

    if save:
        output_dir = config.output_dir / "sentiment"
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_dir / "sentiment_full.parquet", index=False)
        agg_subreddit.to_csv(output_dir / "sentiment_by_subreddit.csv", index=False)
        agg_monthly.to_csv(output_dir / "sentiment_by_month.csv", index=False)
        agg_both.to_csv(output_dir / "sentiment_by_subreddit_month.csv", index=False)

        print(f"\nSaved sentiment results to {output_dir}")

    return df


def run_topic_modeling(
    df: pd.DataFrame,
    config: AnalysisConfig,
    save: bool = True,
) -> tuple:
    """Run topic modeling on all data."""
    from .topic import fit_topics, get_topic_info, topics_over_time, aggregate_topics_by_group

    print("\n" + "=" * 60)
    print("TOPIC MODELING (BERTopic)")
    print("=" * 60)

    # Fit model
    df, topic_model, embeddings = fit_topics(
        df,
        text_column="text",
        embedding_model=config.embedding_model,
        n_topics=config.n_topics,
        min_topic_size=config.min_cluster_size,
    )

    # Get topic info
    topic_info = get_topic_info(topic_model)
    print(f"\nTop Topics:")
    for _, row in topic_info.head(10).iterrows():
        if row["Topic"] != -1:
            print(f"  Topic {row['Topic']}: {row['Name'][:50]}... ({row['Count']:,} docs)")

    # Topics over time
    texts = df["text"].tolist()
    timestamps = df["created_datetime"].tolist()
    tot = topics_over_time(topic_model, texts, timestamps, nr_bins=30)

    # Aggregate by subreddit
    topic_by_subreddit = aggregate_topics_by_group(df, ["subreddit"])

    if save:
        output_dir = config.output_dir / "topics"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        topic_model.save(str(output_dir / "bertopic_model"))

        # Save results
        df[["id", "type", "subreddit", "year_month", "topic_id", "topic_label", "topic_prob"]].to_parquet(
            output_dir / "topic_assignments.parquet", index=False
        )
        topic_info.to_csv(output_dir / "topic_info.csv", index=False)
        tot.to_csv(output_dir / "topics_over_time.csv", index=False)
        topic_by_subreddit.to_csv(output_dir / "topics_by_subreddit.csv", index=False)

        # Save embeddings for clustering
        np.save(output_dir / "document_embeddings.npy", embeddings)

        print(f"\nSaved topic results to {output_dir}")

    return df, topic_model, embeddings


def run_clustering(
    df: pd.DataFrame,
    embeddings: Optional[np.ndarray],
    config: AnalysisConfig,
    save: bool = True,
) -> pd.DataFrame:
    """Run clustering analysis."""
    from .clustering import (
        TextEmbedder,
        reduce_dimensions,
        TemporalClusterer,
        track_cluster_evolution,
    )

    print("\n" + "=" * 60)
    print("CLUSTERING ANALYSIS")
    print("=" * 60)

    # Load or create embeddings
    if embeddings is None:
        embedder = TextEmbedder(model_name=config.embedding_model)
        embeddings, index_df = embedder.embed_dataframe(df)
    else:
        print(f"Using provided embeddings: {embeddings.shape}")

    # Global clustering
    clusterer = TemporalClusterer(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
    )
    labels = clusterer.fit(embeddings)
    df = clusterer.add_clusters_to_df(df, labels)

    # Cluster summary
    cluster_summary = clusterer.get_cluster_summary(df)
    print(f"\nTop Clusters:")
    for _, row in cluster_summary.head(10).iterrows():
        print(f"  Cluster {row['cluster_id']}: {row['count']:,} docs, top subreddit: r/{row['top_subreddit']}")

    # Temporal analysis
    temporal_clusters = clusterer.get_temporal_clusters(df)

    # Reduce for visualization
    embeddings_2d = reduce_dimensions(embeddings, n_components=2)
    df["umap_1"] = embeddings_2d[:, 0]
    df["umap_2"] = embeddings_2d[:, 1]

    if save:
        output_dir = config.output_dir / "clusters"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save cluster assignments
        df[["id", "type", "subreddit", "year_month", "cluster_id", "cluster_prob", "umap_1", "umap_2"]].to_parquet(
            output_dir / "cluster_assignments.parquet", index=False
        )
        cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)
        temporal_clusters.to_csv(output_dir / "temporal_clusters.csv", index=False)

        # Save embeddings
        np.save(output_dir / "embeddings.npy", embeddings)
        np.save(output_dir / "embeddings_2d.npy", embeddings_2d)

        print(f"\nSaved cluster results to {output_dir}")

    return df, embeddings, embeddings_2d


def run_visualizations(
    df: pd.DataFrame,
    embeddings_2d: np.ndarray,
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

    output_dir = config.output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Static UMAP scatter
    print("Creating UMAP scatter plot...")
    create_umap_scatter(
        embeddings_2d,
        df,
        color_column="cluster_id",
        output_path=output_dir / "umap_clusters.png",
        title="Document Clusters (UMAP)",
    )

    # By subreddit
    create_umap_scatter(
        embeddings_2d,
        df,
        color_column="subreddit",
        output_path=output_dir / "umap_subreddits.png",
        title="Documents by Subreddit (UMAP)",
    )

    # Animated UMAP
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

    # River plot
    print("Creating cluster river plot...")
    create_cluster_river_plot(
        df,
        time_column="year_month",
        output_path=output_dir / "cluster_river.png",
        top_n_clusters=10,
    )

    # Heatmap
    print("Creating cluster heatmap...")
    create_cluster_heatmap(
        df,
        group_column="subreddit",
        output_path=output_dir / "cluster_heatmap.png",
    )

    # Interactive plot
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
    """Generate LLM summaries for clusters."""
    from .clustering import summarize_all_clusters, generate_keyword_summary

    print("\n" + "=" * 60)
    print("CLUSTER SUMMARIZATION")
    print("=" * 60)

    output_dir = config.output_dir / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, generate keyword summaries (no API needed)
    print("Generating keyword summaries...")
    cluster_ids = sorted(df[df["cluster_id"] != -1]["cluster_id"].unique())

    keyword_summaries = []
    for cluster_id in cluster_ids:
        keywords = generate_keyword_summary(df, cluster_id)
        keyword_summaries.append({
            "cluster_id": cluster_id,
            "keywords": ", ".join(keywords[:10]),
        })

    keywords_df = pd.DataFrame(keyword_summaries)
    keywords_df.to_csv(output_dir / "cluster_keywords.csv", index=False)

    # LLM summaries (requires API key)
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Venezuela-US Reddit Discourse Analysis Pipeline"
    )

    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--sentiment", action="store_true", help="Run sentiment analysis")
    parser.add_argument("--topics", action="store_true", help="Run topic modeling")
    parser.add_argument("--clusters", action="store_true", help="Run clustering")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--summarize", action="store_true", help="Generate cluster summaries")
    parser.add_argument("--llm", default="anthropic", choices=["anthropic", "openai"],
                        help="LLM provider for summarization")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N documents for testing")

    args = parser.parse_args()

    # Default to all if no specific task
    if not any([args.all, args.sentiment, args.topics, args.clusters, args.visualize, args.summarize]):
        args.all = True

    # Initialize
    config = AnalysisConfig()
    config.ensure_directories()

    # Load data
    print("Loading data...")
    df = load_all_data(config)

    if args.sample:
        print(f"Sampling {args.sample:,} documents for testing...")
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)

    embeddings = None
    embeddings_2d = None

    # Run pipeline stages
    if args.all or args.sentiment:
        df = run_sentiment_analysis(df, config)

    if args.all or args.topics:
        df, topic_model, embeddings = run_topic_modeling(df, config)

    if args.all or args.clusters:
        df, embeddings, embeddings_2d = run_clustering(df, embeddings, config)

    if args.all or args.visualize:
        if embeddings_2d is None:
            # Load from saved
            embeddings_2d_path = config.output_dir / "clusters" / "embeddings_2d.npy"
            if embeddings_2d_path.exists():
                embeddings_2d = np.load(embeddings_2d_path)
            else:
                print("No 2D embeddings found. Run --clusters first.")
                return

        run_visualizations(df, embeddings_2d, config)

    if args.all or args.summarize:
        run_cluster_summarization(df, config, args.llm)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
