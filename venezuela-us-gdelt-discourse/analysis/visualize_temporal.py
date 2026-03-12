"""
Enhanced temporal visualizations for topics and clusters.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path(__file__).parent / "outputs"
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> dict[str, pd.DataFrame]:
    """Load all analysis results."""
    topic_info = pd.read_csv(OUTPUT_DIR / "topics" / "topic_info.csv")
    topics_over_time = pd.read_csv(OUTPUT_DIR / "topics" / "topics_over_time.csv")
    topic_assignments = pd.read_parquet(OUTPUT_DIR / "topics" / "topic_assignments.parquet")

    cluster_summary = pd.read_csv(OUTPUT_DIR / "clusters" / "cluster_summary.csv")
    temporal_clusters = pd.read_csv(OUTPUT_DIR / "clusters" / "temporal_clusters.csv")
    cluster_assignments = pd.read_parquet(OUTPUT_DIR / "clusters" / "cluster_assignments.parquet")

    return {
        "topic_info": topic_info,
        "topics_over_time": topics_over_time,
        "topic_assignments": topic_assignments,
        "cluster_summary": cluster_summary,
        "temporal_clusters": temporal_clusters,
        "cluster_assignments": cluster_assignments,
    }


def plot_top_topics_over_time(data: dict[str, pd.DataFrame], top_n: int = 15) -> None:
    """Plot top N topics over time as a stacked area chart."""
    topic_info = data["topic_info"]
    topic_assignments = data["topic_assignments"].copy()

    top_topics = topic_info[topic_info["Topic"] != -1].nlargest(top_n, "Count")
    top_topic_ids = top_topics["Topic"].tolist()

    topic_labels = {}
    for _, row in top_topics.iterrows():
        name = row["Name"] if "Name" in row else f"Topic {row['Topic']}"
        topic_labels[row["Topic"]] = name[:40] + "..." if len(name) > 40 else name

    topic_assignments["year_month"] = pd.to_datetime(topic_assignments["year_month"])
    monthly = (
        topic_assignments[topic_assignments["topic_id"].isin(top_topic_ids)]
        .groupby(["year_month", "topic_id"])
        .size()
        .unstack(fill_value=0)
    )

    col_order = monthly.sum().sort_values(ascending=False).index
    monthly = monthly[col_order]
    monthly.columns = [topic_labels.get(c, f"Topic {c}") for c in monthly.columns]

    fig, ax = plt.subplots(figsize=(18, 10))
    monthly.plot.area(ax=ax, alpha=0.8, linewidth=0.5)

    ax.set_title(f"Top {top_n} Topics Over Time (Monthly)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Documents", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "topics_over_time_stacked.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: topics_over_time_stacked.png")


def plot_topics_heatmap(data: dict[str, pd.DataFrame], top_n: int = 20) -> None:
    """Plot topics as a heatmap over time."""
    topic_info = data["topic_info"]
    topic_assignments = data["topic_assignments"].copy()

    top_topics = topic_info[topic_info["Topic"] != -1].nlargest(top_n, "Count")
    top_topic_ids = top_topics["Topic"].tolist()

    topic_labels = {}
    for _, row in top_topics.iterrows():
        name = row["Name"] if "Name" in row else f"Topic {row['Topic']}"
        topic_labels[row["Topic"]] = name[:50] + "..." if len(name) > 50 else name

    topic_assignments["year_month"] = pd.to_datetime(topic_assignments["year_month"])
    monthly = (
        topic_assignments[topic_assignments["topic_id"].isin(top_topic_ids)]
        .groupby(["year_month", "topic_id"])
        .size()
        .unstack(fill_value=0)
    )

    monthly_norm = monthly.div(monthly.sum(axis=1), axis=0)
    monthly_norm.columns = [topic_labels.get(c, f"Topic {c}") for c in monthly_norm.columns]

    fig, ax = plt.subplots(figsize=(20, 12))
    im = ax.imshow(monthly_norm.T.values, aspect="auto", cmap="YlOrRd")

    ax.set_yticks(range(len(monthly_norm.columns)))
    ax.set_yticklabels(monthly_norm.columns, fontsize=9)

    n_months = len(monthly_norm.index)
    tick_positions = list(range(0, n_months, 12))
    tick_labels = [monthly_norm.index[i].strftime("%Y-%m") for i in tick_positions if i < n_months]
    ax.set_xticks(tick_positions[: len(tick_labels)])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_title(f"Top {top_n} Topics Distribution Over Time (Normalized)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Topic", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Proportion", fontsize=11)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "topics_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: topics_heatmap.png")


def plot_top_clusters_over_time(data: dict[str, pd.DataFrame], top_n: int = 20) -> None:
    """Plot top N clusters over time."""
    cluster_summary = data["cluster_summary"]
    cluster_assignments = data["cluster_assignments"].copy()

    top_clusters = cluster_summary[cluster_summary["cluster_id"] != -1].nlargest(top_n, "count")
    top_cluster_ids = top_clusters["cluster_id"].tolist()

    cluster_labels = {}
    for _, row in top_clusters.iterrows():
        top_group = row.get("top_group", "unknown")
        cluster_labels[row["cluster_id"]] = f"C{row['cluster_id']} ({top_group}, n={row['count']:,})"

    cluster_assignments["year_month"] = pd.to_datetime(cluster_assignments["year_month"])
    monthly = (
        cluster_assignments[cluster_assignments["cluster_id"].isin(top_cluster_ids)]
        .groupby(["year_month", "cluster_id"])
        .size()
        .unstack(fill_value=0)
    )
    monthly.columns = [cluster_labels.get(c, f"Cluster {c}") for c in monthly.columns]

    fig, ax = plt.subplots(figsize=(18, 10))
    monthly.plot.area(ax=ax, alpha=0.8, linewidth=0.5)

    ax.set_title(f"Top {top_n} Clusters Over Time (Monthly)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Documents", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "clusters_over_time_stacked.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: clusters_over_time_stacked.png")


def plot_clusters_heatmap(data: dict[str, pd.DataFrame], top_n: int = 30) -> None:
    """Plot clusters as a heatmap over time."""
    cluster_summary = data["cluster_summary"]
    cluster_assignments = data["cluster_assignments"].copy()

    top_clusters = cluster_summary[cluster_summary["cluster_id"] != -1].nlargest(top_n, "count")
    top_cluster_ids = top_clusters["cluster_id"].tolist()

    cluster_labels = {}
    for _, row in top_clusters.iterrows():
        cluster_labels[row["cluster_id"]] = f"C{row['cluster_id']}: {row.get('top_group', 'unknown')}"

    cluster_assignments["year_month"] = pd.to_datetime(cluster_assignments["year_month"])
    monthly = (
        cluster_assignments[cluster_assignments["cluster_id"].isin(top_cluster_ids)]
        .groupby(["year_month", "cluster_id"])
        .size()
        .unstack(fill_value=0)
    )
    monthly_norm = monthly.div(monthly.sum(axis=1), axis=0)
    monthly_norm.columns = [cluster_labels.get(c, f"Cluster {c}") for c in monthly_norm.columns]

    fig, ax = plt.subplots(figsize=(20, 14))
    im = ax.imshow(monthly_norm.T.values, aspect="auto", cmap="viridis")

    ax.set_yticks(range(len(monthly_norm.columns)))
    ax.set_yticklabels(monthly_norm.columns, fontsize=8)

    n_months = len(monthly_norm.index)
    tick_positions = list(range(0, n_months, 12))
    tick_labels = [monthly_norm.index[i].strftime("%Y-%m") for i in tick_positions if i < n_months]
    ax.set_xticks(tick_positions[: len(tick_labels)])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_title(f"Top {top_n} Clusters Distribution Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Proportion", fontsize=11)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "clusters_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: clusters_heatmap.png")


def plot_topic_trends(data: dict[str, pd.DataFrame], top_n: int = 10) -> None:
    """Plot individual topic trends as line charts."""
    topic_info = data["topic_info"]
    topic_assignments = data["topic_assignments"].copy()

    top_topics = topic_info[topic_info["Topic"] != -1].nlargest(top_n, "Count")
    topic_assignments["year_month"] = pd.to_datetime(topic_assignments["year_month"])

    fig, axes = plt.subplots(5, 2, figsize=(18, 20))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_topics.iterrows()):
        if idx >= 10:
            break

        topic_id = row["Topic"]
        topic_name = row["Name"][:60] + "..." if len(row["Name"]) > 60 else row["Name"]

        topic_data = topic_assignments[topic_assignments["topic_id"] == topic_id]
        monthly = topic_data.groupby("year_month").size()

        ax = axes[idx]
        ax.fill_between(monthly.index, monthly.values, alpha=0.3)
        ax.plot(monthly.index, monthly.values, linewidth=2)
        ax.set_title(f"Topic {topic_id}: {topic_name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.annotate(
            f"Total: {row['Count']:,}",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle("Top 10 Topics - Individual Trends", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "topic_individual_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: topic_individual_trends.png")


def plot_cluster_by_source_domain(data: dict[str, pd.DataFrame], top_n: int = 15) -> None:
    """Plot cluster distribution by source domain."""
    cluster_assignments = data["cluster_assignments"]
    cluster_summary = data["cluster_summary"]

    if "source_domain" not in cluster_assignments.columns:
        print("Skipping source-domain chart; source_domain column is missing.")
        return

    top_clusters = cluster_summary[cluster_summary["cluster_id"] != -1].nlargest(top_n, "count")
    top_cluster_ids = top_clusters["cluster_id"].tolist()

    filtered = cluster_assignments[cluster_assignments["cluster_id"].isin(top_cluster_ids)]
    pivot = pd.crosstab(filtered["source_domain"], filtered["cluster_id"], normalize="index")

    cluster_labels = {row["cluster_id"]: f"C{row['cluster_id']}" for _, row in top_clusters.iterrows()}
    pivot.columns = [cluster_labels.get(c, f"C{c}") for c in pivot.columns]
    pivot = pivot.sort_values(pivot.columns.tolist(), ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(16, 10))
    pivot.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")

    ax.set_title("Cluster Distribution by Source Domain (Top 15 Clusters)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Proportion", fontsize=12)
    ax.set_ylabel("Source Domain", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "clusters_by_source_domain.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: clusters_by_source_domain.png")


def create_topic_summary_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a summary table of top topics with keywords."""
    topic_info = data["topic_info"]
    top_topics = topic_info[topic_info["Topic"] != -1].nlargest(30, "Count")

    summary = []
    for _, row in top_topics.iterrows():
        summary.append(
            {
                "Topic ID": row["Topic"],
                "Count": row["Count"],
                "Keywords/Name": row["Name"][:80] + "..." if len(row["Name"]) > 80 else row["Name"],
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(VIZ_DIR / "topic_summary_table.csv", index=False)
    print("Saved: topic_summary_table.csv")
    return summary_df


def create_cluster_summary_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a summary table of top clusters."""
    cluster_summary = data["cluster_summary"]
    top_clusters = cluster_summary[cluster_summary["cluster_id"] != -1].nlargest(30, "count")

    try:
        keywords_df = pd.read_csv(OUTPUT_DIR / "clusters" / "cluster_keywords.csv")
        keywords_dict = dict(zip(keywords_df["cluster_id"], keywords_df["keywords"]))
    except Exception:
        keywords_dict = {}

    summary = []
    for _, row in top_clusters.iterrows():
        keywords = keywords_dict.get(row["cluster_id"], "")
        summary.append(
            {
                "Cluster ID": row["cluster_id"],
                "Count": row["count"],
                "Top Source Domain": row.get("top_group", ""),
                "Keywords": keywords[:80] + "..." if len(keywords) > 80 else keywords,
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(VIZ_DIR / "cluster_summary_table.csv", index=False)
    print("Saved: cluster_summary_table.csv")
    return summary_df


def main() -> None:
    """Run the script entry point."""
    print("Loading data...")
    data = load_data()

    print("\nGenerating enhanced visualizations...")

    print("\n1. Topics over time (stacked area)...")
    plot_top_topics_over_time(data, top_n=15)

    print("2. Topics heatmap...")
    plot_topics_heatmap(data, top_n=20)

    print("3. Individual topic trends...")
    plot_topic_trends(data, top_n=10)

    print("4. Clusters over time (stacked area)...")
    plot_top_clusters_over_time(data, top_n=20)

    print("5. Clusters heatmap...")
    plot_clusters_heatmap(data, top_n=30)

    print("6. Clusters by source domain...")
    plot_cluster_by_source_domain(data, top_n=15)

    print("\n7. Creating summary tables...")
    topic_summary = create_topic_summary_table(data)
    cluster_summary = create_cluster_summary_table(data)

    print("\n" + "=" * 60)
    print("TOP 10 TOPICS:")
    print("=" * 60)
    print(topic_summary.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP 10 CLUSTERS:")
    print("=" * 60)
    print(cluster_summary.head(10).to_string(index=False))

    print(f"\nAll visualizations saved to: {VIZ_DIR}")


if __name__ == "__main__":
    main()
