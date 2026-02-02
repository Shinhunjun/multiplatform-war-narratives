"""
Enhanced temporal visualizations for topics and clusters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path("outputs")
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all analysis results."""
    # Topic data
    topic_info = pd.read_csv(OUTPUT_DIR / "topics" / "topic_info.csv")
    topics_over_time = pd.read_csv(OUTPUT_DIR / "topics" / "topics_over_time.csv")
    topic_assignments = pd.read_parquet(OUTPUT_DIR / "topics" / "topic_assignments.parquet")

    # Cluster data
    cluster_summary = pd.read_csv(OUTPUT_DIR / "clusters" / "cluster_summary.csv")
    temporal_clusters = pd.read_csv(OUTPUT_DIR / "clusters" / "temporal_clusters.csv")
    cluster_assignments = pd.read_parquet(OUTPUT_DIR / "clusters" / "cluster_assignments.parquet")

    return {
        'topic_info': topic_info,
        'topics_over_time': topics_over_time,
        'topic_assignments': topic_assignments,
        'cluster_summary': cluster_summary,
        'temporal_clusters': temporal_clusters,
        'cluster_assignments': cluster_assignments,
    }


def plot_top_topics_over_time(data, top_n=15):
    """Plot top N topics over time as a stacked area chart."""
    topic_info = data['topic_info']
    topic_assignments = data['topic_assignments']

    # Get top N topics by count (excluding -1 which is outliers)
    top_topics = topic_info[topic_info['Topic'] != -1].nlargest(top_n, 'Count')
    top_topic_ids = top_topics['Topic'].tolist()

    # Create topic labels
    topic_labels = {}
    for _, row in top_topics.iterrows():
        # Extract first few keywords from Name
        name = row['Name'] if 'Name' in row else f"Topic {row['Topic']}"
        # Clean up the name - take first 30 chars
        short_name = name[:40] + "..." if len(name) > 40 else name
        topic_labels[row['Topic']] = short_name

    # Aggregate by month
    topic_assignments['year_month'] = pd.to_datetime(topic_assignments['year_month'])
    monthly = topic_assignments[topic_assignments['topic_id'].isin(top_topic_ids)].groupby(
        ['year_month', 'topic_id']
    ).size().unstack(fill_value=0)

    # Sort columns by total count
    col_order = monthly.sum().sort_values(ascending=False).index
    monthly = monthly[col_order]

    # Rename columns
    monthly.columns = [topic_labels.get(c, f"Topic {c}") for c in monthly.columns]

    # Plot stacked area
    fig, ax = plt.subplots(figsize=(18, 10))
    monthly.plot.area(ax=ax, alpha=0.8, linewidth=0.5)

    ax.set_title(f'Top {top_n} Topics Over Time (Monthly)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Documents', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'topics_over_time_stacked.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: topics_over_time_stacked.png")


def plot_topics_heatmap(data, top_n=20):
    """Plot topics as a heatmap over time."""
    topic_info = data['topic_info']
    topic_assignments = data['topic_assignments']

    # Get top N topics
    top_topics = topic_info[topic_info['Topic'] != -1].nlargest(top_n, 'Count')
    top_topic_ids = top_topics['Topic'].tolist()

    # Create topic labels
    topic_labels = {}
    for _, row in top_topics.iterrows():
        name = row['Name'] if 'Name' in row else f"Topic {row['Topic']}"
        short_name = name[:50] + "..." if len(name) > 50 else name
        topic_labels[row['Topic']] = short_name

    # Aggregate by month
    topic_assignments['year_month'] = pd.to_datetime(topic_assignments['year_month'])
    monthly = topic_assignments[topic_assignments['topic_id'].isin(top_topic_ids)].groupby(
        ['year_month', 'topic_id']
    ).size().unstack(fill_value=0)

    # Normalize by row (month) to show proportions
    monthly_norm = monthly.div(monthly.sum(axis=1), axis=0)

    # Rename columns
    monthly_norm.columns = [topic_labels.get(c, f"Topic {c}") for c in monthly_norm.columns]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(20, 12))

    im = ax.imshow(monthly_norm.T.values, aspect='auto', cmap='YlOrRd')

    # Set ticks
    ax.set_yticks(range(len(monthly_norm.columns)))
    ax.set_yticklabels(monthly_norm.columns, fontsize=9)

    # X-axis: show every 12th month (yearly)
    n_months = len(monthly_norm.index)
    tick_positions = list(range(0, n_months, 12))
    tick_labels = [monthly_norm.index[i].strftime('%Y-%m') for i in tick_positions if i < n_months]
    ax.set_xticks(tick_positions[:len(tick_labels)])
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    ax.set_title(f'Top {top_n} Topics Distribution Over Time (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Topic', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Proportion', fontsize=11)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'topics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: topics_heatmap.png")


def plot_top_clusters_over_time(data, top_n=20):
    """Plot top N clusters over time."""
    cluster_summary = data['cluster_summary']
    cluster_assignments = data['cluster_assignments']

    # Get top N clusters by count (excluding -1)
    top_clusters = cluster_summary[cluster_summary['cluster_id'] != -1].nlargest(top_n, 'count')
    top_cluster_ids = top_clusters['cluster_id'].tolist()

    # Create labels from top subreddit
    cluster_labels = {}
    for _, row in top_clusters.iterrows():
        label = f"C{row['cluster_id']} (r/{row['top_subreddit']}, n={row['count']:,})"
        cluster_labels[row['cluster_id']] = label

    # Aggregate by month
    cluster_assignments['year_month'] = pd.to_datetime(cluster_assignments['year_month'])
    monthly = cluster_assignments[cluster_assignments['cluster_id'].isin(top_cluster_ids)].groupby(
        ['year_month', 'cluster_id']
    ).size().unstack(fill_value=0)

    # Rename columns
    monthly.columns = [cluster_labels.get(c, f"Cluster {c}") for c in monthly.columns]

    # Plot
    fig, ax = plt.subplots(figsize=(18, 10))
    monthly.plot.area(ax=ax, alpha=0.8, linewidth=0.5)

    ax.set_title(f'Top {top_n} Clusters Over Time (Monthly)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Documents', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'clusters_over_time_stacked.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: clusters_over_time_stacked.png")


def plot_clusters_heatmap(data, top_n=30):
    """Plot clusters as a heatmap over time."""
    cluster_summary = data['cluster_summary']
    cluster_assignments = data['cluster_assignments']

    # Get top N clusters
    top_clusters = cluster_summary[cluster_summary['cluster_id'] != -1].nlargest(top_n, 'count')
    top_cluster_ids = top_clusters['cluster_id'].tolist()

    # Create labels
    cluster_labels = {}
    for _, row in top_clusters.iterrows():
        label = f"C{row['cluster_id']}: r/{row['top_subreddit']}"
        cluster_labels[row['cluster_id']] = label

    # Aggregate by month
    cluster_assignments['year_month'] = pd.to_datetime(cluster_assignments['year_month'])
    monthly = cluster_assignments[cluster_assignments['cluster_id'].isin(top_cluster_ids)].groupby(
        ['year_month', 'cluster_id']
    ).size().unstack(fill_value=0)

    # Normalize
    monthly_norm = monthly.div(monthly.sum(axis=1), axis=0)
    monthly_norm.columns = [cluster_labels.get(c, f"Cluster {c}") for c in monthly_norm.columns]

    # Plot
    fig, ax = plt.subplots(figsize=(20, 14))

    im = ax.imshow(monthly_norm.T.values, aspect='auto', cmap='viridis')

    ax.set_yticks(range(len(monthly_norm.columns)))
    ax.set_yticklabels(monthly_norm.columns, fontsize=8)

    n_months = len(monthly_norm.index)
    tick_positions = list(range(0, n_months, 12))
    tick_labels = [monthly_norm.index[i].strftime('%Y-%m') for i in tick_positions if i < n_months]
    ax.set_xticks(tick_positions[:len(tick_labels)])
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    ax.set_title(f'Top {top_n} Clusters Distribution Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Proportion', fontsize=11)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'clusters_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: clusters_heatmap.png")


def plot_topic_trends(data, top_n=10):
    """Plot individual topic trends as line charts."""
    topic_info = data['topic_info']
    topic_assignments = data['topic_assignments']

    top_topics = topic_info[topic_info['Topic'] != -1].nlargest(top_n, 'Count')

    topic_assignments['year_month'] = pd.to_datetime(topic_assignments['year_month'])

    fig, axes = plt.subplots(5, 2, figsize=(18, 20))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_topics.iterrows()):
        if idx >= 10:
            break

        topic_id = row['Topic']
        topic_name = row['Name'][:60] + "..." if len(row['Name']) > 60 else row['Name']

        # Get monthly counts for this topic
        topic_data = topic_assignments[topic_assignments['topic_id'] == topic_id]
        monthly = topic_data.groupby('year_month').size()

        ax = axes[idx]
        ax.fill_between(monthly.index, monthly.values, alpha=0.3)
        ax.plot(monthly.index, monthly.values, linewidth=2)
        ax.set_title(f"Topic {topic_id}: {topic_name}", fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Count')
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Add total count annotation
        ax.annotate(f"Total: {row['Count']:,}", xy=(0.98, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Top 10 Topics - Individual Trends', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'topic_individual_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: topic_individual_trends.png")


def plot_cluster_by_subreddit(data, top_n=15):
    """Plot cluster distribution by subreddit."""
    cluster_assignments = data['cluster_assignments']
    cluster_summary = data['cluster_summary']

    # Get top clusters
    top_clusters = cluster_summary[cluster_summary['cluster_id'] != -1].nlargest(top_n, 'count')
    top_cluster_ids = top_clusters['cluster_id'].tolist()

    # Filter and pivot
    filtered = cluster_assignments[cluster_assignments['cluster_id'].isin(top_cluster_ids)]
    pivot = pd.crosstab(filtered['subreddit'], filtered['cluster_id'], normalize='index')

    # Rename columns
    cluster_labels = {row['cluster_id']: f"C{row['cluster_id']}" for _, row in top_clusters.iterrows()}
    pivot.columns = [cluster_labels.get(c, f"C{c}") for c in pivot.columns]

    # Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    pivot.plot(kind='barh', stacked=True, ax=ax, colormap='tab20')

    ax.set_title('Cluster Distribution by Subreddit (Top 15 Clusters)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Proportion', fontsize=12)
    ax.set_ylabel('Subreddit', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'clusters_by_subreddit.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: clusters_by_subreddit.png")


def create_topic_summary_table(data):
    """Create a summary table of top topics with keywords."""
    topic_info = data['topic_info']

    # Top 30 topics
    top_topics = topic_info[topic_info['Topic'] != -1].nlargest(30, 'Count')

    summary = []
    for _, row in top_topics.iterrows():
        summary.append({
            'Topic ID': row['Topic'],
            'Count': row['Count'],
            'Keywords/Name': row['Name'][:80] + "..." if len(row['Name']) > 80 else row['Name']
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(VIZ_DIR / 'topic_summary_table.csv', index=False)
    print(f"Saved: topic_summary_table.csv")

    return summary_df


def create_cluster_summary_table(data):
    """Create a summary table of top clusters."""
    cluster_summary = data['cluster_summary']

    # Top 30 clusters
    top_clusters = cluster_summary[cluster_summary['cluster_id'] != -1].nlargest(30, 'count')

    # Load keywords
    try:
        keywords_df = pd.read_csv(OUTPUT_DIR / "clusters" / "cluster_keywords.csv")
        keywords_dict = dict(zip(keywords_df['cluster_id'], keywords_df['keywords']))
    except:
        keywords_dict = {}

    summary = []
    for _, row in top_clusters.iterrows():
        keywords = keywords_dict.get(row['cluster_id'], '')
        summary.append({
            'Cluster ID': row['cluster_id'],
            'Count': row['count'],
            'Top Subreddit': row['top_subreddit'],
            'Keywords': keywords[:80] + "..." if len(keywords) > 80 else keywords
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(VIZ_DIR / 'cluster_summary_table.csv', index=False)
    print(f"Saved: cluster_summary_table.csv")

    return summary_df


def main():
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

    print("6. Clusters by subreddit...")
    plot_cluster_by_subreddit(data, top_n=15)

    print("\n7. Creating summary tables...")
    topic_summary = create_topic_summary_table(data)
    cluster_summary = create_cluster_summary_table(data)

    print("\n" + "="*60)
    print("TOP 10 TOPICS:")
    print("="*60)
    print(topic_summary.head(10).to_string(index=False))

    print("\n" + "="*60)
    print("TOP 10 CLUSTERS:")
    print("="*60)
    print(cluster_summary.head(10).to_string(index=False))

    print(f"\n\nAll visualizations saved to: {VIZ_DIR}")


if __name__ == "__main__":
    main()
