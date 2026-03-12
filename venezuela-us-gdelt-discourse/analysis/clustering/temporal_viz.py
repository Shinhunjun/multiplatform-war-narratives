"""
Temporal cluster visualization - animated and static views.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd


def create_umap_scatter(
    embeddings_2d: np.ndarray,
    df: pd.DataFrame,
    color_column: str = "cluster_id",
    output_path: Optional[Path] = None,
    title: str = "Document Clusters",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Create static UMAP scatter plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    unique_values = df[color_column].dropna().unique()
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_values))))
    color_map = dict(zip(unique_values, colors))

    for value in unique_values:
        mask = df[color_column] == value
        if color_column == "cluster_id":
            label = f"Cluster {value}" if value != -1 else "Noise"
            alpha = 0.3 if value == -1 else 0.6
        else:
            label = str(value)
            alpha = 0.55

        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color_map[value]],
            label=label,
            alpha=alpha,
            s=10,
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2, fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    plt.close()


def create_animated_umap(
    embeddings_2d: np.ndarray,
    df: pd.DataFrame,
    time_column: str = "year_month",
    color_column: str = "cluster_id",
    output_path: Optional[Path] = None,
    fps: int = 2,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Create animated UMAP showing cluster evolution over time.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    periods = sorted(df[time_column].dropna().unique())
    fig, ax = plt.subplots(figsize=figsize)

    x_min, x_max = embeddings_2d[:, 0].min() - 1, embeddings_2d[:, 0].max() + 1
    y_min, y_max = embeddings_2d[:, 1].min() - 1, embeddings_2d[:, 1].max() + 1

    unique_clusters = df[color_column].dropna().unique()
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_clusters))))
    color_map = dict(zip(unique_clusters, colors))

    def update(frame: int) -> None:
        """Execute update."""
        ax.clear()
        period = periods[frame]

        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c="lightgray",
            alpha=0.08,
            s=5,
        )

        mask = df[time_column] == period
        period_data = df[mask]
        period_embeddings = embeddings_2d[mask]

        for cluster_id in period_data[color_column].dropna().unique():
            cluster_mask = period_data[color_column] == cluster_id
            color = color_map.get(cluster_id, "gray")
            alpha = 0.3 if (color_column == "cluster_id" and cluster_id == -1) else 0.8

            ax.scatter(
                period_embeddings[cluster_mask, 0],
                period_embeddings[cluster_mask, 1],
                c=[color],
                alpha=alpha,
                s=20,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Cluster Evolution - {period}", fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        n_points = int(mask.sum())
        ax.annotate(
            f"n = {n_points:,}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
        )
        return None

    anim = FuncAnimation(
        fig,
        update,
        frames=len(periods),
        interval=1000 // fps,
        blit=False,
    )

    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == ".gif":
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)
        else:
            anim.save(output_path, fps=fps)
        print(f"Saved animation to {output_path}")

    plt.close()


def create_sankey_diagram(
    evolution_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Cluster Flow Over Time",
) -> Any:
    """
    Create Sankey diagram showing cluster flow between periods.
    """
    import plotly.graph_objects as go

    periods = sorted(set(evolution_df["from_period"]) | set(evolution_df["to_period"]))
    nodes = []
    node_labels = []

    for period in periods:
        clusters_in_period = set(
            evolution_df[evolution_df["from_period"] == period]["from_cluster"].tolist()
            + evolution_df[evolution_df["to_period"] == period]["to_cluster"].tolist()
        )
        for cluster in sorted(clusters_in_period):
            node_id = f"{period}_{cluster}"
            nodes.append(node_id)
            node_labels.append(f"{period}\nC{cluster}")

    node_map = {n: i for i, n in enumerate(nodes)}
    sources = []
    targets = []
    values = []

    for _, row in evolution_df.iterrows():
        src_id = f"{row['from_period']}_{row['from_cluster']}"
        tgt_id = f"{row['to_period']}_{row['to_cluster']}"
        if src_id in node_map and tgt_id in node_map:
            sources.append(node_map[src_id])
            targets.append(node_map[tgt_id])
            values.append(row.get("from_count", row.get("similarity", 1)) * 10)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                ),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title_text=title, font_size=10, height=600)

    if output_path:
        fig.write_html(output_path)
        print(f"Saved Sankey diagram to {output_path}")
    return fig


def create_cluster_river_plot(
    df: pd.DataFrame,
    time_column: str = "year_month",
    output_path: Optional[Path] = None,
    top_n_clusters: int = 10,
    title: str = "Cluster Sizes Over Time",
) -> None:
    """
    Create river/stream plot showing cluster sizes over time.
    """
    import matplotlib.pyplot as plt

    temporal = df[df["cluster_id"] != -1].groupby([time_column, "cluster_id"]).size().unstack(fill_value=0)
    if temporal.empty:
        print("No non-noise clusters available for river plot.")
        return

    cluster_totals = temporal.sum().sort_values(ascending=False)
    top_clusters = cluster_totals.head(top_n_clusters).index.tolist()
    temporal = temporal[top_clusters]

    fig, ax = plt.subplots(figsize=(14, 6))
    periods = temporal.index.tolist()
    x = range(len(periods))

    ax.stackplot(
        x,
        temporal.T.values,
        labels=[f"Cluster {c}" for c in top_clusters],
        alpha=0.8,
    )

    tick_step = max(1, len(periods) // 20)
    tick_idx = list(range(0, len(periods), tick_step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([periods[i] for i in tick_idx], rotation=45, ha="right")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Number of Documents")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved river plot to {output_path}")
    plt.close()


def create_cluster_heatmap(
    df: pd.DataFrame,
    group_column: str = "source_domain",
    output_path: Optional[Path] = None,
    title: str = "Cluster Distribution by Source Domain",
) -> None:
    """
    Create heatmap showing cluster distribution by a group column.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if group_column not in df.columns:
        raise ValueError(f"Group column not found: {group_column}")

    pivot = df[df["cluster_id"] != -1].groupby([group_column, "cluster_id"]).size().unstack(fill_value=0)
    if pivot.empty:
        print("No non-noise clusters available for heatmap.")
        return

    pivot = pivot.sort_values(pivot.columns.tolist(), ascending=False).head(40)
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        pivot_norm,
        cmap="YlOrRd",
        ax=ax,
        annot=False,
        cbar_kws={"label": "Proportion"},
    )

    ax.set_title(title)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel(group_column.replace("_", " ").title())

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")
    plt.close()


def create_interactive_scatter(
    embeddings_2d: np.ndarray,
    df: pd.DataFrame,
    color_column: str = "cluster_id",
    hover_columns: List[str] = ["text", "source_domain", "year_month", "event_category"],
    output_path: Optional[Path] = None,
    title: str = "Interactive Cluster Visualization",
) -> Any:
    """
    Create interactive Plotly scatter plot with hover information.
    """
    import plotly.express as px

    plot_df = df.copy()
    plot_df["umap_1"] = embeddings_2d[:, 0]
    plot_df["umap_2"] = embeddings_2d[:, 1]

    if "text" in hover_columns and "text" in plot_df.columns:
        plot_df["text_preview"] = plot_df["text"].fillna("").astype(str).str[:200] + "..."

    hover_data = []
    if "text" in hover_columns and "text_preview" in plot_df.columns:
        hover_data.append("text_preview")
    for col in ["source_domain", "year_month", "event_category", "id"]:
        if col in hover_columns and col in plot_df.columns:
            hover_data.append(col)

    fig = px.scatter(
        plot_df,
        x="umap_1",
        y="umap_2",
        color=color_column,
        hover_data=hover_data,
        title=title,
        opacity=0.6,
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        height=700,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved interactive plot to {output_path}")
    return fig
