"""
Visualization functions for data analysis and reporting.
Matches the Google Colab visualization outputs.
"""

import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from .config import (
    CRISIS_COLORS,
    CRISIS_PERIODS,
    FLASHPOINT_WINDOWS,
    PipelineConfig,
    SUBREDDIT_COLORS,
    TOPIC_CATEGORIES,
    TOPIC_COLORS,
)
from .processors import get_stopwords, identify_crisis_posts, preprocess_text

warnings.filterwarnings("ignore")


# =============================================================================
# VISUALIZATION 1 - Temporal Overview
# =============================================================================
def plot_temporal_overview(
    df: pd.DataFrame,
    config: PipelineConfig,
    save: bool = True,
) -> plt.Figure:
    """Create comprehensive temporal overview with improved colors and legend."""
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Local crisis periods for visualization
    CRISIS_PERIODS_LOCAL = {
        "Pulwama Attack": ("2019-02-14", "2019-02-28"),
        "Balakot Airstrike": ("2019-02-26", "2019-03-10"),
        "Article 370": ("2019-08-05", "2019-08-31"),
        "Pahalgam Attack": ("2025-04-22", "2025-04-30"),
        "Operation Sindoor": ("2025-05-07", "2025-05-10"),
    }

    # Panel A: Monthly time series with crisis annotations
    ax1 = fig.add_subplot(gs[0, :])
    monthly = df.groupby(df["created_datetime"].dt.to_period("M")).size()
    monthly.index = monthly.index.to_timestamp()

    ax1.fill_between(monthly.index, monthly.values, alpha=0.3, color="#1f77b4")
    ax1.plot(monthly.index, monthly.values, linewidth=2, color="#1f77b4")
    ax1.axhline(monthly.mean(), color="#d62728", linestyle="--", linewidth=1.5,
                label=f"Mean: {monthly.mean():.0f}")

    legend_handles = [plt.Line2D([0], [0], color="#d62728", linestyle="--",
                                 label=f"Mean: {monthly.mean():.0f}")]
    for crisis_name, (start, end) in CRISIS_PERIODS_LOCAL.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        color = CRISIS_COLORS.get(crisis_name, "#888888")
        if start_dt >= monthly.index.min() and start_dt <= monthly.index.max():
            ax1.axvspan(start_dt, end_dt, alpha=0.4, color=color, zorder=1)
            legend_handles.append(
                Patch(facecolor=color, alpha=0.4, label=crisis_name))

    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Number of Posts", fontsize=11)
    ax1.set_title("A. Monthly Posting Volume", fontweight="bold", fontsize=13)
    ax1.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax1.set_xlim(monthly.index.min(), monthly.index.max())

    # Panel B: Year-over-year comparison
    ax2 = fig.add_subplot(gs[1, 0])
    yearly = df.groupby("year").size()
    colors_yearly = plt.cm.Blues(np.linspace(0.4, 0.9, len(yearly)))
    bars = ax2.bar(yearly.index.astype(str), yearly.values, color=colors_yearly,
                   edgecolor="#2c3e50", linewidth=1.2)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Number of Posts", fontsize=11)
    ax2.set_title("B. Annual Distribution", fontweight="bold", fontsize=13)
    for bar, val in zip(bars, yearly.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
                 f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, yearly.max() * 1.15)
    peak_idx = yearly.values.argmax()
    bars[peak_idx].set_edgecolor("#d62728")
    bars[peak_idx].set_linewidth(3)
    ax2.legend([Patch(facecolor=colors_yearly[len(yearly) // 2], edgecolor="#2c3e50"),
                Patch(facecolor=colors_yearly[peak_idx], edgecolor="#d62728", linewidth=2)],
               ["Posts per Year", f"Peak: {yearly.index[peak_idx]}"],
               loc="upper right", fontsize=9)

    # Panel C: Monthly seasonality
    ax3 = fig.add_subplot(gs[1, 1])
    monthly_pattern = df.groupby("month").size()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    season_colors_map = {
        "Winter": "#3498db", "Spring": "#2ecc71",
        "Summer": "#e74c3c", "Fall": "#f39c12"
    }
    month_to_season = {
        1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall",
        11: "Fall", 12: "Winter"
    }
    colors_monthly = [season_colors_map[month_to_season[m]]
                      for m in range(1, 13)]
    bars = ax3.bar(range(1, 13), monthly_pattern.reindex(range(1, 13), fill_value=0),
                   color=colors_monthly, edgecolor="#2c3e50", linewidth=1)
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(month_names, rotation=45, ha="right")
    ax3.set_xlabel("Month", fontsize=11)
    ax3.set_ylabel("Number of Posts", fontsize=11)
    ax3.set_title("C. Monthly Seasonality", fontweight="bold", fontsize=13)
    ax3.legend([Patch(color=season_colors_map[s]) for s in ["Winter", "Spring", "Summer", "Fall"]],
               ["Winter (Dec-Feb)", "Spring (Mar-May)",
                "Summer (Jun-Aug)", "Fall (Sep-Nov)"],
               loc="upper right", fontsize=8)

    # Panel D: Day of week pattern
    ax4 = fig.add_subplot(gs[2, 0])
    dow_pattern = df.groupby("dayofweek").size()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors_dow = ["#3498db"] * 5 + ["#e74c3c"] * 2
    bars = ax4.bar(range(7), dow_pattern.reindex(range(7), fill_value=0),
                   color=colors_dow, edgecolor="#2c3e50", linewidth=1.2)
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(day_names)
    ax4.set_xlabel("Day of Week", fontsize=11)
    ax4.set_ylabel("Number of Posts", fontsize=11)
    ax4.set_title("D. Day of Week Distribution",
                  fontweight="bold", fontsize=13)
    ax4.legend([Patch(color="#3498db"), Patch(color="#e74c3c")],
               ["Weekday", "Weekend"], loc="upper left", fontsize=9)

    # Panel E: Hourly pattern
    ax5 = fig.add_subplot(gs[2, 1])
    hourly_pattern = df.groupby("hour").size()
    hours = range(24)
    values = hourly_pattern.reindex(hours, fill_value=0)
    ax5.fill_between(hours, values, alpha=0.4, color="#27ae60")
    ax5.plot(hours, values, linewidth=2.5,
             color="#1e8449", marker="o", markersize=4)
    peak_hour = values.idxmax()
    ax5.axvline(peak_hour, color="#e74c3c", linestyle="--", alpha=0.7,
                label=f"Peak: {peak_hour}:00 UTC")
    ax5.set_xticks(range(0, 24, 3))
    ax5.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax5.set_xlabel("Hour (UTC)", fontsize=11)
    ax5.set_ylabel("Number of Posts", fontsize=11)
    ax5.set_title("E. Hourly Activity Pattern", fontweight="bold", fontsize=13)
    ax5.legend(loc="upper right", fontsize=9)
    ax5.annotate("Peak ~15:00 UTC = 20:30 IST", xy=(15, values.iloc[15]),
                 xytext=(18, values.max() * 0.8), fontsize=8, color="#666666",
                 arrowprops=dict(arrowstyle="->", color="#666666", lw=0.5))

    plt.suptitle("Temporal Analysis of India-Pakistan Conflict Discourse on Reddit (2019-2025)",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.subplots_adjust(right=0.85)

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.analysis_dir / "01_temporal_analysis.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {config.analysis_dir / '01_temporal_analysis.png'}")

    return fig


# =============================================================================
# VISUALIZATION 2 - Subreddit Analysis
# =============================================================================
def plot_subreddit_analysis(
    df: pd.DataFrame,
    config: PipelineConfig,
    save: bool = True,
) -> plt.Figure:
    """Create subreddit-level analysis visualization."""
    # Compute subreddit stats
    sub_stats = df.groupby("subreddit").agg({
        "_id": "count",
        "score": ["mean", "median", "sum"],
        "num_comments": ["mean", "sum"],
        "author": "nunique"
    }).round(1)
    sub_stats.columns = ["posts", "avg_score", "med_score", "total_score",
                         "avg_comments", "total_comments", "unique_authors"]
    sub_stats = sub_stats.sort_values("posts", ascending=False)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    subreddits = sub_stats.index.tolist()
    colors = [SUBREDDIT_COLORS.get(s, "#888888") for s in subreddits]

    # Panel A: Post count
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(
        subreddits, sub_stats["posts"], color=colors, edgecolor="black")
    ax1.set_xlabel("Number of Posts")
    ax1.set_title("A. Post Distribution by Subreddit", fontweight="bold")
    for bar, val in zip(bars, sub_stats["posts"]):
        ax1.text(val + 100, bar.get_y() + bar.get_height() / 2, f"{val:,.0f}",
                 va="center", fontsize=9)
    ax1.invert_yaxis()

    # Panel B: Average engagement
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(subreddits))
    width = 0.35
    ax2.bar(x - width / 2, sub_stats["avg_score"],
            width, label="Avg Score", color="steelblue")
    ax2.bar(x + width / 2, sub_stats["avg_comments"],
            width, label="Avg Comments", color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(subreddits, rotation=45, ha="right")
    ax2.set_ylabel("Count")
    ax2.set_title("B. Average Engagement by Subreddit", fontweight="bold")
    ax2.legend()

    # Panel C: Unique authors
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.barh(subreddits, sub_stats["unique_authors"],
             color=colors, edgecolor="black", alpha=0.7)
    ax3.set_xlabel("Unique Authors")
    ax3.set_title("C. Author Diversity by Subreddit", fontweight="bold")
    ax3.invert_yaxis()

    # Panel D: Time series by subreddit
    ax4 = fig.add_subplot(gs[1, 1])
    for sub in subreddits[:5]:
        sub_data = df[df["subreddit"] == sub]
        monthly = sub_data.groupby(
            sub_data["created_datetime"].dt.to_period("M")).size()
        monthly.index = monthly.index.to_timestamp()
        ax4.plot(monthly.index, monthly.values, label=sub,
                 color=SUBREDDIT_COLORS.get(sub, "#888888"), linewidth=1.5)
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Posts per Month")
    ax4.set_title("D. Monthly Activity (Top 5 Subreddits)", fontweight="bold")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle("Subreddit-Level Analysis of India-Pakistan Conflict Discourse",
                 fontsize=16, fontweight="bold", y=1.02)

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.analysis_dir / "02_subreddit_analysis.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {config.analysis_dir / '02_subreddit_analysis.png'}")

    return fig


# =============================================================================
# VISUALIZATION 3 - Engagement Analysis
# =============================================================================
def plot_engagement_analysis(
    df: pd.DataFrame,
    config: PipelineConfig,
    save: bool = True,
) -> plt.Figure:
    """Create engagement metrics visualization."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    scores = df["score"].clip(lower=1)
    ax1.hist(np.log10(scores), bins=50, color="steelblue",
             edgecolor="white", alpha=0.7)
    ax1.axvline(np.log10(df["score"].median()), color="red", linestyle="--",
                label=f'Median: {df["score"].median():.0f}')
    ax1.axvline(np.log10(df["score"].mean()), color="orange", linestyle="--",
                label=f'Mean: {df["score"].mean():.0f}')
    ax1.set_xlabel("Score (log10)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("A. Score Distribution", fontweight="bold")
    ax1.legend()

    # Panel B: Comments distribution
    ax2 = fig.add_subplot(gs[0, 1])
    comments = df["num_comments"].clip(lower=1)
    ax2.hist(np.log10(comments), bins=50, color="coral",
             edgecolor="white", alpha=0.7)
    ax2.axvline(np.log10(df["num_comments"].median()), color="red", linestyle="--",
                label=f'Median: {df["num_comments"].median():.0f}')
    ax2.set_xlabel("Comments (log10)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("B. Comments Distribution", fontweight="bold")
    ax2.legend()

    # Panel C: Score vs Comments
    ax3 = fig.add_subplot(gs[1, 0])
    sample = df.sample(min(5000, len(df)))
    colors_scatter = [SUBREDDIT_COLORS.get(
        s, "#888888") for s in sample["subreddit"]]
    ax3.scatter(sample["score"], sample["num_comments"],
                alpha=0.3, s=10, c=colors_scatter)
    ax3.set_xlabel("Score")
    ax3.set_ylabel("Number of Comments")
    ax3.set_title("C. Score vs Comments", fontweight="bold")
    ax3.set_xscale("log")
    ax3.set_yscale("log")

    # Panel D: Engagement percentiles
    ax4 = fig.add_subplot(gs[1, 1])
    percentiles = [50, 75, 90, 95, 99]
    score_pct = [df["score"].quantile(p / 100) for p in percentiles]
    comments_pct = [df["num_comments"].quantile(p / 100) for p in percentiles]
    x = np.arange(len(percentiles))
    width = 0.35
    ax4.bar(x - width / 2, score_pct, width, label="Score", color="steelblue")
    ax4.bar(x + width / 2, comments_pct, width,
            label="Comments", color="coral")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{p}%" for p in percentiles])
    ax4.set_xlabel("Percentile")
    ax4.set_ylabel("Value")
    ax4.set_title("D. Engagement Percentiles", fontweight="bold")
    ax4.legend()

    plt.suptitle("Engagement Analysis of India-Pakistan Conflict Discourse",
                 fontsize=16, fontweight="bold", y=1.02)

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.analysis_dir / "03_engagement_analysis.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {config.analysis_dir / '03_engagement_analysis.png'}")

    return fig


# =============================================================================
# VISUALIZATION 4A - Crisis Overview
# =============================================================================
def plot_crisis_overview(
    df: pd.DataFrame,
    config: PipelineConfig,
    save: bool = True,
) -> plt.Figure:
    """Create crisis period overview visualization."""
    df_crisis = identify_crisis_posts(df, CRISIS_PERIODS)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[
                  1, 1.2], hspace=0.35, wspace=0.3)

    # Panel A: Posts per crisis period
    ax1 = fig.add_subplot(gs[0, 0])
    crisis_counts = df_crisis[df_crisis["crisis_period"]
                              != "Normal"]["crisis_period"].value_counts()
    colors = [CRISIS_PERIODS.get(c, ("", "", "#888888"))[2]
              for c in crisis_counts.index]
    bars = ax1.barh(crisis_counts.index, crisis_counts.values, color=colors,
                    edgecolor="black", height=0.6)
    ax1.set_xlabel("Number of Posts", fontsize=11)
    ax1.set_title("A. Posts During Crisis Periods",
                  fontweight="bold", fontsize=13, pad=10)
    for bar, val in zip(bars, crisis_counts.values):
        ax1.text(val + 80, bar.get_y() + bar.get_height() / 2, f"{val:,}",
                 va="center", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, crisis_counts.max() * 1.25)

    # Panel B: Engagement comparison
    ax2 = fig.add_subplot(gs[0, 1])
    crisis_engagement = df_crisis.groupby("crisis_period").agg({
        "score": "mean", "num_comments": "mean"
    }).round(1)
    order = ["Normal"] + [c for c in crisis_engagement.index if c != "Normal"]
    crisis_engagement = crisis_engagement.reindex(order)
    x = np.arange(len(crisis_engagement))
    width = 0.35
    bars1 = ax2.bar(x - width / 2, crisis_engagement["score"], width, label="Avg Score",
                    color="#3498db", edgecolor="#2c3e50")
    bars2 = ax2.bar(x + width / 2, crisis_engagement["num_comments"], width, label="Avg Comments",
                    color="#e74c3c", edgecolor="#922b21")
    ax2.set_xticks(x)
    ax2.set_xticklabels(crisis_engagement.index,
                        rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Average Count", fontsize=11)
    ax2.set_title("B. Engagement: Crisis vs Normal Periods",
                  fontweight="bold", fontsize=13, pad=10)
    ax2.legend(loc="upper center", fontsize=10)
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{bar.get_height():.0f}",
                 ha="center", va="bottom", fontsize=8, color="#2c3e50", fontweight="bold")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{bar.get_height():.0f}",
                 ha="center", va="bottom", fontsize=8, color="#922b21", fontweight="bold")

    # Panel C: Subreddit distribution (full width bottom)
    ax3 = fig.add_subplot(gs[1, :])
    crisis_data = df_crisis[df_crisis["crisis_period"] != "Normal"]
    if len(crisis_data) > 0:
        crisis_sub = pd.crosstab(
            crisis_data["crisis_period"], crisis_data["subreddit"])
        crisis_sub_pct = crisis_sub.div(crisis_sub.sum(axis=1), axis=0) * 100
        sub_order = crisis_sub.sum().sort_values(ascending=False).index
        crisis_sub_pct = crisis_sub_pct[sub_order]
        crisis_sub_pct.plot(kind="bar", stacked=True, ax=ax3,
                            color=[SUBREDDIT_COLORS.get(
                                s, "#888888") for s in crisis_sub_pct.columns],
                            edgecolor="white", linewidth=0.5, width=0.7)
        ax3.set_xlabel("Crisis Period", fontsize=11)
        ax3.set_ylabel("Percentage of Posts (%)", fontsize=11)
        ax3.set_title("C. Subreddit Distribution During Each Crisis Period",
                      fontweight="bold", fontsize=13, pad=10)
        ax3.legend(title="Subreddit", bbox_to_anchor=(
            1.01, 1), loc="upper left", fontsize=9)
        plt.setp(ax3.xaxis.get_majorticklabels(),
                 rotation=30, ha="right", fontsize=10)
        ax3.set_ylim(0, 100)

    plt.suptitle("Crisis Period Analysis: Overview",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.subplots_adjust(right=0.85)

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.analysis_dir / "04a_crisis_overview.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {config.analysis_dir / '04a_crisis_overview.png'}")

    return fig


# =============================================================================
# VISUALIZATION 4B - Crisis Timelines
# =============================================================================
def plot_crisis_timelines(
    df: pd.DataFrame,
    config: PipelineConfig,
    save: bool = True,
) -> plt.Figure:
    """Create crisis timeline visualizations."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    panel_labels = ["A", "B", "C"]

    for idx, (fp_name, fp_config) in enumerate(FLASHPOINT_WINDOWS.items()):
        ax = axes[idx]
        start_dt = pd.to_datetime(fp_config["start"])
        end_dt = pd.to_datetime(fp_config["end"])
        crisis_window = df[(df["created_datetime"] >= start_dt)
                           & (df["created_datetime"] <= end_dt)]

        if len(crisis_window) > 0:
            daily = crisis_window.groupby("date").size()
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
            daily = daily.reindex(date_range.date, fill_value=0)

            line_color = fp_config["line_color"]
            label_color = fp_config["label_color"]

            ax.fill_between(daily.index, daily.values,
                            alpha=0.3, color=line_color)
            ax.plot(daily.index, daily.values, linewidth=2.5, color=line_color,
                    marker="o", markersize=4)

            y_max = daily.max()

            # Mark events
            for event_date_str, event_config in fp_config["events"].items():
                event_date = pd.to_datetime(event_date_str).date()
                if event_date in daily.index:
                    ax.axvline(event_date, color=label_color,
                               linestyle="--", linewidth=2, alpha=0.8)

                    x_off = event_config["x_offset"]
                    y_off = event_config["y_offset"]

                    ax.annotate(
                        event_config["name"],
                        xy=(event_date, daily.get(event_date, 0)),
                        xytext=(x_off, y_off),
                        textcoords="offset points",
                        fontsize=10, fontweight="bold", color=label_color,
                        ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=label_color, alpha=0.95),
                        arrowprops=dict(arrowstyle="->",
                                        color=label_color, lw=1.5)
                    )

            # Peak annotation
            peak_date = daily.idxmax()
            peak_val = daily.max()
            ax.annotate(f"Peak: {peak_val:,}", xy=(peak_date, peak_val),
                        xytext=(-70, 10), textcoords="offset points",
                        fontsize=10, color="#333333", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffcc",
                                  edgecolor="#333333", alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color="#333333", lw=1))

            # Stats box
            stats_text = f"Total: {len(crisis_window):,}  |  Peak: {daily.max():,}  |  Avg: {daily.mean():.0f}/day"
            ax.text(0.99, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9, edgecolor="#d4a574"))
            ax.set_ylim(0, y_max * 1.25)

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Posts per Day", fontsize=12)
        ax.set_title(f"{panel_labels[idx]}. {fp_name} Crisis Timeline",
                     fontweight="bold", fontsize=14, pad=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=45, ha="right", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("India-Pakistan Crisis Timelines",
                 fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.analysis_dir / "04b_crisis_timelines.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {config.analysis_dir / '04b_crisis_timelines.png'}")

    return fig


# =============================================================================
# VISUALIZATION 5 - Author Analysis
# =============================================================================
def plot_author_analysis(
    df: pd.DataFrame,
    config: PipelineConfig,
    save: bool = True,
) -> plt.Figure:
    """Create author analysis visualization."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.25)
    df_authors = df[df["author"] != "[deleted]"]

    # Panel A: Top authors
    ax1 = fig.add_subplot(gs[0, 0])
    top_authors = df_authors["author"].value_counts().head(15)
    ax1.barh(top_authors.index, top_authors.values,
             color="steelblue", edgecolor="darkblue")
    ax1.set_xlabel("Number of Posts")
    ax1.set_title("A. Top 15 Authors by Post Count", fontweight="bold")
    ax1.invert_yaxis()

    # Panel B: Author Activity Distribution (LINE CHART)
    ax2 = fig.add_subplot(gs[0, 1])
    author_posts = df_authors.groupby("author").size()
    bins = [1, 2, 5, 10, 20, 50, 100, author_posts.max() + 1]
    hist, edges = np.histogram(author_posts, bins=bins)
    bin_labels = [
        f"{int(edges[i])}-{int(edges[i + 1] - 1)}" for i in range(len(hist))]
    x_positions = np.arange(len(hist))

    ax2.plot(x_positions, hist, marker="o", markersize=8, linewidth=2.5,
             color="coral", markeredgecolor="darkred", markerfacecolor="coral")
    ax2.fill_between(x_positions, hist, alpha=0.3, color="coral")
    for i, (x, y) in enumerate(zip(x_positions, hist)):
        ax2.annotate(f"{y:,}", xy=(x, y), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold", color="darkred")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax2.set_xlabel("Posts per Author")
    ax2.set_ylabel("Number of Authors")
    ax2.set_title("B. Author Activity Distribution", fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_ylim(0, max(hist) * 1.15)

    # Panel C: Cross-Subreddit Posting (LINE CHART)
    ax3 = fig.add_subplot(gs[1, 0])
    author_subs = df_authors.groupby("author")["subreddit"].nunique()
    cross_post_dist = author_subs.value_counts().sort_index()
    x_vals = cross_post_dist.index[:9].values
    y_vals = cross_post_dist.values[:9]

    ax3.plot(x_vals, y_vals, marker="s", markersize=8, linewidth=2.5,
             color="seagreen", markeredgecolor="darkgreen", markerfacecolor="seagreen")
    ax3.fill_between(x_vals, y_vals, alpha=0.3, color="seagreen")
    for x, y in zip(x_vals, y_vals):
        ax3.annotate(f"{y:,}", xy=(x, y), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold", color="darkgreen")
    ax3.set_xlabel("Number of Subreddits")
    ax3.set_ylabel("Number of Authors")
    ax3.set_title("C. Cross-Subreddit Posting", fontweight="bold")
    ax3.set_xticks(x_vals)
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.set_ylim(0, max(y_vals) * 1.15)

    # Panel D: Author engagement
    ax4 = fig.add_subplot(gs[1, 1])
    top_author_list = top_authors.index[:10].tolist()
    author_engagement = df_authors[df_authors["author"].isin(top_author_list)].groupby("author").agg({
        "score": "mean", "num_comments": "mean"
    }).reindex(top_author_list)
    x = np.arange(len(top_author_list))
    width = 0.35
    ax4.bar(x - width / 2, author_engagement["score"],
            width, label="Avg Score", color="steelblue")
    ax4.bar(x + width / 2, author_engagement["num_comments"],
            width, label="Avg Comments", color="coral")
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_author_list, rotation=45, ha="right")
    ax4.set_ylabel("Count")
    ax4.set_title("D. Top 10 Authors Engagement", fontweight="bold")
    ax4.legend()

    plt.suptitle("Author Analysis of India-Pakistan Conflict Discourse",
                 fontsize=16, fontweight="bold", y=1.02)

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.analysis_dir / "05_author_analysis.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {config.analysis_dir / '05_author_analysis.png'}")

    return fig


# =============================================================================
# VISUALIZATION 6A - Word Clouds (Community)
# =============================================================================
def plot_wordclouds_community(
    df: pd.DataFrame,
    config: PipelineConfig,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Create word cloud visualizations by community."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("WordCloud not available. Install with: pip install wordcloud")
        return None

    df["clean_text"] = df["title"].fillna("").apply(preprocess_text)
    stop_words = get_stopwords()
    all_text = " ".join(df["clean_text"].tolist())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    wc_stopwords = stop_words.copy()

    # A. Overall Word Cloud
    ax = axes[0, 0]
    wc_all = WordCloud(
        width=800, height=400, background_color="white", colormap="viridis",
        max_words=100, stopwords=wc_stopwords, collocations=False,
        normalize_plurals=True, repeat=False
    ).generate(all_text)
    ax.imshow(wc_all, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("A. Overall Discourse Word Cloud",
                 fontweight="bold", fontsize=13, pad=10)

    # B. Indian Subreddits Word Cloud
    ax = axes[0, 1]
    india_subs = ["india", "indiaspeaks", "indiandefense"]
    india_text = " ".join(
        df[df["subreddit"].isin(india_subs)]["clean_text"].tolist())
    india_stops = wc_stopwords | {"india", "indian", "indias"}
    if india_text.strip():
        wc_india = WordCloud(
            width=800, height=400, background_color="white", colormap="Oranges",
            max_words=80, stopwords=india_stops, collocations=False, normalize_plurals=True
        ).generate(india_text)
        ax.imshow(wc_india, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("B. Indian Subreddits (r/india, r/IndiaSpeaks)",
                 fontweight="bold", fontsize=13, pad=10)

    # C. Pakistani Subreddit Word Cloud
    ax = axes[1, 0]
    pak_text = " ".join(df[df["subreddit"] == "pakistan"]
                        ["clean_text"].tolist())
    pak_stops = wc_stopwords | {"pakistan", "pakistani", "pakistans", "pak"}
    if pak_text.strip():
        wc_pak = WordCloud(
            width=800, height=400, background_color="white", colormap="Greens",
            max_words=80, stopwords=pak_stops, collocations=False, normalize_plurals=True
        ).generate(pak_text)
        ax.imshow(wc_pak, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("C. Pakistani Subreddit (r/pakistan)",
                 fontweight="bold", fontsize=13, pad=10)

    # D. International News Word Cloud
    ax = axes[1, 1]
    intl_text = " ".join(
        df[df["subreddit"] == "worldnews"]["clean_text"].tolist())
    if intl_text.strip():
        wc_intl = WordCloud(
            width=800, height=400, background_color="white", colormap="Purples",
            max_words=80, stopwords=wc_stopwords, collocations=False, normalize_plurals=True
        ).generate(intl_text)
        ax.imshow(wc_intl, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("D. International News (r/worldnews)",
                 fontweight="bold", fontsize=13, pad=10)

    plt.suptitle("Word Cloud Analysis by Community",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.analysis_dir / "06a_wordclouds_community.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {config.analysis_dir / '06a_wordclouds_community.png'}")

    return fig


# =============================================================================
# VISUALIZATION 6B - Word Clouds (Crisis)
# =============================================================================
def plot_wordclouds_crisis(
    df: pd.DataFrame,
    config: PipelineConfig,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Create crisis-specific word cloud visualizations."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("WordCloud not available. Install with: pip install wordcloud")
        return None

    df["clean_text"] = df["title"].fillna("").apply(preprocess_text)
    df_crisis = identify_crisis_posts(df, CRISIS_PERIODS)
    stop_words = get_stopwords()
    crisis_stops = stop_words | {"india", "pakistan", "indian", "pakistani"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    crisis_config = {
        "Pulwama Attack": {"ax": axes[0], "cmap": "Reds", "title": "Pulwama-Balakot (Feb 2019)"},
        "Article 370 Abrogation": {"ax": axes[1], "cmap": "Purples", "title": "Article 370 (Aug 2019)"},
        "Operation Sindoor": {"ax": axes[2], "cmap": "Blues", "title": "Operation Sindoor (May 2025)"},
    }

    for crisis, cfg in crisis_config.items():
        ax = cfg["ax"]
        crisis_df = df_crisis[df_crisis["crisis_period"] == crisis]
        if len(crisis_df) > 20:
            crisis_text = " ".join(crisis_df["clean_text"].tolist())
            wc = WordCloud(
                width=800, height=500, background_color="white", colormap=cfg["cmap"],
                max_words=60, stopwords=crisis_stops, collocations=False,
                normalize_plurals=True, min_font_size=8, max_font_size=100, relative_scaling=0.5
            ).generate(crisis_text)
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(f"{cfg['title']}\n(n={len(crisis_df):,})",
                         fontweight="bold", fontsize=12, pad=10)
        else:
            ax.text(0.5, 0.5, f"Insufficient data\n({len(crisis_df)} posts)",
                    ha="center", va="center", fontsize=14, color="gray")
            ax.set_title(cfg["title"], fontweight="bold", fontsize=12, pad=10)
        ax.axis("off")

    plt.suptitle("Word Clouds by Crisis Period (India-Pakistan)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.analysis_dir / "06b_wordclouds_crisis.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {config.analysis_dir / '06b_wordclouds_crisis.png'}")

    return fig


# =============================================================================
# VISUALIZATION 6C - N-gram Analysis & Topic Trends
# =============================================================================
def plot_ngrams_topics(
    df: pd.DataFrame,
    config: PipelineConfig,
    save: bool = True,
) -> plt.Figure:
    """Create n-gram analysis and topic trends visualization."""
    try:
        from nltk.tokenize import word_tokenize
        from nltk.util import ngrams
        from .analyzers import ensure_nltk_resources
        ensure_nltk_resources()
    except ImportError:
        print("NLTK not available. Skipping n-gram analysis.")
        return None

    df["clean_text"] = df["title"].fillna("").apply(preprocess_text)
    stop_words = get_stopwords()
    ngram_stopwords = stop_words | {
        "india", "pakistan", "indian", "pakistani", "indias", "pakistans", "pak",
        "says", "said", "will", "would", "could", "may", "also", "one", "two",
        "first", "new", "last", "year", "years", "day", "days", "today", "now"
    }

    all_text = " ".join(df["clean_text"].tolist())
    tokens = word_tokenize(all_text)
    filtered_tokens_enhanced = [
        w for w in tokens if w not in ngram_stopwords and len(w) > 2]

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Panel A: Top Unigrams
    ax1 = fig.add_subplot(gs[0, 0])
    unigram_freq = Counter(filtered_tokens_enhanced).most_common(20)
    words, counts = zip(*unigram_freq)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))
    bars = ax1.barh(range(len(words)), counts, color=colors,
                    edgecolor="black", alpha=0.85)
    ax1.set_yticks(range(len(words)))
    ax1.set_yticklabels(words, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Frequency", fontsize=11)
    ax1.set_title("A. Top 20 Unigrams (Excluding Country Names)",
                  fontweight="bold", fontsize=13, pad=10)
    for i, (w, c) in enumerate(zip(words, counts)):
        ax1.text(c + max(counts) * 0.02, i, f"{c:,}", va="center", fontsize=9)
    ax1.set_xlim(0, max(counts) * 1.15)

    # Panel B: Top Bigrams
    ax2 = fig.add_subplot(gs[0, 1])
    bigrams_list = list(ngrams(filtered_tokens_enhanced, 2))
    country_terms = {"india", "pakistan", "indian",
                     "pakistani", "indias", "pakistans"}
    filtered_bigrams = [b for b in bigrams_list if not (
        b[0] in country_terms and b[1] in country_terms)]
    bigram_freq = Counter(filtered_bigrams).most_common(20)
    bigram_labels = [f"{b[0]} {b[1]}" for b, _ in bigram_freq]
    bigram_counts = [c for _, c in bigram_freq]
    colors_bi = plt.cm.plasma(np.linspace(0.2, 0.8, len(bigram_labels)))
    ax2.barh(range(len(bigram_labels)), bigram_counts,
             color=colors_bi, edgecolor="black", alpha=0.85)
    ax2.set_yticks(range(len(bigram_labels)))
    ax2.set_yticklabels(bigram_labels, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Frequency", fontsize=11)
    ax2.set_title("B. Top 20 Bigrams (Meaningful Pairs)",
                  fontweight="bold", fontsize=13, pad=10)
    for i, c in enumerate(bigram_counts):
        ax2.text(c + max(bigram_counts) * 0.02, i,
                 f"{c:,}", va="center", fontsize=9)
    ax2.set_xlim(0, max(bigram_counts) * 1.15)

    # Panel C: Topic Category Trends
    ax3 = fig.add_subplot(gs[1, :])
    df_topics = df.copy()
    titles_lower = df_topics["clean_text"]
    for topic, keywords in TOPIC_CATEGORIES.items():
        pattern = "|".join(keywords)
        df_topics[topic] = titles_lower.str.contains(
            pattern, regex=True, na=False)
    monthly_topics = df_topics.groupby(
        "year_month")[list(TOPIC_CATEGORIES.keys())].sum()
    monthly_topics.index = monthly_topics.index.to_timestamp()

    for topic in TOPIC_CATEGORIES.keys():
        ax3.plot(monthly_topics.index, monthly_topics[topic], label=topic,
                 linewidth=2.5, color=TOPIC_COLORS[topic], marker="o", markersize=4, alpha=0.85)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.set_ylabel("Mentions per Month", fontsize=11)
    ax3.set_title("C. Topic Category Trends Over Time",
                  fontweight="bold", fontsize=13, pad=10)
    ax3.legend(loc="upper left", fontsize=10, ncol=5,
               frameon=True, bbox_to_anchor=(0.5, 1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax3.grid(True, alpha=0.3, linestyle="--")

    # Annotate crisis periods
    crisis_annotations = [("2019-02-14", "Pulwama"),
                          ("2019-08-05", "Art.370"), ("2025-05-01", "Op.Sindoor")]
    for date_str, label in crisis_annotations:
        date = pd.to_datetime(date_str)
        if monthly_topics.index.min() <= date <= monthly_topics.index.max():
            ax3.axvline(date, color="#d62728", linestyle="--",
                        linewidth=2, alpha=0.6)
            ax3.annotate(label, xy=(date, ax3.get_ylim()[1] * 0.85), fontsize=9, fontweight="bold",
                         rotation=90, color="#d62728", ha="right", va="top",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d62728", alpha=0.8))

    plt.suptitle("N-gram Analysis & Topic Trends",
                 fontsize=16, fontweight="bold", y=0.98)

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.analysis_dir / "06c_ngrams_topics.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {config.analysis_dir / '06c_ngrams_topics.png'}")

    return fig


# =============================================================================
# VISUALIZATION 6D - Vocabulary Comparison
# =============================================================================
def plot_vocabulary_comparison(
    df: pd.DataFrame,
    config: PipelineConfig,
    save: bool = True,
) -> plt.Figure:
    """Create subreddit and crisis vocabulary comparison visualization."""
    try:
        from nltk.tokenize import word_tokenize
        from .analyzers import ensure_nltk_resources
        ensure_nltk_resources()
    except ImportError:
        print("NLTK not available. Skipping vocabulary comparison.")
        return None

    df["clean_text"] = df["title"].fillna("").apply(preprocess_text)
    df_crisis = identify_crisis_posts(df, CRISIS_PERIODS)
    stop_words = get_stopwords()
    vocab_stopwords = stop_words | {
        "india", "pakistan", "indian", "pakistani", "indias", "pakistans", "pak",
        "country", "countries", "government", "world", "people", "news"
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel A: r/india vs r/pakistan vocabulary
    ax = axes[0]
    sub_words = {}
    for sub in ["india", "pakistan"]:
        sub_df = df[df["subreddit"] == sub]
        if len(sub_df) > 0:
            sub_text = " ".join(sub_df["clean_text"].tolist())
            sub_tokens = word_tokenize(sub_text)
            sub_filtered = [
                w for w in sub_tokens if w not in vocab_stopwords and len(w) > 2]
            sub_words[sub] = Counter(sub_filtered).most_common(20)

    if "india" in sub_words and "pakistan" in sub_words:
        india_dict = dict(sub_words["india"])
        pak_dict = dict(sub_words["pakistan"])
        all_words_freq = {}
        for w in set(india_dict.keys()) | set(pak_dict.keys()):
            all_words_freq[w] = india_dict.get(w, 0) + pak_dict.get(w, 0)
        top_words = sorted(all_words_freq.items(),
                           key=lambda x: x[1], reverse=True)[:15]
        all_words_set = [w for w, _ in top_words]
        india_counts = [india_dict.get(w, 0) for w in all_words_set]
        pak_counts = [pak_dict.get(w, 0) for w in all_words_set]

        x = np.arange(len(all_words_set))
        width = 0.4
        ax.barh(x - width / 2, india_counts, width, label="r/india", color="#FF9933",
                edgecolor="black", alpha=0.85)
        ax.barh(x + width / 2, pak_counts, width, label="r/pakistan", color="#01411C",
                edgecolor="black", alpha=0.85)
        ax.set_yticks(x)
        ax.set_yticklabels(all_words_set, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency", fontsize=11)
        ax.set_title("A. Vocabulary: r/india vs r/pakistan",
                     fontweight="bold", fontsize=13, pad=10)
        ax.legend(loc="lower right", fontsize=11, frameon=True)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    # Panel B: Crisis vocabulary comparison
    ax = axes[1]
    crisis_vocab_stops = {
        "Pulwama Attack": vocab_stopwords | {"pulwama", "attack", "attacks", "february"},
        "Article 370 Abrogation": vocab_stopwords | {"article", "370", "august", "revoked"},
        "Operation Sindoor": vocab_stopwords | {"sindoor", "operation", "pahalgam", "may"},
    }
    crisis_words = {}
    for crisis in ["Pulwama Attack", "Article 370 Abrogation", "Operation Sindoor"]:
        crisis_df = df_crisis[df_crisis["crisis_period"] == crisis]
        if len(crisis_df) > 50:
            crisis_text = " ".join(crisis_df["clean_text"].tolist())
            crisis_tokens = word_tokenize(crisis_text)
            stops = crisis_vocab_stops.get(crisis, vocab_stopwords)
            crisis_filtered = [
                w for w in crisis_tokens if w not in stops and len(w) > 2]
            crisis_words[crisis] = Counter(crisis_filtered).most_common(15)

    if crisis_words:
        crisis_names = list(crisis_words.keys())
        all_crisis_words_freq = {}
        for crisis in crisis_names:
            for w, c in crisis_words[crisis]:
                all_crisis_words_freq[w] = all_crisis_words_freq.get(w, 0) + c
        top_crisis_words = sorted(
            all_crisis_words_freq.items(), key=lambda x: x[1], reverse=True)[:12]
        all_crisis_words = [w for w, _ in top_crisis_words]

        x = np.arange(len(all_crisis_words))
        width = 0.25
        crisis_colors_map = {
            "Pulwama Attack": "#E63946",
            "Article 370 Abrogation": "#9467bd",
            "Operation Sindoor": "#3A86FF"
        }
        for i, crisis in enumerate(crisis_names):
            crisis_dict = dict(crisis_words[crisis])
            counts = [crisis_dict.get(w, 0) for w in all_crisis_words]
            ax.barh(x + i * width, counts, width, label=crisis,
                    color=crisis_colors_map[crisis], edgecolor="black", alpha=0.85)
        ax.set_yticks(x + width)
        ax.set_yticklabels(all_crisis_words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency", fontsize=11)
        ax.set_title("B. Crisis-Specific Vocabulary\n(Excluding Crisis Names)",
                     fontweight="bold", fontsize=13, pad=10)
        ax.legend(loc="lower right", fontsize=9, frameon=True)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    plt.suptitle("Comparative Vocabulary Analysis",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.analysis_dir / "06d_vocabulary_comparison.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        print(
            f"Saved: {config.analysis_dir / '06d_vocabulary_comparison.png'}")

    return fig


# =============================================================================
# GENERATE ALL VISUALIZATIONS
# =============================================================================
def generate_all_visualizations(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    """Generate all standard visualizations."""
    print("\nGenerating visualizations...")
    config.analysis_dir.mkdir(parents=True, exist_ok=True)

    plot_temporal_overview(df, config, save=True)
    plot_subreddit_analysis(df, config, save=True)
    plot_engagement_analysis(df, config, save=True)
    plot_crisis_overview(df, config, save=True)
    plot_crisis_timelines(df, config, save=True)
    plot_author_analysis(df, config, save=True)

    # NLP visualizations (require wordcloud and nltk)
    try:
        plot_wordclouds_community(df, config, save=True)
        plot_wordclouds_crisis(df, config, save=True)
        plot_ngrams_topics(df, config, save=True)
        plot_vocabulary_comparison(df, config, save=True)
    except Exception as e:
        print(f"NLP visualizations skipped: {e}")

    print("\nAll visualizations generated successfully.")
