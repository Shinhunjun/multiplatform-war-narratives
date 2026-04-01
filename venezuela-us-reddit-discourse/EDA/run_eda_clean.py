"""
EDA script for preprocessed Venezuela-US Reddit data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path(__file__).parent.parent / "data-collection" / "data" / "preprocessed"
OUTPUT_DIR = Path(__file__).parent

# Regional classification
VENEZUELA_SUBS = ["vzla", "venezuela"]
US_SUBS = ["politics", "news", "worldnews", "Conservative", "Libertarian", "neoliberal", "socialism", "geopolitics", "LatinAmerica"]

# Colors
COLORS = {
    "venezuela": "#FCDD09",
    "vzla": "#CF142B",
    "politics": "#3C3B6E",
    "news": "#B22234",
    "worldnews": "#1f77b4",
    "geopolitics": "#2ca02c",
    "LatinAmerica": "#FF6B35",
    "Conservative": "#d62728",
    "neoliberal": "#17becf",
    "socialism": "#e377c2",
    "Libertarian": "#bcbd22",
}


def load_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    submissions = pd.read_parquet(DATA_DIR / "submissions_clean.parquet")
    comments = pd.read_parquet(DATA_DIR / "comments_clean.parquet")

    # Convert timestamps
    submissions["created_utc"] = pd.to_datetime(submissions["created_utc"], unit="s")
    comments["created_utc"] = pd.to_datetime(comments["created_utc"], unit="s")

    # Add date columns
    submissions["date"] = submissions["created_utc"].dt.date
    submissions["year"] = submissions["created_utc"].dt.year
    submissions["month"] = submissions["created_utc"].dt.to_period("M")

    comments["date"] = comments["created_utc"].dt.date
    comments["year"] = comments["created_utc"].dt.year
    comments["month"] = comments["created_utc"].dt.to_period("M")

    # Add region
    submissions["region"] = submissions["subreddit"].apply(
        lambda x: "Venezuela" if x.lower() in [s.lower() for s in VENEZUELA_SUBS] else "US/English"
    )
    comments["region"] = comments["subreddit"].apply(
        lambda x: "Venezuela" if x.lower() in [s.lower() for s in VENEZUELA_SUBS] else "US/English"
    )

    print(f"Loaded {len(submissions):,} submissions, {len(comments):,} comments")
    return submissions, comments


def plot_timeline(submissions, comments):
    """Plot full timeline."""
    print("Creating timeline plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Monthly aggregation
    sub_monthly = submissions.groupby("month").size()
    com_monthly = comments.groupby("month").size()

    # Convert to datetime for plotting
    sub_dates = [p.to_timestamp() for p in sub_monthly.index]
    com_dates = [p.to_timestamp() for p in com_monthly.index]

    # Submissions
    axes[0].fill_between(sub_dates, sub_monthly.values, alpha=0.7, color="#1f77b4")
    axes[0].set_ylabel("Submissions", fontsize=12)
    axes[0].set_title("Venezuela-US Reddit Discourse Timeline (Preprocessed Data)", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Key events
    events = {
        "2014-02": "2014 Protests",
        "2017-08": "Trump Sanctions",
        "2019-01": "Guaido Crisis",
        "2024-07": "2024 Election",
        "2026-01": "Maduro Captured",
    }

    for date_str, label in events.items():
        date = datetime.strptime(date_str, "%Y-%m")
        axes[0].axvline(date, color="red", linestyle="--", alpha=0.5)
        axes[0].text(date, axes[0].get_ylim()[1] * 0.9, label, rotation=45, fontsize=8, ha="left")

    # Comments
    axes[1].fill_between(com_dates, com_monthly.values, alpha=0.7, color="#2ca02c")
    axes[1].set_ylabel("Comments", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    for date_str, label in events.items():
        date = datetime.strptime(date_str, "%Y-%m")
        axes[1].axvline(date, color="red", linestyle="--", alpha=0.5)

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_timeline_clean.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 01_timeline_clean.png")


def plot_yearly_distribution(submissions, comments):
    """Plot yearly distribution."""
    print("Creating yearly distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Submissions by year
    sub_yearly = submissions.groupby("year").size()
    axes[0].bar(sub_yearly.index, sub_yearly.values, color="#1f77b4", alpha=0.8)
    axes[0].set_title("Submissions by Year", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Comments by year
    com_yearly = comments.groupby("year").size()
    axes[1].bar(com_yearly.index, com_yearly.values, color="#2ca02c", alpha=0.8)
    axes[1].set_title("Comments by Year", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Yearly Distribution (Preprocessed Data)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_yearly_distribution_clean.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 02_yearly_distribution_clean.png")


def plot_subreddit_distribution(submissions, comments):
    """Plot subreddit distribution."""
    print("Creating subreddit distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Submissions by subreddit
    sub_by_subreddit = submissions["subreddit"].value_counts()
    colors_sub = [COLORS.get(s, "#999999") for s in sub_by_subreddit.index]
    axes[0].pie(sub_by_subreddit.values, labels=sub_by_subreddit.index, colors=colors_sub,
                autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Submissions by Subreddit", fontsize=12, fontweight="bold")

    # Comments by subreddit
    com_by_subreddit = comments["subreddit"].value_counts()
    colors_com = [COLORS.get(s, "#999999") for s in com_by_subreddit.index]
    axes[1].pie(com_by_subreddit.values, labels=com_by_subreddit.index, colors=colors_com,
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Comments by Subreddit", fontsize=12, fontweight="bold")

    plt.suptitle("Subreddit Distribution (Preprocessed Data)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_subreddit_distribution_clean.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 03_subreddit_distribution_clean.png")


def plot_region_comparison(submissions, comments):
    """Plot region comparison."""
    print("Creating region comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Submissions by region
    sub_region = submissions["region"].value_counts()
    colors = ["#FCDD09", "#3C3B6E"]
    axes[0].pie(sub_region.values, labels=sub_region.index, colors=colors,
                autopct="%1.1f%%", startangle=90, explode=[0.02, 0.02])
    axes[0].set_title(f"Submissions\n(n={len(submissions):,})", fontsize=12)

    # Comments by region
    com_region = comments["region"].value_counts()
    axes[1].pie(com_region.values, labels=com_region.index, colors=colors,
                autopct="%1.1f%%", startangle=90, explode=[0.02, 0.02])
    axes[1].set_title(f"Comments\n(n={len(comments):,})", fontsize=12)

    plt.suptitle("Regional Distribution (Preprocessed Data)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_region_comparison_clean.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 04_region_comparison_clean.png")


def plot_engagement_metrics(submissions):
    """Plot engagement metrics by subreddit."""
    print("Creating engagement metrics plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Average score by subreddit
    avg_score = submissions.groupby("subreddit")["score"].mean().sort_values(ascending=True)
    colors = [COLORS.get(s, "#999999") for s in avg_score.index]
    axes[0].barh(avg_score.index, avg_score.values, color=colors)
    axes[0].set_xlabel("Average Score")
    axes[0].set_title("Average Score by Subreddit", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Average comments by subreddit
    avg_comments = submissions.groupby("subreddit")["num_comments"].mean().sort_values(ascending=True)
    colors = [COLORS.get(s, "#999999") for s in avg_comments.index]
    axes[1].barh(avg_comments.index, avg_comments.values, color=colors)
    axes[1].set_xlabel("Average Comments")
    axes[1].set_title("Average Comments per Post", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.suptitle("Engagement Metrics (Preprocessed Data)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_engagement_metrics_clean.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 05_engagement_metrics_clean.png")


def plot_top_authors(submissions, comments):
    """Plot top authors."""
    print("Creating top authors plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top submission authors (excluding bots/deleted)
    top_sub_authors = submissions["author"].value_counts().head(15)
    axes[0].barh(top_sub_authors.index[::-1], top_sub_authors.values[::-1], color="#1f77b4")
    axes[0].set_xlabel("Number of Posts")
    axes[0].set_title("Top 15 Submission Authors", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Top comment authors
    top_com_authors = comments["author"].value_counts().head(15)
    axes[1].barh(top_com_authors.index[::-1], top_com_authors.values[::-1], color="#2ca02c")
    axes[1].set_xlabel("Number of Comments")
    axes[1].set_title("Top 15 Comment Authors", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.suptitle("Top Authors (Preprocessed Data - Bots Removed)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_top_authors_clean.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 06_top_authors_clean.png")


def generate_report(submissions, comments):
    """Generate markdown report."""
    print("Generating EDA report...")

    # Calculate statistics
    total_submissions = len(submissions)
    total_comments = len(comments)
    total_data = total_submissions + total_comments
    unique_sub_authors = submissions["author"].nunique()
    unique_com_authors = comments["author"].nunique()
    subreddits = submissions["subreddit"].nunique()

    # Date range
    min_date = submissions["created_utc"].min().strftime("%Y-%m-%d")
    max_date = submissions["created_utc"].max().strftime("%Y-%m-%d")

    # Regional stats
    sub_region = submissions["region"].value_counts()
    com_region = comments["region"].value_counts()

    # Subreddit stats
    sub_stats = submissions.groupby("subreddit").agg({
        "score": ["count", "mean", "median"],
        "num_comments": "mean"
    }).round(1)
    sub_stats.columns = ["Posts", "Avg Score", "Median Score", "Avg Comments"]
    sub_stats = sub_stats.sort_values("Posts", ascending=False)

    # Yearly stats
    yearly_sub = submissions.groupby("year").size()
    yearly_com = comments.groupby("year").size()

    # Top months
    monthly_sub = submissions.groupby("month").size().sort_values(ascending=False).head(10)

    # Top posts
    top_posts = submissions.nlargest(10, "score")[["subreddit", "title", "score", "num_comments", "created_utc"]]

    report = f"""# Venezuela-US Reddit Discourse EDA Report (Preprocessed)

## Overview

| Metric | Value |
|--------|-------|
| **Data Period** | {min_date} ~ {max_date} |
| **Total Submissions** | {total_submissions:,} |
| **Total Comments** | {total_comments:,} |
| **Total Data Points** | {total_data:,} |
| **Unique Submission Authors** | {unique_sub_authors:,} |
| **Unique Comment Authors** | {unique_com_authors:,} |
| **Subreddits** | {subreddits} |

### Preprocessing Applied
- Removed `[deleted]` and `[removed]` content
- Removed bot accounts (AutoModerator, autotldr, etc.)
- Removed comments with < 5 words
- Cleaned URLs, markdown formatting, edit markers

---

## Regional Distribution

### Subreddit Categories

| Region | Subreddits |
|--------|------------|
| Venezuela | r/vzla, r/venezuela |
| US/English | r/politics, r/news, r/worldnews, r/Conservative, r/Libertarian, r/neoliberal, r/socialism, r/geopolitics, r/LatinAmerica |

### Data by Region

| Region | Submissions | % | Comments | % |
|--------|-------------|---|----------|---|
| Venezuela | {sub_region.get('Venezuela', 0):,} | {sub_region.get('Venezuela', 0)/total_submissions*100:.1f}% | {com_region.get('Venezuela', 0):,} | {com_region.get('Venezuela', 0)/total_comments*100:.1f}% |
| US/English | {sub_region.get('US/English', 0):,} | {sub_region.get('US/English', 0)/total_submissions*100:.1f}% | {com_region.get('US/English', 0):,} | {com_region.get('US/English', 0)/total_comments*100:.1f}% |

![Region Comparison](04_region_comparison_clean.png)

---

## Timeline Analysis

### Full Timeline (2013-2026)

![Timeline](01_timeline_clean.png)

### Key Events

| Date | Event | Impact |
|------|-------|--------|
| 2014-02 | Venezuelan Protests | First major spike |
| 2017-08 | Trump Sanctions | Increased US attention |
| 2019-01 | Guaido Crisis | Highest peak |
| 2024-07 | 2024 Presidential Election | Recent surge |
| 2026-01 | Maduro Captured by US Forces | Latest peak |

### Top 10 Peak Months

| Month | Posts |
|-------|-------|
"""

    for month, count in monthly_sub.items():
        report += f"| {month} | {count:,} |\n"

    report += f"""
### Yearly Distribution

![Yearly Distribution](02_yearly_distribution_clean.png)

| Year | Submissions | Comments |
|------|-------------|----------|
"""

    for year in sorted(set(yearly_sub.index) | set(yearly_com.index)):
        sub_count = yearly_sub.get(year, 0)
        com_count = yearly_com.get(year, 0)
        report += f"| {year} | {sub_count:,} | {com_count:,} |\n"

    report += f"""
---

## Subreddit Analysis

![Subreddit Distribution](03_subreddit_distribution_clean.png)

### Subreddit Statistics

| Subreddit | Posts | Avg Score | Median Score | Avg Comments |
|-----------|-------|-----------|--------------|--------------|
"""

    for subreddit, row in sub_stats.iterrows():
        region = "Venezuela" if subreddit.lower() in [s.lower() for s in VENEZUELA_SUBS] else "US/English"
        emoji = "\U0001F1FB\U0001F1EA" if region == "Venezuela" else "\U0001F1FA\U0001F1F8"
        report += f"| {emoji} r/{subreddit} | {int(row['Posts']):,} | {row['Avg Score']:.1f} | {row['Median Score']:.1f} | {row['Avg Comments']:.1f} |\n"

    report += f"""
---

## Engagement Analysis

![Engagement Metrics](05_engagement_metrics_clean.png)

### Key Insights

- **US/English subreddits** have significantly higher average scores (more upvotes)
- **Venezuelan subreddits** have more total posts but lower individual engagement
- **r/worldnews** and **r/politics** drive the highest engagement per post

---

## Author Analysis

![Top Authors](06_top_authors_clean.png)

### Top 10 Submission Authors

| Author | Posts |
|--------|-------|
"""

    top_sub_authors = submissions["author"].value_counts().head(10)
    for author, count in top_sub_authors.items():
        report += f"| {author} | {count:,} |\n"

    report += f"""
### Top 10 Comment Authors

| Author | Comments |
|--------|----------|
"""

    top_com_authors = comments["author"].value_counts().head(10)
    for author, count in top_com_authors.items():
        report += f"| {author} | {count:,} |\n"

    report += f"""
---

## Top Posts by Score

| Rank | Subreddit | Title | Score | Comments | Date |
|------|-----------|-------|-------|----------|------|
"""

    for i, (_, row) in enumerate(top_posts.iterrows(), 1):
        title = row["title"][:50] + "..." if len(str(row["title"])) > 50 else row["title"]
        date = row["created_utc"].strftime("%Y-%m-%d")
        report += f"| {i} | r/{row['subreddit']} | {title} | {row['score']:,} | {row['num_comments']:,} | {date} |\n"

    report += f"""
---

## Data Files

### Preprocessed Data Location
```
data-collection/data/preprocessed/
├── submissions_clean.parquet
└── comments_clean.parquet
```

---

*Generated: {datetime.now().strftime('%Y-%m-%d')}*
"""

    with open(OUTPUT_DIR / "EDA_Report_Clean.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("  Saved EDA_Report_Clean.md")


def main():
    """Run full EDA."""
    print("=" * 60)
    print("EDA for Preprocessed Venezuela-US Reddit Data")
    print("=" * 60)

    # Load data
    submissions, comments = load_data()

    # Generate plots
    plot_timeline(submissions, comments)
    plot_yearly_distribution(submissions, comments)
    plot_subreddit_distribution(submissions, comments)
    plot_region_comparison(submissions, comments)
    plot_engagement_metrics(submissions)
    plot_top_authors(submissions, comments)

    # Generate report
    generate_report(submissions, comments)

    print("=" * 60)
    print("EDA Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
