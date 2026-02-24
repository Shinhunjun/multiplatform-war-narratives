"""
EDA script for GDELT Venezuela-US Dataset.
Mirrors the structure/style of the Reddit Discourse EDA.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import seaborn as sns

# Configuration
# Ideally, place your CSV in a folder named 'data' relative to this script
DATA_PATH = Path(__file__).parent / "../data" / "bq-results-20260128-004024-1769560909144.csv" 
OUTPUT_DIR = Path(__file__).parent

# Colors (Matching the teammate's aesthetic where possible)
COLORS = {
    "Verbal Cooperation": "#1f77b4",   # Blue
    "Material Cooperation": "#2ca02c", # Green
    "Verbal Conflict": "#ff7f0e",      # Orange
    "Material Conflict": "#d62728",    # Red
    "USA": "#3C3B6E",
    "VEN": "#FCDD09",
    "Neutral": "#999999"
}

# Mapping for GDELT QuadClasses
QUADCLASS_MAP = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict"
}

def load_data(filepath):
    """Load and preprocess GDELT data."""
    print("Loading GDELT data...")
    
    # Check if file exists, if not, create dummy data for demonstration if needed
    # (In your real usage, this simply loads your file)
    if not filepath.exists():
        print(f"File not found at {filepath}. Please update DATA_PATH.")
        return None

    df = pd.read_csv(filepath)

    # Convert Date (GDELT format is usually YYYYMMDD)
    df["DateObject"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    
    # Add time components
    df["Year"] = df["DateObject"].dt.year
    df["Month"] = df["DateObject"].dt.to_period("M")
    
    # Map QuadClass to Labels
    df["EventCategory"] = df["QuadClass"].map(QUADCLASS_MAP)
    
    # Determine Region/Initiator Label for color coding
    # (Assuming Actor1 is the initiator)
    df["Initiator"] = df["Actor1CountryCode"].apply(
        lambda x: "Venezuela" if x == "VEN" else ("USA" if x == "USA" else "Other")
    )

    print(f"Loaded {len(df):,} events from {df['DateObject'].min().date()} to {df['DateObject'].max().date()}")
    return df

def plot_timeline(df):
    """Plot event volume timeline with Goldstein intensity overlay."""
    print("Creating timeline plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Monthly Event Counts
    monthly_counts = df.groupby("Month").size()
    dates = [p.to_timestamp() for p in monthly_counts.index]

    # Plot 1: Volume
    axes[0].fill_between(dates, monthly_counts.values, alpha=0.7, color="#1f77b4")
    axes[0].set_ylabel("Total Events", fontsize=12)
    axes[0].set_title("Venezuela-US Event Timeline (GDELT)", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Key Events (Mirrored from teammate's report for consistency)
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
        axes[0].text(
            date,
            axes[0].get_ylim()[1] * 0.82,
            label,
            rotation=45,
            fontsize=8,
            ha="left",
            va="top",
        )

    # Plot 2: Average Goldstein Scale (Stability/Conflict indicator)
    # Resample to monthly average
    monthly_goldstein = df.groupby("Month")["GoldsteinScale"].mean()
    
    axes[1].plot(dates, monthly_goldstein.values, color="#d62728", linewidth=1.5)
    axes[1].axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    axes[1].set_ylabel("Avg Goldstein Scale\n(Conflict < 0 < Coop)", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Fill positive/negative areas
    axes[1].fill_between(dates, monthly_goldstein.values, 0, where=(monthly_goldstein.values >= 0), facecolor='green', alpha=0.3)
    axes[1].fill_between(dates, monthly_goldstein.values, 0, where=(monthly_goldstein.values < 0), facecolor='red', alpha=0.3)

    for date_str, label in events.items():
        date = datetime.strptime(date_str, "%Y-%m")
        axes[1].axvline(date, color="red", linestyle="--", alpha=0.5)

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_gdelt_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 01_gdelt_timeline.png")

def plot_yearly_distribution(df):
    """Plot yearly event counts and AvgTone distribution by year."""
    print("Creating yearly distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Events by year
    yearly_counts = df.groupby("Year").size()
    axes[0].bar(yearly_counts.index, yearly_counts.values, color="#1f77b4", alpha=0.8)
    axes[0].set_title("Total Events by Year", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3, axis="y")

    # AvgTone distribution by year
    tone_df = df[["Year", "AvgTone"]].dropna().copy()
    tone_df["Year"] = tone_df["Year"].astype(int)
    ordered_years = sorted(tone_df["Year"].unique())

    sns.boxplot(
        data=tone_df,
        x="Year",
        y="AvgTone",
        order=ordered_years,
        ax=axes[1],
        color="#8fd3e8",
        fliersize=1.5,
        linewidth=1,
    )
    axes[1].set_title("Avg Tone Distribution by Year", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Avg Tone")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle("Yearly Activity & Tone Dispersion (GDELT)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_gdelt_yearly_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 02_gdelt_yearly_stats.png")

def plot_tone_trend(df, rolling_window=12):
    """Plot monthly AvgTone with rolling mean and +/-1 SD band."""
    print("Creating smoothed tone trend plot...")

    monthly_tone = (
        df.dropna(subset=["DateObject", "AvgTone"])
        .set_index("DateObject")["AvgTone"]
        .resample("ME")
        .mean()
    )

    rolling_mean = monthly_tone.rolling(window=rolling_window, min_periods=max(3, rolling_window // 3)).mean()
    rolling_std = monthly_tone.rolling(window=rolling_window, min_periods=max(3, rolling_window // 3)).std()
    upper = rolling_mean + rolling_std
    lower = rolling_mean - rolling_std

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(monthly_tone.index, monthly_tone.values, color="#9aa3ad", alpha=0.45, linewidth=1, label="Monthly Mean Tone")
    ax.plot(rolling_mean.index, rolling_mean.values, color="#1f77b4", linewidth=2, label=f"{rolling_window}-Month Rolling Mean")
    ax.fill_between(
        rolling_mean.index,
        lower.values,
        upper.values,
        color="#1f77b4",
        alpha=0.2,
        label="Rolling ±1 SD",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Smoothed Monthly Avg Tone with Volatility Band", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Tone")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_gdelt_tone_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 05_gdelt_tone_trend.png")

def plot_quadclass_distribution(df):
    """Plot QuadClass distribution (Verbal/Material Coop/Conflict)."""
    print("Creating QuadClass distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall QuadClass Distribution
    qc_counts = df["EventCategory"].value_counts()
    colors_qc = [COLORS.get(cat, "#999999") for cat in qc_counts.index]
    
    axes[0].pie(qc_counts.values, labels=qc_counts.index, colors=colors_qc,
                autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Event Categories (QuadClass)", fontsize=12, fontweight="bold")

    # Split by Initiator (USA vs VEN)
    # Create a crosstab
    ct = pd.crosstab(df["Initiator"], df["EventCategory"])
    # Normalize to get percentages
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    
    # Filter for just VEN and USA rows if others exist
    ct_norm = ct_norm.loc[ct_norm.index.isin(["Venezuela", "USA"])]
    
    ct_norm.plot(kind="bar", stacked=True, ax=axes[1], color=[COLORS.get(c) for c in ct_norm.columns])
    axes[1].set_title("Event Category Mix by Initiator", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("Initiator")
    axes[1].legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_gdelt_categories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 03_gdelt_categories.png")

def plot_intensity_metrics(df):
    """Plot Goldstein Scale and AvgTone distributions."""
    print("Creating intensity metrics plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Goldstein Histogram
    sns.histplot(df["GoldsteinScale"], bins=30, kde=True, ax=axes[0], color="purple")
    axes[0].set_title("Distribution of Goldstein Scale", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Scale (-10 Conflict to +10 Cooperation)")
    axes[0].axvline(0, color="black", linestyle="--")

    # AvgTone Histogram
    sns.histplot(df["AvgTone"], bins=30, kde=True, ax=axes[1], color="teal")
    axes[1].set_title("Distribution of Average Tone", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Tone (Negative to Positive)")
    axes[1].axvline(0, color="black", linestyle="--")

    plt.suptitle("Intensity & Sentiment Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_gdelt_intensity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 04_gdelt_intensity.png")

def generate_report(df):
    """Generate markdown report matching the teammate's style."""
    print("Generating Markdown report...")

    # Calculate statistics
    total_events = len(df)
    years_span = f"{df['Year'].min()} - {df['Year'].max()}"
    avg_goldstein = df["GoldsteinScale"].mean()
    avg_tone = df["AvgTone"].mean()
    
    # Regional/Actor stats
    ven_initiated = len(df[df["Initiator"] == "Venezuela"])
    usa_initiated = len(df[df["Initiator"] == "USA"])
    
    # Top Peak Months
    peak_months = df.groupby("Month").size().sort_values(ascending=False).head(10)

    # Top Intensity Events (Most Negative Goldstein)
    top_conflict = df.sort_values("GoldsteinScale").head(5)
    # Top Coop Events
    top_coop = df.sort_values("GoldsteinScale", ascending=False).head(5)

    report = f"""# Venezuela-US GDELT Analysis Report

## Overview

| Metric | Value |
|--------|-------|
| **Data Period** | {df['DateObject'].min().date()} ~ {df['DateObject'].max().date()} |
| **Total Events** | {total_events:,} |
| **Avg Goldstein Scale** | {avg_goldstein:.2f} |
| **Avg Tone** | {avg_tone:.2f} |
| **Initiator Split** | 🇻🇪 {ven_initiated:,} / 🇺🇸 {usa_initiated:,} |

### Data Source
- **Dataset**: GDELT (Global Database of Events, Language, and Tone)
- **Filters**: Actor1/Actor2 filtered for VEN and USA interactions.

---

## Timeline Analysis

### Full Timeline ({years_span})

![Timeline](01_gdelt_timeline.png)

### Key Insights
- **Volume**: Spikes in event volume often correlate with major political crises.
- **Stability**: The Goldstein Scale (bottom chart) typically dips into negative territory (red zones) during protest years and sanction implementations.

### Top 10 Peak Activity Months

| Month | Events |
|-------|-------|
"""
    for month, count in peak_months.items():
        report += f"| {month} | {count:,} |\n"

    report += f"""
---

## Yearly Trends

![Yearly Stats](02_gdelt_yearly_stats.png)

### Summary
- **Activity**: Yearly volume indicates the intensity of diplomatic/conflictual engagement.
- **Tone Distribution**: Box plots show the within-year spread and outliers in media tone.

### Smoothed Tone Trend

![Tone Trend](05_gdelt_tone_trend.png)

- **Rolling Mean**: Highlights medium-term shifts in sentiment trajectory.
- **Volatility Band (±1 SD)**: Captures periods of higher/lower tone dispersion.

---

## Event Categories (QuadClass)

![Categories](03_gdelt_categories.png)

### Categories Defined
- **Verbal Cooperation**: Statements of support, negotiation, promises.
- **Material Cooperation**: Economic aid, agreements, visits.
- **Verbal Conflict**: Threats, demands, disapproval.
- **Material Conflict**: Sanctions, protests, military acts.

---

## Intensity & Sentiment

![Intensity](04_gdelt_intensity.png)

### Metric Distributions
- **Goldstein Scale**: Measures the theoretical impact of an event on stability. Range: -10 (Military Attack) to +10 (Military Assistance).
- **AvgTone**: Measures the tone of the news coverage surrounding the event.

---

## Extreme Events

### Top Conflict Events (Lowest Goldstein)
| Date | Actor 1 | Actor 2 | Code | Goldstein | Source |
|------|---------|---------|------|-----------|--------|
"""
    for _, row in top_conflict.iterrows():
        url_short = str(row['SourceURL'])[:40] + "..." if isinstance(row['SourceURL'], str) else "N/A"
        report += f"| {row['Date']} | {row['Actor1Name']} | {row['Actor2Name']} | {row['EventCode']} | {row['GoldsteinScale']} | {url_short} |\n"

    report += f"""
### Top Cooperation Events (Highest Goldstein)
| Date | Actor 1 | Actor 2 | Code | Goldstein | Source |
|------|---------|---------|------|-----------|--------|
"""
    for _, row in top_coop.iterrows():
        url_short = str(row['SourceURL'])[:40] + "..." if isinstance(row['SourceURL'], str) else "N/A"
        report += f"| {row['Date']} | {row['Actor1Name']} | {row['Actor2Name']} | {row['EventCode']} | {row['GoldsteinScale']} | {url_short} |\n"

    report += f"""
---

*Generated: {datetime.now().strftime('%Y-%m-%d')}*
"""

    with open(OUTPUT_DIR / "GDELT_EDA_Report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("  Saved GDELT_EDA_Report.md")

def main():
    print("=" * 60)
    print("EDA for GDELT Venezuela-US Data")
    print("=" * 60)

    # 1. Load Data
    df = load_data(DATA_PATH)
    
    if df is not None:
        # 2. Generate Plots
        plot_timeline(df)
        plot_yearly_distribution(df)
        plot_quadclass_distribution(df)
        plot_intensity_metrics(df)
        plot_tone_trend(df)
        
        # 3. Generate Report
        generate_report(df)

    print("=" * 60)
    print("EDA Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
