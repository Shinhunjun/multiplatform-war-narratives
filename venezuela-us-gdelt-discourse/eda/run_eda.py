"""
Comprehensive EDA script for GDELT Venezuela-US data.
Combines event-level analysis (timeline, categories, tone/conflict metrics)
with scrape-quality and content analysis (status, URL uniqueness, word clouds).
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import pandas as pd
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud

PREPROCESSING_DIR = Path(__file__).resolve().parent.parent / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from build_text_relevance_tokens import (
    CONTRACTION_FRAGMENTS,
    LETTER_TOKEN_RE,
    SPECIAL_KEEP_TOKENS,
    build_stopword_set,
    ensure_nltk_resources,
    parse_text_tokens,
    penn_to_wordnet,
)


BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_READY_DIR = BASE_DIR / "data" / "analysis_ready"
EVENTS_PATH = ANALYSIS_READY_DIR / "analysis_events.parquet"
URL_CONTENT_PATH = ANALYSIS_READY_DIR / "analysis_url_content.parquet"
OUTPUT_DIR = Path(__file__).parent

COLORS = {
    "Verbal Cooperation": "#1f77b4",
    "Material Cooperation": "#2ca02c",
    "Verbal Conflict": "#ff7f0e",
    "Material Conflict": "#d62728",
}

QUADCLASS_MAP = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict",
}

KEY_EVENTS = {
    "2014-02": "2014 Protests",
    "2017-08": "Trump Sanctions",
    "2019-01": "Guaido Crisis",
    "2024-07": "2024 Election",
    "2026-01": "Maduro Captured",
}

DOMAIN_STOPWORDS = {
    "said",
    "says",
    "say",
    "will",
    "news",
    "new",
    "reuters",
    "report",
    "reports",
    "breaking",
    "update",
}

EVENT_REQUIRED_COLUMNS = [
    "Date",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor2Name",
    "Actor2CountryCode",
    "EventCode",
    "QuadClass",
    "GoldsteinScale",
    "AvgTone",
    "SourceURL",
    "Scrape_Status",
    "url_id",
]

URL_CONTENT_REQUIRED_COLUMNS = [
    "url_id",
    "SourceURL",
    "Title",
    "Text",
    "Tokens",
]


def _load_required_parquet(
    filepath: Path,
    required_cols: list[str],
    dataset_label: str,
) -> pd.DataFrame:
    """Load a parquet dataset subset and raise a clear error when required columns are unavailable.
    
    Args:
        filepath (Path): Path to the input parquet file.
        required_cols (list[str]): Required columns for the requested dataset.
        dataset_label (str): Human-readable dataset label used in error messages.
    
    Returns:
        pd.DataFrame: Loaded pandas DataFrame.
    """
    try:
        return pd.read_parquet(filepath, columns=required_cols)
    except Exception as exc:
        raise ValueError(f"{dataset_label} missing required columns: {required_cols}") from exc


def load_data(
    events_path: Path = EVENTS_PATH,
    url_content_path: Path = URL_CONTENT_PATH,
) -> pd.DataFrame | None:
    """Load and validate analysis-ready GDELT event and URL-content parquet data.
    
    Args:
        events_path (Path): Path to the event-level parquet file.
        url_content_path (Path): Path to the URL-content parquet file.
    
    Returns:
        pd.DataFrame | None: Loaded DataFrame when available; otherwise None.
    """
    print("Loading analysis-ready GDELT parquet data...")
    missing_paths = [path for path in [events_path, url_content_path] if not path.exists()]
    if missing_paths:
        for path in missing_paths:
            print(f"File not found: {path}")
        return None

    print("  Reading event parquet into memory...")
    t0 = perf_counter()
    events = _load_required_parquet(events_path, EVENT_REQUIRED_COLUMNS, "analysis_events parquet")
    url_content = _load_required_parquet(
        url_content_path,
        URL_CONTENT_REQUIRED_COLUMNS,
        "analysis_url_content parquet",
    ).rename(
        columns={
            "SourceURL": "Content_SourceURL",
            "Title": "Content_Title",
            "Text": "Content_Text",
            "Tokens": "Content_Tokens",
        }
    )

    df = events.merge(url_content, on="url_id", how="left", validate="many_to_one")
    source_url = df["SourceURL"].fillna("").astype(str)
    fallback_url = df["Content_SourceURL"].fillna("").astype(str)
    blank_mask = source_url.str.strip().eq("")
    df["SourceURL"] = source_url.where(~blank_mask, fallback_url)
    df["Title"] = df["Content_Title"]
    df["Text"] = df["Content_Text"]
    df["Tokens"] = df["Content_Tokens"]
    df = df.drop(columns=["Content_SourceURL", "Content_Title", "Content_Text", "Content_Tokens"])

    print(f"Loaded {len(df):,} rows")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create shared derived columns used across all analyses.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
    
    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    t0 = perf_counter()
    date_str = df["Date"].astype(str).str.extract(r"(\d{8})")[0]
    df["DateObject"] = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    df["Year"] = df["DateObject"].dt.year
    df["Month"] = df["DateObject"].dt.to_period("M")

    quad_numeric = pd.to_numeric(df["QuadClass"], errors="coerce")
    df["EventCategory"] = quad_numeric.map(QUADCLASS_MAP)
    df["Initiator"] = df["Actor1CountryCode"].apply(
        lambda x: "Venezuela" if x == "VEN" else ("USA" if x == "USA" else "Other")
    )

    if df["DateObject"].notna().any():
        min_date = df["DateObject"].min().date()
        max_date = df["DateObject"].max().date()
        print(f"Date range: {min_date} to {max_date}")
    else:
        print("Warning: no valid dates parsed from Date column.")
    return df


def plot_timeline(df: pd.DataFrame) -> None:
    """Plot event timeline and monthly Goldstein mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
    
    Returns:
        None: No return value.
    """
    monthly_counts = df.groupby("Month").size()
    monthly_goldstein = df.groupby("Month")["GoldsteinScale"].mean()
    dates = [p.to_timestamp() for p in monthly_counts.index]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].fill_between(dates, monthly_counts.values, alpha=0.7, color="#1f77b4")
    axes[0].set_ylabel("Total Events", fontsize=12)
    axes[0].set_title("Venezuela-US Event Timeline (GDELT Scraped)", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    for date_str, label in KEY_EVENTS.items():
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

    axes[1].plot(dates, monthly_goldstein.values, color="#d62728", linewidth=1.5)
    axes[1].axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    axes[1].set_ylabel("Avg Goldstein Scale\n(Conflict < 0 < Coop)", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(
        dates,
        monthly_goldstein.values,
        0,
        where=(monthly_goldstein.values >= 0),
        facecolor="green",
        alpha=0.3,
    )
    axes[1].fill_between(
        dates,
        monthly_goldstein.values,
        0,
        where=(monthly_goldstein.values < 0),
        facecolor="red",
        alpha=0.3,
    )

    for date_str in KEY_EVENTS:
        date = datetime.strptime(date_str, "%Y-%m")
        axes[1].axvline(date, color="red", linestyle="--", alpha=0.5)

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_gdelt_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_yearly_distribution(df: pd.DataFrame) -> None:
    """Plot yearly event counts and AvgTone distribution by year.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
    
    Returns:
        None: No return value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    yearly_counts = df.groupby("Year").size()
    axes[0].bar(yearly_counts.index, yearly_counts.values, color="#1f77b4", alpha=0.8)
    axes[0].set_title("Total Events by Year", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3, axis="y")

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

    plt.suptitle("Yearly Activity & Tone Dispersion (GDELT Scraped)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_gdelt_yearly_stats.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_quadclass_distribution(df: pd.DataFrame) -> None:
    """Plot overall and initiator-split QuadClass distribution.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
    
    Returns:
        None: No return value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    qc_counts = df["EventCategory"].value_counts(dropna=False)
    qc_counts = qc_counts[qc_counts.index.notna()]
    colors_qc = [COLORS.get(cat, "#999999") for cat in qc_counts.index]
    axes[0].pie(qc_counts.values, labels=qc_counts.index, colors=colors_qc, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Event Categories (QuadClass)", fontsize=12, fontweight="bold")

    ct = pd.crosstab(df["Initiator"], df["EventCategory"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_norm = ct_norm.loc[ct_norm.index.isin(["Venezuela", "USA"])]
    ct_norm.plot(kind="bar", stacked=True, ax=axes[1], color=[COLORS.get(c, "#999999") for c in ct_norm.columns])
    axes[1].set_title("Event Category Mix by Initiator", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("Initiator")
    axes[1].legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_gdelt_categories.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_intensity_metrics(df: pd.DataFrame) -> None:
    """Plot GoldsteinScale and AvgTone histograms.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
    
    Returns:
        None: No return value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(df["GoldsteinScale"], bins=30, kde=True, ax=axes[0], color="purple")
    axes[0].set_title("Distribution of Goldstein Scale", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Scale (-10 Conflict to +10 Cooperation)")
    axes[0].axvline(0, color="black", linestyle="--")

    sns.histplot(df["AvgTone"], bins=30, kde=True, ax=axes[1], color="teal")
    axes[1].set_title("Distribution of Average Tone", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Tone (Negative to Positive)")
    axes[1].axvline(0, color="black", linestyle="--")

    plt.suptitle("Intensity & Sentiment Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_gdelt_intensity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_tone_trend(df: pd.DataFrame, rolling_window: int = 12) -> None:
    """Plot monthly AvgTone with rolling mean and +/-1 SD band.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
        rolling_window (int): Window size used for rolling trend smoothing. Defaults to 12.
    
    Returns:
        None: No return value.
    """
    monthly_tone = (
        df.dropna(subset=["DateObject", "AvgTone"]).set_index("DateObject")["AvgTone"].resample("ME").mean()
    )

    min_periods = max(3, rolling_window // 3)
    rolling_mean = monthly_tone.rolling(window=rolling_window, min_periods=min_periods).mean()
    rolling_std = monthly_tone.rolling(window=rolling_window, min_periods=min_periods).std()
    upper = rolling_mean + rolling_std
    lower = rolling_mean - rolling_std

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(monthly_tone.index, monthly_tone.values, color="#9aa3ad", alpha=0.45, linewidth=1, label="Monthly Mean Tone")
    ax.plot(rolling_mean.index, rolling_mean.values, color="#1f77b4", linewidth=2, label=f"{rolling_window}-Month Rolling Mean")
    ax.fill_between(rolling_mean.index, lower.values, upper.values, color="#1f77b4", alpha=0.2, label="Rolling ±1 SD")
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


def plot_scrape_status(df: pd.DataFrame) -> dict[str, float]:
    """Plot scrape status counts and return scrape success metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
    
    Returns:
        dict[str, float]: Dictionary containing computed values.
    """
    status_counts = df["Scrape_Status"].fillna("Missing").value_counts()
    total = len(df)
    success_mask = df["Scrape_Status"].fillna("").str.contains("success", case=False)
    success_count = int(success_mask.sum())
    success_rate = (success_count / total * 100.0) if total else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=status_counts.index,
        y=status_counts.values,
        hue=status_counts.index,
        palette="Blues_r",
        legend=False,
        ax=ax,
    )
    ax.set_title("Scrape Status Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scrape Status")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)

    for i, v in enumerate(status_counts.values):
        pct = (v / total * 100.0) if total else 0.0
        ax.text(i, v, f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.text(
        0.02,
        0.98,
        f"Success rate (all success statuses): {success_rate:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_scraped_status.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "total_rows": total,
        "success_count": success_count,
        "success_rate": success_rate,
    }


def plot_url_uniqueness(df: pd.DataFrame) -> dict[str, int]:
    """Plot unique vs duplicate URL rows and return URL metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
    
    Returns:
        dict[str, int]: Dictionary containing computed values.
    """
    url_series = df["SourceURL"].fillna("").astype(str).str.strip()
    valid_urls = url_series[url_series != ""]

    total_valid = int(valid_urls.shape[0])
    unique_count = int(valid_urls.nunique())
    duplicate_rows = int(total_valid - unique_count)

    labels = ["Unique URLs", "Duplicate URL Rows"]
    values = [unique_count, duplicate_rows]
    colors = ["#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_title("URL Uniqueness in Scraped Dataset", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, v in zip(bars, values):
        pct = (v / total_valid * 100.0) if total_valid else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_scraped_url_uniqueness.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "valid_url_rows": total_valid,
        "unique_urls": unique_count,
        "duplicate_url_rows": duplicate_rows,
    }


def _token_counter(series: pd.Series) -> Counter:
    """Build token counts using the shared NLTK normalization pipeline.
    
    Args:
        series (pd.Series): Input pandas Series.
    
    Returns:
        Counter: Token frequency counter.
    """
    ensure_nltk_resources()
    stopwords = build_stopword_set() | DOMAIN_STOPWORDS
    counts: Counter = Counter()
    lemmatizer = WordNetLemmatizer()

    for value in series.dropna().astype(str):
        parsed_tokens = parse_text_tokens(value)
        if not parsed_tokens:
            continue

        special_tokens = [tok for tok in parsed_tokens if tok in SPECIAL_KEEP_TOKENS]
        lexical_tokens = [
            tok
            for tok in parsed_tokens
            if tok not in SPECIAL_KEEP_TOKENS and LETTER_TOKEN_RE.fullmatch(tok)
        ]

        if lexical_tokens:
            for token, pos in pos_tag(lexical_tokens):
                lemma = lemmatizer.lemmatize(token, pos=penn_to_wordnet(pos)).strip("'")
                if not lemma:
                    continue
                if lemma in CONTRACTION_FRAGMENTS:
                    continue
                if lemma.isdigit():
                    continue
                if lemma in stopwords and lemma not in SPECIAL_KEEP_TOKENS:
                    continue
                counts[lemma] += 1

        for token in special_tokens:
            if token not in stopwords or token in SPECIAL_KEEP_TOKENS:
                counts[token] += 1
    return counts


def _token_counter_from_precomputed(series: pd.Series) -> Counter:
    """Build token counts from precomputed token lists stored in parquet.
    
    Args:
        series (pd.Series): Series of token-list values.
    
    Returns:
        Counter: Token frequency counter.
    """
    counts: Counter = Counter()
    for value in series:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if hasattr(value, "tolist"):
            tokens = value.tolist()
        elif isinstance(value, (list, tuple)):
            tokens = list(value)
        else:
            tokens = []

        for token in tokens:
            token_text = str(token).strip()
            if token_text:
                counts[token_text] += 1
    return counts


def make_wordcloud_from_counts(
    counts: Counter, chart_title: str, out_name: str, max_words: int = 300
) -> None:
    """Generate and save a word cloud image.
    
    Args:
        counts (Counter): Token frequency counter.
        chart_title (str): Display title for the generated chart.
        out_name (str): Output filename for the generated chart.
        max_words (int): Maximum number of words in the word cloud. Defaults to 300.
    
    Returns:
        None: No return value.
    """
    if not counts:
        return

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        stopwords=set(STOPWORDS) | DOMAIN_STOPWORDS,
        collocations=False,
        max_words=max_words,
    ).generate_from_frequencies(dict(counts.most_common(10000)))

    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(chart_title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close()


def top_words_with_share(counts: Counter, top_n: int = 10) -> list[tuple[str, int, float]]:
    """Compute top token counts and relative share percentages from a token-frequency counter.
    
    Args:
        counts (Counter): Token frequency counter.
        top_n (int): Number of top items to include. Defaults to 10.
    
    Returns:
        list[tuple[str, int, float]]: List result produced by this function.
    """
    total_tokens = sum(counts.values())
    return [
        (word, freq, (freq / total_tokens * 100.0) if total_tokens else 0.0)
        for word, freq in counts.most_common(top_n)
    ]


def generate_report(
    df: pd.DataFrame,
    scrape_metrics: dict[str, float],
    url_metrics: dict[str, int],
    top_title_words: list[tuple[str, int, float]],
    top_text_words: list[tuple[str, int, float]],
) -> None:
    """Generate comprehensive markdown report.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
        scrape_metrics (dict[str, float]): Scrape-quality metrics dictionary.
        url_metrics (dict[str, int]): URL-uniqueness metrics dictionary.
        top_title_words (list[tuple[str, int, float]]): Top title token statistics used in reporting.
        top_text_words (list[tuple[str, int, float]]): Top body-text token statistics used in reporting.
    
    Returns:
        None: No return value.
    """
    total_events = len(df)
    avg_goldstein = df["GoldsteinScale"].mean()
    avg_tone = df["AvgTone"].mean()
    median_tone = df["AvgTone"].median()
    ven_initiated = int((df["Initiator"] == "Venezuela").sum())
    usa_initiated = int((df["Initiator"] == "USA").sum())
    years_span = f"{int(df['Year'].min())} - {int(df['Year'].max())}"
    status_counts = df["Scrape_Status"].fillna("Missing").value_counts()
    peak_months = df.groupby("Month").size().sort_values(ascending=False).head(10)

    title_series = df["Title"].fillna("").astype(str).str.strip()
    has_title_mask = title_series != ""
    df_with_titles = df[has_title_mask].copy()

    top_conflict = df_with_titles.sort_values("GoldsteinScale", ascending=True).head(5)
    top_coop = df_with_titles.sort_values("GoldsteinScale", ascending=False).head(5)

    report = f"""# Venezuela-US GDELT Comprehensive Analysis Report

## Overview

| Metric | Value |
|--------|-------|
| **Data Period** | {df['DateObject'].min().date()} ~ {df['DateObject'].max().date()} |
| **Total Events** | {total_events:,} |
| **Avg Goldstein Scale** | {avg_goldstein:.2f} |
| **Avg Tone** | {avg_tone:.2f} |
| **Median Tone** | {median_tone:.2f} |
| **Initiator Split (VEN / USA)** | {ven_initiated:,} / {usa_initiated:,} |
| **Successful Scrapes** | {scrape_metrics["success_count"]:,} |
| **Scrape Success Rate** | {scrape_metrics["success_rate"]:.2f}% |
| **Unique URLs** | {url_metrics["unique_urls"]:,} |
| **Duplicate URL Rows** | {url_metrics["duplicate_url_rows"]:,} |

### Data Source
- **Dataset**: Analysis-ready GDELT parquet join (Venezuela-US filtered interactions)
- **Scope**: Event metadata from `analysis_events.parquet` + scraped article title/text content from `analysis_url_content.parquet`

---

## Timeline Analysis

### Full Timeline ({years_span})

![Timeline](01_gdelt_timeline.png)

### Key Insights
- **Volume**: Spikes in event volume correlate with major political milestones.
- **Stability**: Monthly Goldstein means moving below zero indicate more conflict-heavy periods.

### Top 10 Peak Activity Months

| Month | Events |
|-------|--------|
"""
    for month, count in peak_months.items():
        report += f"| {month} | {count:,} |\n"

    report += """

---

## Yearly Trends

![Yearly Stats](02_gdelt_yearly_stats.png)

### Summary
- **Activity**: Event volume by year captures macro-level intensity of interaction.
- **Tone Distribution**: Box plots show within-year dispersion and outliers in media tone.

### Smoothed Tone Trend

![Tone Trend](05_gdelt_tone_trend.png)

- **Rolling Mean**: Highlights medium-term direction shifts.
- **Volatility Band (±1 SD)**: Shows periods of high/low tone variability.

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
- **Goldstein Scale**: Event impact on stability from conflict (-) to cooperation (+).
- **AvgTone**: Sentiment proxy of related coverage.

---

## Extreme Events

### Top Conflict Events (Lowest Goldstein)
| Date | Actor 1 | Actor 2 | Code | Goldstein | Title |
|------|---------|---------|------|-----------|-------|
"""
    for _, row in top_conflict.iterrows():
        title = str(row["Title"]).strip() if pd.notna(row.get("Title")) else ""
        title_label = "Not found" if not title else (title[:100] + "..." if len(title) > 100 else title)
        date_label = row["DateObject"].date() if pd.notna(row["DateObject"]) else row["Date"]
        report += (
            f"| {date_label} | {row['Actor1Name']} | {row['Actor2Name']} | "
            f"{row['EventCode']} | {row['GoldsteinScale']} | {title_label} |\n"
        )

    report += """

### Top Cooperation Events (Highest Goldstein)
| Date | Actor 1 | Actor 2 | Code | Goldstein | Title |
|------|---------|---------|------|-----------|-------|
"""
    for _, row in top_coop.iterrows():
        title = str(row["Title"]).strip() if pd.notna(row.get("Title")) else ""
        title_label = "Not found" if not title else (title[:100] + "..." if len(title) > 100 else title)
        date_label = row["DateObject"].date() if pd.notna(row["DateObject"]) else row["Date"]
        report += (
            f"| {date_label} | {row['Actor1Name']} | {row['Actor2Name']} | "
            f"{row['EventCode']} | {row['GoldsteinScale']} | {title_label} |\n"
        )

    report += """

---

## Scrape Quality

![Scrape Status](06_scraped_status.png)

### Scrape Status Breakdown

| Status | Count |
|--------|-------|
"""
    for status, count in status_counts.items():
        report += f"| {status} | {count:,} |\n"

    report += """

### URL Uniqueness

![URL Uniqueness](07_scraped_url_uniqueness.png)

---

## Content Analysis: Title

![Title Word Cloud](08_title_wordcloud.png)

### Top Title Terms

| Word | Frequency | Share |
|------|-----------|-------|
"""
    for word, freq, share in top_title_words:
        report += f"| {word} | {freq:,} | {share:.2f}% |\n"

    report += """

## Content Analysis: Text

![Text Word Cloud](09_text_wordcloud.png)

### Top Text Terms

| Word | Frequency | Share |
|------|-----------|-------|
"""
    for word, freq, share in top_text_words:
        report += f"| {word} | {freq:,} | {share:.2f}% |\n"

    report += f"""

---

*Generated: {datetime.now().strftime('%Y-%m-%d')}*
"""

    report_path = OUTPUT_DIR / "GDELT_EDA_Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)


def main() -> None:
    """Run the script entry point.
    
    Returns:
        None: No return value.
    """
    print("=" * 60)
    print("Comprehensive EDA for GDELT Venezuela-US data")
    print("=" * 60)

    t_all = perf_counter()
    df = load_data()
    if df is None:
        return
    df = preprocess_data(df)

    # Event-level EDA
    plot_timeline(df)
    plot_yearly_distribution(df)
    plot_quadclass_distribution(df)
    plot_intensity_metrics(df)
    plot_tone_trend(df)

    # Scrape/content EDA
    scrape_metrics = plot_scrape_status(df)
    url_metrics = plot_url_uniqueness(df)

    print("Counting tokens for Title/Text fields...")
    title_counts = _token_counter(df["Title"])
    text_counts = _token_counter_from_precomputed(df["Tokens"])

    make_wordcloud_from_counts(title_counts, "Title Word Cloud", "08_title_wordcloud.png")
    make_wordcloud_from_counts(text_counts, "Text Word Cloud", "09_text_wordcloud.png")

    top_title_words = top_words_with_share(title_counts, top_n=10)
    top_text_words = top_words_with_share(text_counts, top_n=10)
    generate_report(df, scrape_metrics, url_metrics, top_title_words, top_text_words)

    print("=" * 60)
    print("Comprehensive EDA complete")
    print(f"Total runtime: {perf_counter() - t_all:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
