"""
Analysis functions for statistics, NLP, and report generation.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import CRISIS_PERIODS, PipelineConfig, TOPIC_CATEGORIES
from .processors import get_stopwords, identify_crisis_posts, preprocess_text


def ensure_nltk_resources():
    """Download required NLTK resources if not present."""
    import nltk

    resources = ["punkt", "punkt_tab", "stopwords"]
    for resource in resources:
        try:
            if "punkt" in resource:
                nltk.data.find(f"tokenizers/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


def quick_stats(df: pd.DataFrame, comments_df: Optional[pd.DataFrame] = None) -> None:
    """Print quick statistics."""
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total Submissions: {len(df):,}")
    if comments_df is not None and len(comments_df) > 0:
        print(f"Total Comments: {len(comments_df):,}")
    print(f"Subreddits: {df['subreddit'].nunique() if 'subreddit' in df.columns else 'N/A'}")
    print(f"Unique Authors (Submissions): {df['author'].nunique() if 'author' in df.columns else 'N/A':,}")
    if comments_df is not None and len(comments_df) > 0 and "author" in comments_df.columns:
        print(f"Unique Authors (Comments): {comments_df['author'].nunique():,}")
    if "created_datetime" in df.columns:
        print(f"Date Range: {df['created_datetime'].min().date()} to {df['created_datetime'].max().date()}")
    if "score" in df.columns:
        print(f"Avg Score: {df['score'].mean():.1f}")
    if "num_comments" in df.columns:
        print(f"Avg Comments: {df['num_comments'].mean():.1f}")
    if "total_engagement" in df.columns:
        print(f"Total Engagement (Score + Comments): {df['total_engagement'].sum():,}")
    print(f"{'='*60}")


def compute_keyword_coverage(df: pd.DataFrame) -> pd.Series:
    """Compute keyword coverage across posts."""
    key_terms = {
        "India-Pakistan": ["india", "pakistan"],
        "Kashmir": ["kashmir", "j&k", "jammu"],
        "Pulwama": ["pulwama", "crpf"],
        "Balakot": ["balakot", "airstrike", "surgical"],
        "Article 370": ["article 370", "370", "special status"],
        "Nuclear": ["nuclear", "atomic", "nuke"],
        "Military": ["army", "military", "defense"],
        "Terror": ["terror", "terrorist", "attack"],
        "Ceasefire": ["ceasefire", "peace"],
        "Op Sindoor": ["sindoor", "pahalgam"],
    }

    coverage = {}
    titles_lower = df["title"].fillna("").str.lower()

    for term_group, terms in key_terms.items():
        pattern = "|".join(terms)
        mask = titles_lower.str.contains(pattern, regex=True, na=False)
        coverage[term_group] = mask.sum()

    return pd.Series(coverage).sort_values(ascending=False)


def analyze_engagement(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze engagement metrics."""
    results = {}

    if "score" in df.columns:
        results["score"] = {
            "mean": df["score"].mean(),
            "median": df["score"].median(),
            "max": df["score"].max(),
            "percentile_95": df["score"].quantile(0.95),
        }

    if "num_comments" in df.columns:
        results["comments"] = {
            "mean": df["num_comments"].mean(),
            "median": df["num_comments"].median(),
            "total": df["num_comments"].sum(),
            "max": df["num_comments"].max(),
        }

    return results


def analyze_authors(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze author statistics."""
    if "author" not in df.columns:
        return {}

    df_authors = df[df["author"] != "[deleted]"]
    author_posts = df_authors.groupby("author").size()

    return {
        "total_unique": df["author"].nunique(),
        "deleted_count": (df["author"] == "[deleted]").sum(),
        "top_authors": author_posts.nlargest(15).to_dict(),
        "posts_per_author": {
            "mean": author_posts.mean(),
            "median": author_posts.median(),
            "max": author_posts.max(),
        },
    }


def analyze_temporal(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze temporal distribution."""
    results = {}

    if "year" in df.columns:
        results["yearly"] = df.groupby("year").size().to_dict()

    if "month" in df.columns and "year" in df.columns:
        results["monthly"] = df.groupby(["year", "month"]).size().to_dict()

    if "dayofweek" in df.columns:
        results["by_dayofweek"] = df.groupby("dayofweek").size().to_dict()

    if "hour" in df.columns:
        results["by_hour"] = df.groupby("hour").size().to_dict()

    return results


def analyze_subreddits(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze subreddit distribution."""
    if "subreddit" not in df.columns:
        return {}

    subreddit_counts = df["subreddit"].value_counts()

    results = {
        "distribution": subreddit_counts.to_dict(),
        "total_subreddits": df["subreddit"].nunique(),
    }

    if "score" in df.columns:
        results["avg_score_by_sub"] = df.groupby("subreddit")["score"].mean().to_dict()

    if "num_comments" in df.columns:
        results["avg_comments_by_sub"] = (
            df.groupby("subreddit")["num_comments"].mean().to_dict()
        )

    return results


def compute_ngrams(
    tokens: List[str], n: int = 2, top_k: int = 20
) -> List[Tuple[tuple, int]]:
    """Compute n-grams from tokens."""
    try:
        from nltk.util import ngrams

        n_grams = list(ngrams(tokens, n))
        return Counter(n_grams).most_common(top_k)
    except ImportError:
        return []


def analyze_text_content(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform NLP analysis on text content."""
    try:
        ensure_nltk_resources()
        from nltk.tokenize import word_tokenize
    except ImportError:
        print("NLTK not available. Skipping text analysis.")
        return {}

    df["clean_text"] = df["title"].fillna("").apply(preprocess_text)
    stop_words = get_stopwords()

    all_text = " ".join(df["clean_text"].tolist())
    tokens = word_tokenize(all_text)
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    # Enhanced stopwords for n-gram analysis
    ngram_stopwords = stop_words | {
        "india", "pakistan", "indian", "pakistani", "indias", "pakistans", "pak",
        "says", "said", "will", "would", "could", "may", "also", "one", "two",
        "first", "new", "last", "year", "years", "day", "days", "today", "now",
    }
    filtered_tokens_enhanced = [w for w in tokens if w not in ngram_stopwords and len(w) > 2]

    unigram_freq = Counter(filtered_tokens_enhanced).most_common(20)
    bigram_freq = compute_ngrams(filtered_tokens_enhanced, n=2, top_k=20)
    trigram_freq = compute_ngrams(filtered_tokens_enhanced, n=3, top_k=10)

    return {
        "total_tokens": len(tokens),
        "filtered_tokens": len(filtered_tokens),
        "filtered_tokens_enhanced": len(filtered_tokens_enhanced),
        "unique_vocab": len(set(filtered_tokens_enhanced)),
        "lexical_diversity": len(set(filtered_tokens_enhanced)) / len(filtered_tokens_enhanced) * 100
        if filtered_tokens_enhanced
        else 0,
        "top_unigrams": unigram_freq,
        "top_bigrams": bigram_freq,
        "top_trigrams": trigram_freq,
        "tokens": tokens,
        "filtered_tokens_enhanced": filtered_tokens_enhanced,
    }


def generate_summary_report(
    df: pd.DataFrame,
    config: PipelineConfig,
    comments_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> str:
    """Generate comprehensive summary report."""
    df_crisis = identify_crisis_posts(df, CRISIS_PERIODS)
    coverage_df = compute_keyword_coverage(df)

    # Compute subreddit stats
    sub_stats = df.groupby("subreddit").agg({
        "_id": "count",
        "score": ["mean", "median", "sum"],
        "num_comments": ["mean", "sum"],
        "author": "nunique"
    }).round(1)
    sub_stats.columns = [
        "posts", "avg_score", "med_score", "total_score",
        "avg_comments", "total_comments", "unique_authors"
    ]
    sub_stats = sub_stats.sort_values("posts", ascending=False)

    # NLP analysis
    try:
        ensure_nltk_resources()
        from nltk.tokenize import word_tokenize

        df["clean_text"] = df["title"].fillna("").apply(preprocess_text)
        stop_words = get_stopwords()
        ngram_stopwords = stop_words | {
            "india", "pakistan", "indian", "pakistani", "indias", "pakistans", "pak",
            "says", "said", "will", "would", "could", "may", "also", "one", "two",
            "first", "new", "last", "year", "years", "day", "days", "today", "now",
        }
        all_text = " ".join(df["clean_text"].tolist())
        tokens = word_tokenize(all_text)
        filtered_tokens_enhanced = [w for w in tokens if w not in ngram_stopwords and len(w) > 2]
        unigram_freq = Counter(filtered_tokens_enhanced).most_common(5)
        bigram_freq = compute_ngrams(filtered_tokens_enhanced, n=2, top_k=5)
    except (ImportError, LookupError) as e:
        print(f"NLTK not available or resources missing: {e}")
        tokens = []
        filtered_tokens_enhanced = []
        unigram_freq = []
        bigram_freq = []

    comments_count = len(comments_df) if comments_df is not None and len(comments_df) > 0 else 0

    report = f"""
================================================================================
RESEARCH SUMMARY REPORT: INDIA-PAKISTAN CONFLICT DISCOURSE ON REDDIT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATASET OVERVIEW
--------------------------------------------------------------------------------
Total Submissions: {len(df):,}
Total Comments: {comments_count:,}
Date Range: {df['created_datetime'].min().date()} to {df['created_datetime'].max().date()}
Time Span: {(df['created_datetime'].max() - df['created_datetime'].min()).days:,} days
Subreddits Analyzed: {df['subreddit'].nunique()}
Unique Authors: {df['author'].nunique():,}
Deleted Authors: {(df['author'] == '[deleted]').sum():,}

2. SUBREDDIT DISTRIBUTION
--------------------------------------------------------------------------------
"""

    for sub in sub_stats.index:
        count = sub_stats.loc[sub, "posts"]
        pct = count / len(df) * 100
        report += f"  {sub:20s}: {count:>8,.0f} posts ({pct:>5.1f}%)\n"

    report += f"""
3. ENGAGEMENT METRICS
--------------------------------------------------------------------------------
Score Statistics:
  Mean:   {df['score'].mean():>10.1f}
  Median: {df['score'].median():>10.1f}
  Max:    {df['score'].max():>10,}
  95th %: {df['score'].quantile(0.95):>10.1f}

Comment Statistics:
  Mean:   {df['num_comments'].mean():>10.1f}
  Median: {df['num_comments'].median():>10.1f}
  Total:  {df['num_comments'].sum():>10,}
  Max:    {df['num_comments'].max():>10,}

4. TEMPORAL DISTRIBUTION
--------------------------------------------------------------------------------
"""

    yearly = df.groupby("year").size()
    for year, count in yearly.items():
        pct = count / len(df) * 100
        report += f"  {year}: {count:>6,} posts ({pct:>5.1f}%)\n"

    report += f"""
5. CRISIS PERIOD COVERAGE
--------------------------------------------------------------------------------
"""

    crisis_counts = df_crisis[df_crisis["crisis_period"] != "Normal"]["crisis_period"].value_counts()
    for crisis, count in crisis_counts.items():
        report += f"  {crisis:25s}: {count:>5,} posts\n"

    report += f"""
6. KEYWORD COVERAGE
--------------------------------------------------------------------------------
"""

    for term, count in coverage_df.sort_values(ascending=False).items():
        pct = count / len(df) * 100
        report += f"  {term:20s}: {count:>6,} posts ({pct:>5.1f}%)\n"

    report += f"""
7. NLP ANALYSIS SUMMARY
--------------------------------------------------------------------------------
Total Tokens: {len(tokens):,}
Filtered Tokens: {len(filtered_tokens_enhanced):,}
Unique Vocabulary: {len(set(filtered_tokens_enhanced)):,}
Lexical Diversity: {len(set(filtered_tokens_enhanced))/len(filtered_tokens_enhanced)*100 if filtered_tokens_enhanced else 0:.2f}%

Top 5 Unigrams:
"""

    for word, count in unigram_freq[:5]:
        report += f"  {word:20s}: {count:,}\n"

    report += f"""
Top 5 Bigrams:
"""

    for bigram, count in bigram_freq[:5]:
        report += f"  {bigram[0]} {bigram[1]:15s}: {count:,}\n"

    report += """
================================================================================
END OF REPORT
================================================================================
"""

    if save:
        config.analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(config.analysis_dir / "summary_report.txt", "w") as f:
            f.write(report)
        print(f"Saved: {config.analysis_dir / 'summary_report.txt'}")

    return report