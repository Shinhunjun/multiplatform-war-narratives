"""
RoBERTa-based sentiment analyzer with ID tracking.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Lazy imports for models
_pipeline = None
_tokenizer = None


def _load_model(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """Lazy load the sentiment model."""
    global _pipeline, _tokenizer
    if _pipeline is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Use MPS (Metal) on Apple Silicon if available
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = 0
        else:
            device = -1

        _pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=_tokenizer,
            max_length=512,
            truncation=True,
            device=device,
        )
        print(f"Sentiment model using device: {device}")
    return _pipeline


def analyze_sentiment_batch(
    texts: List[str],
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 32,
) -> List[Dict]:
    """
    Analyze sentiment for a batch of texts.

    Returns list of dicts with:
    - label: 'positive', 'negative', 'neutral'
    - score: confidence score
    - sentiment_score: -1 to 1 scale
    """
    pipe = _load_model(model_name)

    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment analysis"):
        batch = texts[i : i + batch_size]
        # Truncate long texts
        batch = [t[:512] if len(t) > 512 else t for t in batch]
        batch = [t if t.strip() else "neutral" for t in batch]  # Handle empty

        try:
            batch_results = pipe(batch)
            for res in batch_results:
                label = res["label"].lower()
                score = res["score"]

                # Convert to -1 to 1 scale
                if label == "positive":
                    sentiment_score = score
                elif label == "negative":
                    sentiment_score = -score
                else:
                    sentiment_score = 0.0

                results.append({
                    "label": label,
                    "confidence": score,
                    "sentiment_score": sentiment_score,
                })
        except Exception as e:
            # Fallback for errors
            for _ in batch:
                results.append({
                    "label": "neutral",
                    "confidence": 0.0,
                    "sentiment_score": 0.0,
                })

    return results


def analyze_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Add sentiment columns to DataFrame.

    Adds columns:
    - sentiment_label: positive/negative/neutral
    - sentiment_confidence: model confidence
    - sentiment_score: -1 to 1 scale
    """
    df = df.copy()
    texts = df[text_column].fillna("").tolist()

    print(f"Analyzing sentiment for {len(texts):,} texts...")
    results = analyze_sentiment_batch(texts, model_name, batch_size)

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_confidence"] = [r["confidence"] for r in results]
    df["sentiment_score"] = [r["sentiment_score"] for r in results]

    return df


def aggregate_sentiment(
    df: pd.DataFrame,
    group_by: List[str] = ["subreddit", "year_month"],
) -> pd.DataFrame:
    """
    Aggregate sentiment by groups (e.g., subreddit, time period).

    Returns DataFrame with:
    - mean_sentiment: average sentiment score
    - positive_ratio: fraction of positive posts
    - negative_ratio: fraction of negative posts
    - count: number of posts
    """
    agg = df.groupby(group_by).agg(
        mean_sentiment=("sentiment_score", "mean"),
        std_sentiment=("sentiment_score", "std"),
        positive_count=("sentiment_label", lambda x: (x == "positive").sum()),
        negative_count=("sentiment_label", lambda x: (x == "negative").sum()),
        neutral_count=("sentiment_label", lambda x: (x == "neutral").sum()),
        total_count=("sentiment_label", "count"),
    ).reset_index()

    agg["positive_ratio"] = agg["positive_count"] / agg["total_count"]
    agg["negative_ratio"] = agg["negative_count"] / agg["total_count"]
    agg["neutral_ratio"] = agg["neutral_count"] / agg["total_count"]

    return agg


def get_sentiment_summary(df: pd.DataFrame) -> Dict:
    """Get overall sentiment summary statistics."""
    return {
        "total_records": len(df),
        "mean_sentiment": df["sentiment_score"].mean(),
        "std_sentiment": df["sentiment_score"].std(),
        "positive_count": (df["sentiment_label"] == "positive").sum(),
        "negative_count": (df["sentiment_label"] == "negative").sum(),
        "neutral_count": (df["sentiment_label"] == "neutral").sum(),
        "positive_ratio": (df["sentiment_label"] == "positive").mean(),
        "negative_ratio": (df["sentiment_label"] == "negative").mean(),
    }
