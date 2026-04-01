"""
Cluster summarizer using LLM to generate human-readable descriptions.
"""

import json
from typing import Dict, List, Optional

import pandas as pd


def sample_cluster_texts(
    df: pd.DataFrame,
    cluster_id: int,
    n_samples: int = 20,
    text_column: str = "text",
    random_state: int = 42,
) -> List[str]:
    """
    Sample representative texts from a cluster.

    Prioritizes high-probability cluster members.
    """
    cluster_df = df[df["cluster_id"] == cluster_id].copy()

    if len(cluster_df) == 0:
        return []

    # Sort by cluster probability if available
    if "cluster_prob" in cluster_df.columns:
        cluster_df = cluster_df.sort_values("cluster_prob", ascending=False)

    # Take top samples + random samples
    n_top = min(n_samples // 2, len(cluster_df))
    n_random = min(n_samples - n_top, len(cluster_df) - n_top)

    top_samples = cluster_df.head(n_top)[text_column].tolist()

    if n_random > 0:
        remaining = cluster_df.iloc[n_top:]
        random_samples = remaining.sample(n=n_random, random_state=random_state)[text_column].tolist()
    else:
        random_samples = []

    return top_samples + random_samples


def format_samples_for_prompt(
    texts: List[str],
    max_chars_per_text: int = 500,
    max_total_chars: int = 8000,
) -> str:
    """Format text samples for LLM prompt."""
    formatted = []
    total_chars = 0

    for i, text in enumerate(texts, 1):
        # Truncate individual text
        if len(text) > max_chars_per_text:
            text = text[:max_chars_per_text] + "..."

        entry = f"{i}. {text}"

        if total_chars + len(entry) > max_total_chars:
            break

        formatted.append(entry)
        total_chars += len(entry)

    return "\n\n".join(formatted)


def create_summary_prompt(
    texts: List[str],
    cluster_id: int,
    metadata: Optional[Dict] = None,
) -> str:
    """Create prompt for cluster summarization."""
    samples_text = format_samples_for_prompt(texts)

    metadata_text = ""
    if metadata:
        if "top_subreddit" in metadata:
            metadata_text += f"- Dominant subreddit: r/{metadata['top_subreddit']}\n"
        if "count" in metadata:
            metadata_text += f"- Total posts in cluster: {metadata['count']:,}\n"
        if "sentiment_mean" in metadata:
            sentiment = metadata["sentiment_mean"]
            sentiment_label = "positive" if sentiment > 0.2 else "negative" if sentiment < -0.2 else "neutral"
            metadata_text += f"- Average sentiment: {sentiment_label} ({sentiment:.2f})\n"
        if "time_start" in metadata and "time_end" in metadata:
            metadata_text += f"- Time range: {metadata['time_start']} to {metadata['time_end']}\n"

    prompt = f"""Analyze these Reddit posts/comments from Cluster #{cluster_id} about Venezuela-US relations.

{f"Cluster Statistics:{chr(10)}{metadata_text}" if metadata_text else ""}

Sample texts from this cluster:

{samples_text}

Based on these samples, provide:
1. **Theme**: A short (3-5 word) name for this cluster's main theme
2. **Summary**: A 2-3 sentence description of what this cluster represents
3. **Key Topics**: 3-5 key topics or subjects discussed
4. **Perspective**: The dominant perspective or viewpoint (if discernible)
5. **Tone**: The overall emotional tone (e.g., critical, supportive, neutral, angry, concerned)

Format your response as JSON:
{{
    "theme": "...",
    "summary": "...",
    "key_topics": ["...", "...", "..."],
    "perspective": "...",
    "tone": "..."
}}"""

    return prompt


def summarize_cluster_with_openai(
    texts: List[str],
    cluster_id: int,
    metadata: Optional[Dict] = None,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Dict:
    """
    Summarize cluster using OpenAI API.

    Requires OPENAI_API_KEY environment variable or api_key parameter.
    """
    import os
    from openai import OpenAI

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")

    client = OpenAI(api_key=api_key)

    prompt = create_summary_prompt(texts, cluster_id, metadata)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert analyst studying political discourse on social media."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    content = response.choices[0].message.content

    # Parse JSON response
    try:
        # Find JSON in response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(content[start:end])
            result["cluster_id"] = cluster_id
            return result
    except json.JSONDecodeError:
        pass

    # Fallback if JSON parsing fails
    return {
        "cluster_id": cluster_id,
        "theme": "Unknown",
        "summary": content[:500],
        "key_topics": [],
        "perspective": "Unknown",
        "tone": "Unknown",
    }


def summarize_cluster_with_anthropic(
    texts: List[str],
    cluster_id: int,
    metadata: Optional[Dict] = None,
    model: str = "claude-3-haiku-20240307",
    api_key: Optional[str] = None,
) -> Dict:
    """
    Summarize cluster using Anthropic API.

    Requires ANTHROPIC_API_KEY environment variable or api_key parameter.
    """
    import os
    import anthropic

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key required")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = create_summary_prompt(texts, cluster_id, metadata)

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    content = response.content[0].text

    # Parse JSON response
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(content[start:end])
            result["cluster_id"] = cluster_id
            return result
    except json.JSONDecodeError:
        pass

    return {
        "cluster_id": cluster_id,
        "theme": "Unknown",
        "summary": content[:500],
        "key_topics": [],
        "perspective": "Unknown",
        "tone": "Unknown",
    }


def summarize_all_clusters(
    df: pd.DataFrame,
    n_samples: int = 20,
    text_column: str = "text",
    llm_provider: str = "anthropic",  # or "openai"
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate summaries for all clusters.

    Returns DataFrame with cluster summaries.
    """
    from tqdm import tqdm

    if "cluster_id" not in df.columns:
        raise ValueError("DataFrame must have cluster_id column")

    # Get unique clusters (excluding noise)
    clusters = sorted(df[df["cluster_id"] != -1]["cluster_id"].unique())

    print(f"Summarizing {len(clusters)} clusters...")

    summaries = []
    for cluster_id in tqdm(clusters, desc="Summarizing"):
        # Get samples
        texts = sample_cluster_texts(df, cluster_id, n_samples, text_column)

        if not texts:
            continue

        # Get metadata
        cluster_df = df[df["cluster_id"] == cluster_id]
        metadata = {
            "count": len(cluster_df),
            "top_subreddit": cluster_df["subreddit"].mode().iloc[0] if len(cluster_df) > 0 else None,
        }
        if "sentiment_score" in cluster_df.columns:
            metadata["sentiment_mean"] = cluster_df["sentiment_score"].mean()
        if "created_datetime" in cluster_df.columns:
            metadata["time_start"] = str(cluster_df["created_datetime"].min())
            metadata["time_end"] = str(cluster_df["created_datetime"].max())

        # Summarize
        try:
            if llm_provider == "openai":
                summary = summarize_cluster_with_openai(
                    texts, cluster_id, metadata, model or "gpt-4o-mini", api_key
                )
            else:
                summary = summarize_cluster_with_anthropic(
                    texts, cluster_id, metadata, model or "claude-3-haiku-20240307", api_key
                )

            summary.update(metadata)
            summaries.append(summary)

        except Exception as e:
            print(f"Error summarizing cluster {cluster_id}: {e}")
            summaries.append({
                "cluster_id": cluster_id,
                "theme": "Error",
                "summary": str(e),
                **metadata,
            })

    return pd.DataFrame(summaries)


def generate_keyword_summary(
    df: pd.DataFrame,
    cluster_id: int,
    text_column: str = "text",
    n_keywords: int = 10,
) -> List[str]:
    """
    Generate keyword-based summary without LLM.

    Uses TF-IDF to extract key terms.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_texts = df[df["cluster_id"] == cluster_id][text_column].tolist()

    if not cluster_texts:
        return []

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2),
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        feature_names = vectorizer.get_feature_names_out()

        # Average TF-IDF scores
        avg_scores = tfidf_matrix.mean(axis=0).A1
        top_indices = avg_scores.argsort()[-n_keywords:][::-1]

        return [feature_names[i] for i in top_indices]

    except Exception:
        return []
