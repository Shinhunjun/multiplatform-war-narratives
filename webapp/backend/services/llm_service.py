"""
LLM service using Gemini via Vertex AI.
Provides report generation and chat capabilities.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports_cache"
REPORTS_DIR.mkdir(exist_ok=True)

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        project = os.environ.get("GCP_PROJECT", "theta-bliss-486220-s1")
        location = os.environ.get("GCP_LOCATION", "us-central1")
        _client = genai.Client(vertexai=True, project=project, location=location)
    return _client


def _get_model():
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


# =========================================================================
# DATA AGGREGATION
# =========================================================================

def _aggregate_data(start_month: str, end_month: str) -> dict:
    from . import data_service as ds

    ctx = {"period": f"{start_month} to {end_month}", "platforms": {}}

    # Reddit
    try:
        df = ds.get_sentiment_by_month()
        f = df[(df["year_month"] >= start_month) & (df["year_month"] <= end_month)]
        if not f.empty:
            ctx["platforms"]["reddit"] = {
                "mean_sentiment": round(float(f["mean_sentiment"].mean()), 4),
                "total_docs": int(f["total_count"].sum()),
                "positive_ratio": round(float(f["positive_ratio"].mean()), 4),
                "negative_ratio": round(float(f["negative_ratio"].mean()), 4),
            }
    except Exception:
        pass

    # News
    try:
        df = ds.get_news_sentiment_by_month()
        f = df[(df["year_month"] >= start_month) & (df["year_month"] <= end_month)]
        if not f.empty:
            ctx["platforms"]["news"] = {
                "mean_sentiment": round(float(f["mean_sentiment"].mean()), 4),
                "total_docs": int(f["total_count"].sum()),
                "positive_ratio": round(float(f["positive_ratio"].mean()), 4),
                "negative_ratio": round(float(f["negative_ratio"].mean()), 4),
            }
    except Exception:
        pass

    # TikTok
    try:
        if ds._tiktok_data_available():
            df = ds.get_tiktok_sentiment_by_month()
            f = df[(df["year_month"] >= start_month) & (df["year_month"] <= end_month)]
            if not f.empty:
                tk = {
                    "mean_sentiment": round(float(f["mean_sentiment"].mean()), 4),
                    "total_docs": int(f["total_count"].sum()),
                    "positive_ratio": round(float(f["positive_ratio"].mean()), 4),
                    "negative_ratio": round(float(f["negative_ratio"].mean()), 4),
                }
                ht = ds.get_tiktok_hashtag_trends()
                ht_f = ht[(ht["year_month"] >= start_month) & (ht["year_month"] <= end_month)]
                if not ht_f.empty:
                    tk["top_hashtags"] = ht_f.groupby("hashtag")["count"].sum().nlargest(10).to_dict()
                eng = ds.get_tiktok_engagement_metrics()
                eng_f = eng[(eng["year_month"] >= start_month) & (eng["year_month"] <= end_month)]
                if not eng_f.empty:
                    tk["total_views"] = int(eng_f["total_views"].sum())
                    tk["total_likes"] = int(eng_f["total_likes"].sum())
                    tk["video_count"] = int(eng_f["video_count"].sum())
                ctx["platforms"]["tiktok"] = tk
    except Exception:
        pass

    # Topics
    for pname, getter in [
        ("reddit", ds.get_topics_monthly_fitted),
        ("news", ds.get_news_topics_monthly_fitted),
        ("tiktok", ds.get_tiktok_topics_monthly_fitted),
    ]:
        try:
            tf = getter()
            tf_f = tf[(tf["year_month"] >= start_month) & (tf["year_month"] <= end_month)]
            if not tf_f.empty and pname in ctx["platforms"]:
                ctx["platforms"][pname]["top_topics"] = (
                    tf_f.nlargest(5, "count")[["keywords", "count"]].to_dict("records")
                )
        except Exception:
            pass

    return ctx


# =========================================================================
# REPORT GENERATION
# =========================================================================

def generate_report(start_month: str, end_month: str, force: bool = False) -> dict:
    cache_key = f"report_{start_month}_{end_month}"
    cache_path = REPORTS_DIR / f"{cache_key}.json"

    if cache_path.exists() and not force:
        with open(cache_path) as f:
            return json.load(f)

    ctx = _aggregate_data(start_month, end_month)
    if not ctx["platforms"]:
        return {"error": "No data for this period.", "period": ctx["period"]}

    prompt = f"""You are an analyst monitoring Venezuela-US discourse across Reddit, GDELT News, and TikTok.
Based on the data below for {ctx['period']}, write a concise intelligence report.

DATA:
{json.dumps(ctx, indent=2)}

Write in Markdown:
## Executive Summary
(3 bullet points)

## Platform Analysis
### Reddit
### GDELT News
### TikTok

## Cross-Platform Comparison

## Notable Signals

Be analytical, data-driven, concise. Skip platforms with no data."""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=_get_model(),
            contents=prompt,
        )

        result = {
            "period": ctx["period"],
            "generated_at": datetime.utcnow().isoformat(),
            "report": response.text,
            "data_summary": ctx,
        }
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {"error": str(e), "period": ctx["period"], "data_summary": ctx}


def list_cached_reports() -> list[dict]:
    reports = []
    for f in sorted(REPORTS_DIR.glob("report_*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            reports.append({
                "period": data.get("period", ""),
                "generated_at": data.get("generated_at", ""),
                "has_error": "error" in data,
            })
        except Exception:
            pass
    return reports


# =========================================================================
# CHAT
# =========================================================================

def _filter_by_range(df, start: str | None, end: str | None, col: str = "year_month"):
    """Filter a DataFrame by month range."""
    if df is None or df.empty:
        return df
    if start:
        df = df[df[col] >= start]
    if end:
        df = df[df[col] <= end]
    return df


def _detect_date_from_question(question: str) -> tuple[str | None, str | None]:
    """Try to extract year-month from the question text."""
    import re
    # Match patterns like "2017년 7월", "2017-07", "July 2017", "2024 election"
    # Korean: YYYY년 M월
    m = re.search(r'(\d{4})년\s*(\d{1,2})월', question)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}", f"{m.group(1)}-{m.group(2).zfill(2)}"
    # ISO: YYYY-MM
    m = re.search(r'(\d{4})-(\d{2})', question)
    if m:
        return f"{m.group(1)}-{m.group(2)}", f"{m.group(1)}-{m.group(2)}"
    # English: Month YYYY or YYYY Month
    month_names = {'january':'01','february':'02','march':'03','april':'04','may':'05','june':'06',
                   'july':'07','august':'08','september':'09','october':'10','november':'11','december':'12'}
    for name, num in month_names.items():
        m = re.search(rf'{name}\s+(\d{{4}})', question.lower())
        if m:
            return f"{m.group(1)}-{num}", f"{m.group(1)}-{num}"
        m = re.search(rf'(\d{{4}})\s+{name}', question.lower())
        if m:
            return f"{m.group(1)}-{num}", f"{m.group(1)}-{num}"
    # Just a year: YYYY (without month)
    m = re.search(r'\b(20\d{2})\b', question)
    if m:
        year = m.group(1)
        return f"{year}-01", f"{year}-12"
    return None, None


def _build_chat_context(question: str, start_month: str | None = None, end_month: str | None = None) -> str:
    from . import data_service as ds

    # Auto-detect dates from question if not explicitly set
    if not start_month and not end_month:
        start_month, end_month = _detect_date_from_question(question)

    parts = []
    period_label = ""
    if start_month and end_month:
        if start_month == end_month:
            period_label = f" ({start_month})"
        else:
            period_label = f" ({start_month} to {end_month})"
    elif start_month:
        period_label = f" (from {start_month})"
    elif end_month:
        period_label = f" (until {end_month})"

    if period_label:
        parts.append(f"Analysis period{period_label}")

    try:
        r = ds.get_overview_stats()
        parts.append(f"Reddit: {r['total_documents']:,} docs, {r['subreddits']} subreddits, "
                      f"avg sentiment {r['avg_sentiment']}, {r['num_topics']} topics")
    except Exception:
        pass
    try:
        n = ds.get_news_overview_stats()
        if n:
            parts.append(f"News: {n['total_documents']:,} docs, {n['sources']} sources, avg sentiment {n['avg_sentiment']}")
    except Exception:
        pass
    try:
        t = ds.get_tiktok_overview_stats()
        if t:
            parts.append(f"TikTok: {t['total_documents']:,} docs, avg sentiment {t['avg_sentiment']}, {t.get('num_topics',0)} topics")
    except Exception:
        pass

    for pname, getter in [
        ("Reddit", ds.get_sentiment_by_month),
        ("News", ds.get_news_sentiment_by_month),
        ("TikTok", ds.get_tiktok_sentiment_by_month),
    ]:
        try:
            df = _filter_by_range(getter(), start_month, end_month)
            if not df.empty:
                rows = df[["year_month", "mean_sentiment", "total_count"]].to_string(index=False)
                parts.append(f"\n{pname} monthly sentiment{period_label}:\n{rows}")
        except Exception:
            pass

    q_lower = question.lower()

    if any(w in q_lower for w in ["hashtag", "tiktok", "해시태그", "틱톡"]):
        try:
            ht = _filter_by_range(ds.get_tiktok_hashtag_trends(), start_month, end_month)
            top = ht.groupby("hashtag")["count"].sum().nlargest(15)
            parts.append(f"\nTikTok top hashtags{period_label}:\n{top.to_string()}")
        except Exception:
            pass
        try:
            eng = _filter_by_range(ds.get_tiktok_engagement_metrics(), start_month, end_month)
            if not eng.empty:
                parts.append(f"\nTikTok engagement{period_label}:\n{eng.to_string(index=False)}")
        except Exception:
            pass

    if any(w in q_lower for w in ["topic", "토픽", "주제", "narrative", "담론"]):
        for pname, getter in [
            ("Reddit", ds.get_topics_monthly_fitted),
            ("News", ds.get_news_topics_monthly_fitted),
            ("TikTok", ds.get_tiktok_topics_monthly_fitted),
        ]:
            try:
                tf = _filter_by_range(getter(), start_month, end_month)
                if not tf.empty:
                    top = tf.nlargest(10, "count")[["year_month", "keywords", "count"]]
                    parts.append(f"\n{pname} top topics{period_label}:\n{top.to_string(index=False)}")
            except Exception:
                pass

    if any(w in q_lower for w in ["region", "country", "지역", "국가"]):
        try:
            reg = _filter_by_range(ds.get_tiktok_region_distribution(), start_month, end_month)
            if not reg.empty:
                top = reg.groupby("region_code")["count"].sum().nlargest(10)
                parts.append(f"\nTikTok regions{period_label}:\n{top.to_string()}")
        except Exception:
            pass

    return "\n".join(parts)


def chat(question: str, history: list[dict] | None = None, start_month: str | None = None, end_month: str | None = None) -> str:
    context = _build_chat_context(question, start_month, end_month)

    system_prompt = """You are an expert analyst for the Venezuela-US Multiplatform Narrative Analysis project.
You have access to data from 3 platforms: Reddit (2013-2026), GDELT News (2013-2026), and TikTok (2016-2017).
Answer questions using the provided data context. Be concise, use numbers, cite platforms.
If data is insufficient, say so. Answer in the same language as the question."""

    user_msg = f"""CONTEXT DATA:
{context}

QUESTION: {question}"""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=_get_model(),
            contents=user_msg,
            config={
                "system_instruction": system_prompt,
                "max_output_tokens": 1500,
                "temperature": 0.3,
            },
        )
        return response.text
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return f"Error: {str(e)}"
