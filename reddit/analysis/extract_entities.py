"""
Cross-Platform Entity Extraction using Gemini 2.0 Flash.

Concatenates top documents per month into a single prompt,
extracts entities in ONE API call per month.

Output:
  outputs/entities/
    entities_{platform}.parquet     — (doc_id, year_month, name, type, platform)
    cooccurrence_{platform}.parquet — (entity_a, entity_b, weight)
    monthly_entities_{platform}.parquet — (year_month, name, type, count)

Usage:
    python extract_entities.py
    python extract_entities.py --platform reddit
    python extract_entities.py --docs-per-month 100
"""

import argparse
import json
import logging
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent / "outputs" / "entities"

MONTH_PROMPT = """You are analyzing social media and news texts about Venezuela-US relations from {month}.

Below are {n_docs} documents from {platform}. Extract entities AND relationships.

Return ONLY a valid JSON object with two keys:

"entities": array of top 30 entities, each with:
  - "name": normalized name (e.g., "Nicolás Maduro" not "maduro")
  - "type": PERSON, ORG, EVENT, POLICY, or LOCATION
  - "count": estimated number of documents mentioning this entity

"relationships": array of top 20 key relationships, each with:
  - "source": entity name
  - "target": entity name
  - "relation": short description (e.g., "imposed sanctions on", "recognized as president", "fled to")
  - "count": estimated frequency

Merge duplicates. No markdown formatting.

Documents:
{texts}"""


def get_gemini_client():
    from google import genai
    return genai.Client(vertexai=True, project="theta-bliss-486220-s1", location="us-central1")


def extract_month_entities(client, month: str, platform: str, texts: list[str], max_retries: int = 2) -> dict:
    """Extract entities + relationships from concatenated monthly texts in ONE API call."""
    truncated = [t[:200] for t in texts[:300]]
    joined = "\n---\n".join(truncated)

    if len(joined) > 8000:
        joined = joined[:8000]

    prompt = MONTH_PROMPT.format(month=month, n_docs=len(truncated), platform=platform, texts=joined)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"max_output_tokens": 2000, "temperature": 0.0},
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                raw = raw.rsplit("```", 1)[0]
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
            elif isinstance(result, list):
                return {"entities": result, "relationships": []}
            return {"entities": [], "relationships": []}
        except (json.JSONDecodeError, Exception) as e:
            if attempt == max_retries - 1:
                logger.warning(f"  {month}: Failed — {e}")
                return {"entities": [], "relationships": []}
            time.sleep(1)
    return {"entities": [], "relationships": []}


def load_platform_texts(platform: str) -> pd.DataFrame:
    """Load texts for a platform with year_month."""
    if platform == "reddit":
        assigns = pd.read_parquet(Path(__file__).parent / "outputs" / "topics" / "topic_assignments.parquet")
        data_dir = PROJECT_ROOT / "reddit" / "data-collection" / "data" / "preprocessed"
        sub = pd.read_parquet(data_dir / "submissions_clean.parquet")
        com = pd.read_parquet(data_dir / "comments_clean.parquet")
        text_map = {}
        for _, r in sub.iterrows():
            text_map[r["id"]] = r.get("full_text", r.get("title", ""))
        for _, r in com.iterrows():
            text_map[r["id"]] = r.get("body_clean", r.get("body", ""))
        assigns["text"] = assigns["id"].map(text_map)
        return assigns[assigns["text"].notna() & (assigns["text"].str.len() > 20)].copy()

    elif platform == "news":
        assigns = pd.read_parquet(Path(__file__).parent / "outputs_news" / "topics" / "topic_assignments.parquet")
        gdelt = pd.read_csv(PROJECT_ROOT / "data" / "gdelt" / "gdelt_scraped_updated.csv", low_memory=False)
        gdelt = gdelt[gdelt["Scrape_Status"].str.lower().str.contains("success", na=False)]
        gdelt = gdelt[gdelt["Text"].str.len() >= 50].reset_index(drop=True)
        text_map = {f"gdelt_{i}": row["Text"] for i, row in gdelt.iterrows()}
        assigns["text"] = assigns["id"].map(text_map)
        return assigns[assigns["text"].notna() & (assigns["text"].str.len() > 20)].copy()

    elif platform == "tiktok":
        assigns = pd.read_parquet(Path(__file__).parent / "outputs_tiktok" / "topics" / "topic_assignments.parquet")
        videos_dir = PROJECT_ROOT / "tiktok" / "data-collection" / "data" / "videos"
        all_texts = []
        for f in sorted(videos_dir.glob("videos_*.json")):
            data = json.load(open(f))
            for v in data:
                all_texts.append(v.get("video_description", ""))
        comments_file = PROJECT_ROOT / "tiktok" / "data-collection" / "data" / "comments" / "comments_all_merged.json"
        if comments_file.exists():
            for c in json.load(open(comments_file)):
                all_texts.append(str(c.get("text", "")))
        text_map = {row["id"]: all_texts[i] if i < len(all_texts) else "" for i, (_, row) in enumerate(assigns.iterrows())}
        assigns["text"] = assigns["id"].map(text_map)
        return assigns[assigns["text"].notna() & (assigns["text"].str.len() > 20)].copy()

    raise ValueError(f"Unknown platform: {platform}")


def run_platform(platform: str, docs_per_month: int = 100):
    """Run entity extraction for one platform — 1 API call per month."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Entity Extraction: {platform.upper()}")
    logger.info(f"{'='*60}")

    logger.info("Loading texts...")
    df = load_platform_texts(platform)
    months = sorted(df["year_month"].unique())
    logger.info(f"  {len(df):,} docs, {len(months)} months")

    client = get_gemini_client()
    all_entities = []
    all_relationships = []

    for i, month in enumerate(months):
        month_df = df[df["year_month"] == month]
        month_df = month_df.copy()
        month_df["_len"] = month_df["text"].str.len()
        top = month_df.nlargest(min(docs_per_month, len(month_df)), "_len")
        texts = top["text"].tolist()

        result = extract_month_entities(client, month, platform, texts)
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])

        for e in entities:
            all_entities.append({
                "year_month": month,
                "name": e.get("name", "").strip(),
                "type": e.get("type", "UNKNOWN"),
                "count": e.get("count", 1),
                "platform": platform,
            })

        for r in relationships:
            all_relationships.append({
                "year_month": month,
                "source": r.get("source", "").strip(),
                "target": r.get("target", "").strip(),
                "relation": r.get("relation", ""),
                "count": r.get("count", 1),
                "platform": platform,
            })

        logger.info(f"  [{i+1}/{len(months)}] {month}: {len(texts)} docs → {len(entities)} entities, {len(relationships)} relations")
        time.sleep(0.2)

    # Save entities
    ent_df = pd.DataFrame(all_entities)
    if ent_df.empty:
        logger.warning("No entities extracted!")
        return

    ent_df = ent_df[ent_df["name"].str.len() > 0]
    ent_df.to_parquet(OUTPUT_DIR / f"entities_{platform}.parquet", index=False)
    logger.info(f"  Saved {len(ent_df):,} entity rows")

    # Top entities
    top = ent_df.groupby("name")["count"].sum().nlargest(15)
    logger.info(f"  Top entities: {dict(top)}")

    # Save relationships
    rel_df = pd.DataFrame(all_relationships)
    if not rel_df.empty:
        rel_df = rel_df[(rel_df["source"].str.len() > 0) & (rel_df["target"].str.len() > 0)]
        rel_df.to_parquet(OUTPUT_DIR / f"relationships_{platform}.parquet", index=False)
        logger.info(f"  Saved {len(rel_df):,} relationships")

        top_rels = rel_df.groupby(["source", "target", "relation"])["count"].sum().nlargest(10)
        logger.info(f"  Top relationships: {dict(top_rels)}")

    # Co-occurrence
    cooc = Counter()
    for _, group in ent_df.groupby("year_month"):
        names = sorted(group["name"].unique())
        for a, b in combinations(names, 2):
            cooc[(a, b)] += 1
    cooc_df = pd.DataFrame([{"entity_a": a, "entity_b": b, "weight": w} for (a, b), w in cooc.most_common(500)])
    cooc_df.to_parquet(OUTPUT_DIR / f"cooccurrence_{platform}.parquet", index=False)
    logger.info(f"  Saved {len(cooc_df)} co-occurrence pairs")

    # Monthly entity counts
    monthly = ent_df[["year_month", "name", "type", "count"]].sort_values(["year_month", "count"], ascending=[True, False])
    monthly.to_parquet(OUTPUT_DIR / f"monthly_entities_{platform}.parquet", index=False)
    logger.info(f"  Saved monthly entities: {len(monthly)} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["reddit", "news", "tiktok"])
    parser.add_argument("--docs-per-month", type=int, default=100)
    args = parser.parse_args()

    platforms = [args.platform] if args.platform else ["reddit", "news", "tiktok"]
    for p in platforms:
        run_platform(p, args.docs_per_month)

    logger.info(f"\n{'='*60}")
    logger.info("ALL PLATFORMS COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
