from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from common import LOOKUP_PATH, USER_AGENT, atomic_write_path, bootstrap_project_paths


bootstrap_project_paths()
from build_url_index import canonicalize_url


try:
    from newspaper import Article, Config
except Exception:  # pragma: no cover - optional dependency in this environment
    Article = None
    Config = None


@dataclass
class ScrapeResult:
    title: str
    text: str
    status: str
    error_details: str
    scrape_source: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for weekly URL scraping."""
    parser = argparse.ArgumentParser(
        description=(
            "Scrape article title/text for weekly event rows, reusing existing lookup content when the "
            "canonical URL is already present."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Weekly raw-event CSV path")
    parser.add_argument("--output", type=Path, required=True, help="Weekly scraped-event CSV path")
    parser.add_argument("--lookup", type=Path, default=LOOKUP_PATH, help="Path to data/preprocessing/url_lookup.csv")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    return parser.parse_args()


def build_existing_lookup_map(lookup_path: Path) -> dict[str, dict[str, str]]:
    """Load existing lookup content keyed by canonical URL."""
    if not lookup_path.exists():
        return {}
    df = pd.read_csv(lookup_path, low_memory=False)
    if "SourceURL_Canonical" not in df.columns:
        df["SourceURL_Canonical"] = df["SourceURL"].map(canonicalize_url)
    lookup: dict[str, dict[str, str]] = {}
    for record in df.to_dict(orient="records"):
        key = str(record.get("SourceURL_Canonical", "")).strip()
        if not key:
            continue
        lookup[key] = {
            "Title": str(record.get("Title", "") or ""),
            "Text": str(record.get("Text", "") or ""),
            "Scrape_Status": str(record.get("Scrape_Status", "") or ""),
        }
    return lookup


def reuse_lookup_result(record: dict[str, str]) -> ScrapeResult | None:
    """Reuse existing lookup content when it is already good enough for weekly rows."""
    title = str(record.get("Title", "") or "")
    text = str(record.get("Text", "") or "")
    status = str(record.get("Scrape_Status", "") or "")
    has_content = bool(title.strip() or text.strip())
    success_like = "success" in status.lower()
    if not has_content and not success_like:
        return None
    return ScrapeResult(
        title=title,
        text=text,
        status=status or ("Success" if has_content else "Reused"),
        error_details="",
        scrape_source="existing_lookup",
    )


def scrape_with_newspaper(url: str, timeout: int) -> ScrapeResult | None:
    """Try scraping an article with newspaper3k when available."""
    if Article is None or Config is None:
        return None
    try:
        config = Config()
        config.browser_user_agent = USER_AGENT
        config.request_timeout = timeout
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = (article.text or "").strip()
        if not text:
            return None
        return ScrapeResult(
            title=(article.title or "").strip(),
            text=text,
            status="Success",
            error_details="",
            scrape_source="newspaper3k",
        )
    except Exception as exc:
        return ScrapeResult(title="", text="", status="Error", error_details=str(exc), scrape_source="newspaper3k")


def _meta_content(soup: BeautifulSoup, property_name: str) -> str:
    """Extract a meta content value by property or name."""
    tag = soup.find("meta", attrs={"property": property_name}) or soup.find("meta", attrs={"name": property_name})
    if tag is None:
        return ""
    content = tag.get("content")
    return str(content).strip() if content else ""


def scrape_with_bs4(url: str, timeout: int) -> ScrapeResult:
    """Fallback scraping strategy using requests and BeautifulSoup."""
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        for tag_name in ("script", "style", "noscript"):
            for tag in soup.find_all(tag_name):
                tag.decompose()

        title = (
            _meta_content(soup, "og:title")
            or _meta_content(soup, "twitter:title")
            or (soup.title.get_text(" ", strip=True) if soup.title else "")
        )

        paragraphs = []
        article_tag = soup.find("article")
        paragraph_source = article_tag.find_all("p") if article_tag else soup.find_all("p")
        for p_tag in paragraph_source:
            text = p_tag.get_text(" ", strip=True)
            if len(text) >= 40:
                paragraphs.append(text)

        text = "\n\n".join(paragraphs).strip()
        if text:
            return ScrapeResult(title=title, text=text, status="Success", error_details="", scrape_source="bs4")
        return ScrapeResult(
            title=title,
            text="",
            status="Empty_Content",
            error_details="Parsed HTML but extracted no article text",
            scrape_source="bs4",
        )
    except Exception as exc:
        return ScrapeResult(title="", text="", status="Error", error_details=str(exc), scrape_source="bs4")


def scrape_url(url: str, timeout: int) -> ScrapeResult:
    """Scrape a URL using newspaper3k when available, then fall back to bs4 extraction."""
    newspaper_result = scrape_with_newspaper(url, timeout=timeout)
    if newspaper_result is not None:
        if newspaper_result.status == "Success":
            return newspaper_result
        if newspaper_result.status == "Error" and newspaper_result.error_details:
            fallback = scrape_with_bs4(url, timeout=timeout)
            if fallback.status == "Success":
                return fallback
            if not fallback.error_details:
                fallback.error_details = newspaper_result.error_details
            return fallback
    return scrape_with_bs4(url, timeout=timeout)


def main() -> None:
    """Scrape or reuse article content for each weekly event row."""
    args = parse_args()
    events_df = pd.read_csv(args.input, low_memory=False)
    if events_df.empty:
        events_df = events_df.copy()
        events_df["SourceURL_Canonical"] = pd.Series(dtype="object")
        events_df["Title"] = pd.Series(dtype="object")
        events_df["Text"] = pd.Series(dtype="object")
        events_df["Scrape_Status"] = pd.Series(dtype="object")
        events_df["Error_Details"] = pd.Series(dtype="object")
        events_df["Scrape_Source"] = pd.Series(dtype="object")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(args.output, index=False)
        print(f"No weekly rows to scrape. Output written: {args.output}")
        return

    events_df["SourceURL"] = events_df["SourceURL"].fillna("").astype(str)
    events_df["SourceURL_Canonical"] = events_df["SourceURL"].map(canonicalize_url)

    existing_lookup = build_existing_lookup_map(args.lookup)
    cache: dict[str, ScrapeResult] = {}
    titles: list[str] = []
    texts: list[str] = []
    statuses: list[str] = []
    errors: list[str] = []
    sources: list[str] = []

    total_rows = len(events_df)
    url_num = 0
    print(f"Processing {total_rows:,} event rows ({len(existing_lookup):,} canonical URLs already in lookup)...")
    for record in events_df.to_dict(orient="records"):
        canonical = str(record["SourceURL_Canonical"]).strip()
        url = str(record["SourceURL"]).strip()
        if canonical in cache:
            result = cache[canonical]
        else:
            reused = reuse_lookup_result(existing_lookup.get(canonical, {}))
            result = reused if reused is not None else scrape_url(url, timeout=args.timeout)
            cache[canonical] = result
            url_num += 1
            if url_num % 100 == 0:
                print(f"  [{url_num}] unique URLs processed (status={result.status}): {url[:80]}")

        titles.append(result.title)
        texts.append(result.text)
        statuses.append(result.status)
        errors.append(result.error_details)
        sources.append(result.scrape_source)

    events_df["Title"] = titles
    events_df["Text"] = texts
    events_df["Scrape_Status"] = statuses
    events_df["Error_Details"] = errors
    events_df["Scrape_Source"] = sources

    print(f"Writing output to {args.output} ...")
    with atomic_write_path(args.output) as tmp:
        events_df.to_csv(tmp, index=False)

    fresh_scrapes = sum(1 for value in sources if value != "existing_lookup")
    print(f"Weekly event rows processed: {len(events_df):,}")
    print(f"Unique canonical URLs touched: {len(cache):,}")
    print(f"Fresh scrapes performed: {fresh_scrapes:,}")
    print(f"Output written: {args.output}")


if __name__ == "__main__":
    main()
