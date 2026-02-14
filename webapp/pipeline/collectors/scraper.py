"""
News article scraper for GDELT source URLs.
Extracts article text from news websites for NLP analysis.
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiohttp
from bs4 import BeautifulSoup

from ..config import PipelineConfig

logger = logging.getLogger(__name__)


class ArticleScraper:
    """Scrapes and extracts text from news article URLs."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Research Bot - Academic Project)",
            "Accept": "text/html,application/xhtml+xml",
        }

    def _extract_text(self, html: str, url: str) -> Optional[dict]:
        """Extract article text from HTML."""
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove non-content tags
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            # Try common article selectors
            article = (
                soup.find("article")
                or soup.find("div", class_="article-body")
                or soup.find("div", class_="story-body")
                or soup.find("div", {"id": "article-body"})
            )

            if article:
                paragraphs = article.find_all("p")
            else:
                paragraphs = soup.find_all("p")

            text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)

            if len(text) < 100:
                return None

            # Extract title
            title_tag = soup.find("h1") or soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            return {
                "url": url,
                "title": title[:500],
                "text": text[:10000],  # cap at 10K chars
                "scraped_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.debug(f"Extract failed for {url}: {e}")
            return None

    async def _fetch_one(
        self, session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore
    ) -> Optional[dict]:
        """Fetch and parse a single URL."""
        async with semaphore:
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.config.scraper_timeout),
                    allow_redirects=True,
                ) as resp:
                    if resp.status != 200:
                        return None
                    html = await resp.text()
                    return self._extract_text(html, url)
            except Exception as e:
                logger.debug(f"Fetch failed for {url}: {e}")
                return None

    async def scrape_urls(self, urls: List[str]) -> List[dict]:
        """Scrape multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(self.config.scraper_max_concurrent)
        articles = []

        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = [self._fetch_one(session, url, semaphore) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict) and result is not None:
                    articles.append(result)

        logger.info(f"Scraped {len(articles)}/{len(urls)} articles successfully")
        return articles

    def save_raw(self, articles: List[dict], run_date: str) -> Path:
        """Save scraped articles."""
        news_dir = self.config.raw_dir / "news"
        path = news_dir / f"articles_{run_date}.json"

        with open(path, "w") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(articles)} articles to {path}")
        return path

    def run(self, urls: List[str], run_date: str) -> dict:
        """Execute article scraping."""
        articles = asyncio.run(self.scrape_urls(urls))
        path = self.save_raw(articles, run_date)

        return {
            "articles_count": len(articles),
            "urls_attempted": len(urls),
            "path": str(path),
        }
