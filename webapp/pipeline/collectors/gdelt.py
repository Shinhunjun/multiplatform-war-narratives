"""
GDELT data collector using BigQuery.
Fetches recent news events and articles about Venezuela from the GDELT Project.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import PipelineConfig

logger = logging.getLogger(__name__)


GDELT_EVENTS_TABLE = "gdelt-bq.gdeltv2.events"
GDELT_GKG_TABLE = "gdelt-bq.gdeltv2.gkg"
GDELT_MENTIONS_TABLE = "gdelt-bq.gdeltv2.eventmentions"


class GDELTCollector:
    """Collects Venezuela-related news from GDELT via BigQuery."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        """Lazy init BigQuery client."""
        if self._client is None:
            from google.cloud import bigquery
            self._client = bigquery.Client(project=self.config.gcp_project)
        return self._client

    def collect_events(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Query GDELT events table for Venezuela-related events.

        Args:
            start_date: YYYYMMDD format
            end_date: YYYYMMDD format
        """
        keyword_conditions = " OR ".join(
            [f"Actor1Name LIKE '%{kw.upper()}%' OR Actor2Name LIKE '%{kw.upper()}%'"
             for kw in self.config.gdelt_keywords]
        )

        query = f"""
        SELECT
            GLOBALEVENTID,
            SQLDATE,
            Actor1Name,
            Actor1CountryCode,
            Actor1Type1Code,
            Actor2Name,
            Actor2CountryCode,
            Actor2Type1Code,
            EventCode,
            EventBaseCode,
            EventRootCode,
            QuadClass,
            GoldsteinScale,
            NumMentions,
            NumSources,
            NumArticles,
            AvgTone,
            SOURCEURL
        FROM `{GDELT_EVENTS_TABLE}`
        WHERE SQLDATE BETWEEN '{start_date}' AND '{end_date}'
          AND (
            Actor1CountryCode = 'VEN'
            OR Actor2CountryCode = 'VEN'
            OR {keyword_conditions}
          )
        ORDER BY SQLDATE DESC
        LIMIT {self.config.gdelt_max_articles}
        """

        logger.info(f"Querying GDELT events: {start_date} to {end_date}")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Retrieved {len(df)} events")
        return df

    def collect_gkg(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Query GDELT Global Knowledge Graph for Venezuela-related articles.
        Returns themes, persons, organizations, and tone data.
        """
        keyword_pattern = "|".join(self.config.gdelt_keywords)

        query = f"""
        SELECT
            DATE,
            SourceCollectionIdentifier,
            DocumentIdentifier,
            V2Themes,
            V2Locations,
            V2Persons,
            V2Organizations,
            V2Tone,
            SharingImage,
            CONCAT(
                IFNULL(V2Themes, ''), ' ',
                IFNULL(V2Persons, ''), ' ',
                IFNULL(V2Organizations, '')
            ) AS combined_text
        FROM `{GDELT_GKG_TABLE}`
        WHERE DATE BETWEEN {start_date}000000 AND {end_date}235959
          AND (
            REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'(?i)({keyword_pattern})')
            OR REGEXP_CONTAINS(LOWER(V2Themes), r'(?i)(venezuela|maduro)')
          )
        ORDER BY DATE DESC
        LIMIT {self.config.gdelt_max_articles}
        """

        logger.info(f"Querying GDELT GKG: {start_date} to {end_date}")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Retrieved {len(df)} GKG records")
        return df

    def save_raw(
        self,
        events_df: pd.DataFrame,
        gkg_df: Optional[pd.DataFrame],
        run_date: str,
    ) -> dict:
        """Save collected GDELT data."""
        gdelt_dir = self.config.raw_dir / "gdelt"

        paths = {}

        if not events_df.empty:
            events_path = gdelt_dir / f"events_{run_date}.parquet"
            events_df.to_parquet(events_path, index=False)
            paths["events"] = str(events_path)
            logger.info(f"Saved {len(events_df)} events to {events_path}")

        if gkg_df is not None and not gkg_df.empty:
            gkg_path = gdelt_dir / f"gkg_{run_date}.parquet"
            gkg_df.to_parquet(gkg_path, index=False)
            paths["gkg"] = str(gkg_path)
            logger.info(f"Saved {len(gkg_df)} GKG records to {gkg_path}")

        return paths

    def run(self, run_date: str) -> dict:
        """Execute full GDELT collection pipeline."""
        # Calculate date range
        end = datetime.strptime(run_date, "%Y-%m-%d")
        start = end - timedelta(days=self.config.lookback_days)

        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        events_df = self.collect_events(start_str, end_str)
        gkg_df = self.collect_gkg(start_str, end_str)

        # Extract article URLs for scraping
        article_urls = []
        if not events_df.empty and "SOURCEURL" in events_df.columns:
            article_urls.extend(
                events_df["SOURCEURL"].dropna().unique().tolist()
            )
        if gkg_df is not None and not gkg_df.empty and "DocumentIdentifier" in gkg_df.columns:
            article_urls.extend(
                gkg_df["DocumentIdentifier"].dropna().unique().tolist()
            )
        article_urls = list(set(article_urls))

        paths = self.save_raw(events_df, gkg_df, run_date)

        return {
            "events_count": len(events_df),
            "gkg_count": len(gkg_df) if gkg_df is not None else 0,
            "article_urls": article_urls[:200],  # cap for scraper
            "paths": paths,
        }
