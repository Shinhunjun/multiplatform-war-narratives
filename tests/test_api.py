"""Integration tests for FastAPI backend endpoints."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from webapp.backend.main import app

client = TestClient(app)


# ── Health & root ────────────────────────────────────────────────────────────


class TestHealthEndpoints:
    """Tests for basic server health."""

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "platforms" in data
        assert "reddit" in data["platforms"]

    def test_docs_accessible(self):
        response = client.get("/docs")
        assert response.status_code == 200


# ── Overview ─────────────────────────────────────────────────────────────────


class TestOverviewEndpoints:
    """Tests for overview/stats endpoints."""

    def test_overview_reddit(self):
        response = client.get("/api/overview/stats?platform=reddit")
        assert response.status_code == 200
        data = response.json()
        assert data["platform"] == "reddit"
        assert data["total_documents"] > 0

    def test_overview_news(self):
        response = client.get("/api/overview/stats?platform=news")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] > 0

    def test_overview_tiktok(self):
        response = client.get("/api/overview/stats?platform=tiktok")
        assert response.status_code == 200


# ── Sentiment ────────────────────────────────────────────────────────────────


class TestSentimentEndpoints:
    """Tests for sentiment analysis endpoints."""

    def test_sentiment_by_month(self):
        response = client.get("/api/sentiment/by-month?platform=reddit")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "year_month" in data[0]
            assert "mean_sentiment" in data[0]

    def test_sentiment_by_month_with_range(self):
        response = client.get("/api/sentiment/by-month?platform=reddit&start=2019-01&end=2019-06")
        assert response.status_code == 200
        data = response.json()
        for row in data:
            assert row["year_month"] >= "2019-01"
            assert row["year_month"] <= "2019-06"

    def test_sentiment_by_subreddit(self):
        response = client.get("/api/sentiment/by-subreddit?platform=reddit")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_sentiment_boxplot(self):
        response = client.get("/api/sentiment/boxplot?platform=reddit")
        assert response.status_code == 200
        data = response.json()
        if data:
            assert "median" in data[0]
            assert "q1" in data[0]
            assert "q3" in data[0]


# ── Topics ───────────────────────────────────────────────────────────────────


class TestTopicEndpoints:
    """Tests for topic modeling endpoints."""

    def test_topic_info(self):
        response = client.get("/api/topics/info?platform=reddit")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_monthly_fitted_months(self):
        response = client.get("/api/topics/monthly-fitted/months?platform=reddit")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert len(data[0]) == 7  # YYYY-MM format

    def test_monthly_fitted_topics(self):
        months_resp = client.get("/api/topics/monthly-fitted/months?platform=reddit")
        months = months_resp.json()
        if months:
            response = client.get(f"/api/topics/monthly-fitted?month={months[0]}&platform=reddit")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


# ── Clusters ─────────────────────────────────────────────────────────────────


class TestClusterEndpoints:
    """Tests for clustering endpoints."""

    def test_cluster_summaries(self):
        response = client.get("/api/clusters/summaries?platform=reddit&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "cluster_id" in data[0]

    def test_cluster_monthly_months(self):
        response = client.get("/api/clusters/monthly/months?platform=reddit")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


# ── Entities ─────────────────────────────────────────────────────────────────


class TestEntityEndpoints:
    """Tests for entity/knowledge graph endpoints."""

    def test_entity_network_reddit(self):
        response = client.get("/api/entities/network?platform=reddit")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "communities" in data
        assert len(data["nodes"]) > 0

    def test_entity_network_with_period(self):
        response = client.get("/api/entities/network?platform=reddit&start=2019-01&end=2019-06")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "communities" in data

    def test_entity_network_news(self):
        response = client.get("/api/entities/network?platform=news")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data

    def test_entity_network_tiktok(self):
        response = client.get("/api/entities/network?platform=tiktok")
        assert response.status_code == 200

    def test_entity_relationships(self):
        response = client.get("/api/entities/relationships?platform=reddit")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "source" in data[0]
            assert "target" in data[0]
            assert "relation" in data[0]

    def test_entity_months(self):
        response = client.get("/api/entities/months?platform=reddit")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert data == sorted(data)  # Should be sorted


# ── TikTok ───────────────────────────────────────────────────────────────────


class TestTikTokEndpoints:
    """Tests for TikTok-specific endpoints."""

    def test_tiktok_hashtags(self):
        response = client.get("/api/tiktok/hashtags?top_n=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_tiktok_engagement(self):
        response = client.get("/api/tiktok/engagement")
        assert response.status_code == 200

    def test_tiktok_regions(self):
        response = client.get("/api/tiktok/regions")
        assert response.status_code == 200


# ── Cross-platform ───────────────────────────────────────────────────────────


class TestCrossPlatform:
    """Tests for cross-platform consistency."""

    def test_all_platforms_return_sentiment(self):
        for platform in ["reddit", "news", "tiktok"]:
            response = client.get(f"/api/sentiment/by-month?platform={platform}")
            assert response.status_code == 200, f"Failed for {platform}"

    def test_all_platforms_return_entities(self):
        for platform in ["reddit", "news", "tiktok"]:
            response = client.get(f"/api/entities/network?platform={platform}")
            assert response.status_code == 200, f"Failed for {platform}"

    def test_sentiment_values_in_range(self):
        """Sentiment scores should be between -1 and 1."""
        response = client.get("/api/sentiment/by-month?platform=reddit")
        data = response.json()
        for row in data:
            assert -1.0 <= row["mean_sentiment"] <= 1.0, (
                f"Sentiment {row['mean_sentiment']} out of range for {row['year_month']}"
            )
