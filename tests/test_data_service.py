"""Unit tests for data_service: entity network building and data loading."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.backend.services.data_service import (
    build_entity_network,
    get_entity_months,
    get_entity_relationships_filtered,
    _load_entities_parquet,
    _load_relationships_parquet,
)


# ── Entity parquet loading ───────────────────────────────────────────────────


class TestEntityDataLoading:
    """Tests for parquet data loading."""

    def test_load_reddit_entities(self):
        df = _load_entities_parquet("reddit")
        assert df is not None
        assert len(df) > 0
        assert "year_month" in df.columns
        assert "name" in df.columns
        assert "type" in df.columns
        assert "count" in df.columns

    def test_load_news_entities(self):
        df = _load_entities_parquet("news")
        assert df is not None
        assert len(df) > 0

    def test_load_tiktok_entities(self):
        df = _load_entities_parquet("tiktok")
        assert df is not None
        assert len(df) > 0

    def test_load_relationships(self):
        df = _load_relationships_parquet("reddit")
        assert df is not None
        assert "source" in df.columns
        assert "target" in df.columns
        assert "relation" in df.columns

    def test_invalid_platform_returns_none(self):
        df = _load_entities_parquet("nonexistent")
        assert df is None


# ── Entity months ────────────────────────────────────────────────────────────


class TestEntityMonths:
    """Tests for available month listing."""

    def test_reddit_months_sorted(self):
        months = get_entity_months("reddit")
        assert len(months) > 0
        assert months == sorted(months)

    def test_reddit_month_format(self):
        months = get_entity_months("reddit")
        for m in months:
            assert len(m) == 7  # YYYY-MM
            assert m[4] == "-"

    def test_news_months(self):
        months = get_entity_months("news")
        assert len(months) > 0

    def test_tiktok_months(self):
        months = get_entity_months("tiktok")
        assert len(months) > 0

    def test_invalid_platform(self):
        months = get_entity_months("nonexistent")
        assert months == []


# ── Entity network building ──────────────────────────────────────────────────


class TestBuildEntityNetwork:
    """Tests for dynamic entity network construction."""

    def test_reddit_all_time(self):
        net = build_entity_network("reddit")
        assert "nodes" in net
        assert "edges" in net
        assert "communities" in net
        assert "platform" in net
        assert net["platform"] == "reddit"
        assert len(net["nodes"]) > 0

    def test_network_node_structure(self):
        net = build_entity_network("reddit")
        node = net["nodes"][0]
        assert "id" in node
        assert "community" in node
        assert "frequency" in node
        assert "type" in node
        assert isinstance(node["frequency"], int)
        assert isinstance(node["community"], int)

    def test_network_edge_structure(self):
        net = build_entity_network("reddit")
        if net["edges"]:
            edge = net["edges"][0]
            assert "source" in edge
            assert "target" in edge
            assert "weight" in edge

    def test_community_structure(self):
        net = build_entity_network("reddit")
        if net["communities"]:
            comm = net["communities"][0]
            assert "id" in comm
            assert "size" in comm
            assert "total_frequency" in comm
            assert "top_members" in comm
            assert isinstance(comm["top_members"], list)

    def test_period_filtering(self):
        net_all = build_entity_network("reddit")
        net_2019 = build_entity_network("reddit", "2019-01", "2019-06")
        # Filtered network should have fewer or equal entities
        assert len(net_2019["nodes"]) <= len(net_all["nodes"])
        assert len(net_2019["nodes"]) > 0

    def test_empty_period_returns_empty(self):
        net = build_entity_network("reddit", "2050-01", "2050-12")
        assert len(net["nodes"]) == 0

    def test_news_network(self):
        net = build_entity_network("news")
        assert len(net["nodes"]) > 0
        assert net["platform"] == "news"

    def test_tiktok_network(self):
        net = build_entity_network("tiktok")
        assert len(net["nodes"]) > 0

    def test_community_ids_assigned(self):
        net = build_entity_network("reddit", "2019-01", "2019-06")
        community_ids = {n["community"] for n in net["nodes"]}
        assert len(community_ids) >= 1  # At least 1 community

    def test_communities_sorted_by_frequency(self):
        net = build_entity_network("reddit")
        freqs = [c["total_frequency"] for c in net["communities"]]
        assert freqs == sorted(freqs, reverse=True)


# ── Filtered relationships ───────────────────────────────────────────────────


class TestFilteredRelationships:
    """Tests for period-filtered relationship queries."""

    def test_reddit_relationships(self):
        rels = get_entity_relationships_filtered("reddit")
        assert isinstance(rels, list)
        assert len(rels) > 0

    def test_relationship_structure(self):
        rels = get_entity_relationships_filtered("reddit")
        if rels:
            rel = rels[0]
            assert "source" in rel
            assert "target" in rel
            assert "relation" in rel
            assert "count" in rel

    def test_filtered_by_period(self):
        rels = get_entity_relationships_filtered("reddit", "2019-01", "2019-06")
        assert isinstance(rels, list)
        assert len(rels) > 0

    def test_max_50_results(self):
        rels = get_entity_relationships_filtered("reddit")
        assert len(rels) <= 50

    def test_sorted_by_count(self):
        rels = get_entity_relationships_filtered("reddit")
        counts = [r["count"] for r in rels]
        assert counts == sorted(counts, reverse=True)

    def test_empty_period(self):
        rels = get_entity_relationships_filtered("reddit", "2050-01", "2050-12")
        assert rels == []

    def test_news_relationships(self):
        rels = get_entity_relationships_filtered("news")
        assert isinstance(rels, list)
        assert len(rels) > 0
