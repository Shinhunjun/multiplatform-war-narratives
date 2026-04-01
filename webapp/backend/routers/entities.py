"""Entity network endpoints."""

from typing import Optional

from fastapi import APIRouter, Query

from ..services.data_service import (
    build_entity_network,
    get_entity_months,
    get_entity_network,
    get_entity_relationships,
    get_entity_relationships_filtered,
)

router = APIRouter(prefix="/api/entities", tags=["entities"])


@router.get("/network")
def entity_network(
    platform: str = Query("reddit"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Get entity co-occurrence network with communities.

    If start/end are provided, dynamically builds network from parquet data.
    Otherwise falls back to pre-built JSON (Reddit only).
    """
    if start or end:
        return build_entity_network(platform, start, end)
    # Fallback: pre-built JSON (only exists for reddit)
    result = get_entity_network(platform)
    if result:
        return result
    # No pre-built JSON — build from parquet for all months
    return build_entity_network(platform)


@router.get("/relationships")
def entity_relationships(
    platform: str = Query("reddit"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Get top entity relationships, optionally filtered by period."""
    if start or end:
        return get_entity_relationships_filtered(platform, start, end)
    result = get_entity_relationships(platform)
    if result:
        return result
    return get_entity_relationships_filtered(platform)


@router.get("/months")
def entity_months(platform: str = Query("reddit")):
    """Get available months for entity data."""
    return get_entity_months(platform)
