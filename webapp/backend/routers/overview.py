"""Overview endpoint: summary stats."""

from fastapi import APIRouter

from ..services.data_service import get_overview_stats

router = APIRouter(prefix="/api/overview", tags=["overview"])


@router.get("/stats")
def overview_stats():
    return get_overview_stats()
