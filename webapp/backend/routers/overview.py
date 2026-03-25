"""Overview endpoint: summary stats."""

from typing import Optional

from fastapi import APIRouter, Query

from ..services.data_service import get_overview_stats, get_news_overview_stats, get_tiktok_overview_stats

router = APIRouter(prefix="/api/overview", tags=["overview"])


@router.get("/stats")
def overview_stats(platform: Optional[str] = Query(None, description="Platform: reddit or news")):
    if platform == "news":
        result = get_news_overview_stats()
        if result is None:
            return {"error": "News data not available"}
        return result
    if platform == "tiktok":
        result = get_tiktok_overview_stats()
        if result is None:
            return {"error": "TikTok data not available"}
        return result
    if platform == "reddit" or platform is None:
        return get_overview_stats()
    return {"error": f"Unknown platform: {platform}"}
