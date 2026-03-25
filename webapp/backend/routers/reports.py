"""Report generation endpoints."""

from typing import Optional

from fastapi import APIRouter, Query

from ..services.llm_service import generate_report, list_cached_reports

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/generate")
def gen_report(
    start: str = Query(..., description="Start month YYYY-MM"),
    end: str = Query(..., description="End month YYYY-MM"),
    force: bool = Query(False, description="Force regenerate"),
):
    return generate_report(start, end, force=force)


@router.get("/list")
def list_reports():
    return list_cached_reports()
