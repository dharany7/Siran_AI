"""
routers/health.py — liveness and readiness endpoints.
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """
    Returns service liveness status.
    Use this endpoint to verify the server is running correctly.
    """
    return HealthResponse(
        status="ok",
        service="siren-ai",
        version="0.1.0",
    )
