"""Health check endpoints."""

from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and basic information.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "meeting-agent"
    }
