"""
Main FastAPI application for Meeting Agent.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from api.routes import bots, health

# Get settings and setup logging
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Global state
app_state: Dict[str, Any] = {
    "active_bots": {},
    "stats": {
        "total_bots_deployed": 0,
        "active_connections": 0
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    """
    # Startup
    logger.info("Starting Meeting Agent API...")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"API Port: {settings.api_port}")
    logger.info(f"WebSocket Port: {settings.websocket_port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Meeting Agent API...")
    
    # Clean up active bots
    for bot_id in list(app_state["active_bots"].keys()):
        logger.info(f"Cleaning up bot {bot_id}")
        del app_state["active_bots"][bot_id]
    
    logger.info("Meeting Agent API shutdown complete")


# Initialize FastAPI application
app = FastAPI(
    title="Meeting Agent API",
    description="""
    AI-powered meeting agent with two-way audio capabilities.
    
    ## Features
    * Deploy bots to Zoom, Google Meet, and Microsoft Teams
    * Real-time transcription and conversation
    * Two-way audio interaction
    * Context-aware responses
    * Question relay system
    """,
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(
    bots.router,
    prefix="/api/v1",
    tags=["Bots"]
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Meeting Agent API",
        "version": "0.1.0",
        "status": "running",
        "documentation": "/docs" if settings.debug else "Disabled in production",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.server_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
