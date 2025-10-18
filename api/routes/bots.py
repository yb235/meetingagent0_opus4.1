"""
Bot management routes for Meeting Agent API.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl, field_validator
import httpx
import uuid

from config.settings import get_settings

router = APIRouter(prefix="/bots")
logger = logging.getLogger(__name__)
settings = get_settings()

# Meeting BaaS API URL
MEETING_BAAS_API_URL = "https://api.meetingbaas.com"


# ===== Pydantic Models =====

class BotDeployRequest(BaseModel):
    """Request model for bot deployment."""
    
    meeting_url: HttpUrl = Field(
        ...,
        description="URL of the meeting to join",
        examples=["https://meet.google.com/abc-defg-hij"]
    )
    
    bot_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=50,
        description="Name for the bot",
        examples=["AI Assistant"]
    )
    
    entry_message: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Message to send when bot joins",
        examples=["Hello! I'm here to help with your meeting."]
    )
    
    @field_validator("meeting_url")
    @classmethod
    def validate_meeting_url(cls, v):
        """Validate meeting URL is from supported platform."""
        url_str = str(v)
        supported_platforms = [
            "zoom.us",
            "meet.google.com",
            "teams.microsoft.com",
            "teams.live.com"
        ]
        
        if not any(platform in url_str for platform in supported_platforms):
            raise ValueError(
                f"Unsupported meeting platform. Supported: {supported_platforms}"
            )
        
        return v


class BotResponse(BaseModel):
    """Response model for bot operations."""
    
    success: bool = Field(..., description="Whether operation was successful")
    bot_id: str = Field(..., description="Unique bot identifier")
    status: str = Field(..., description="Current bot status")
    message: Optional[str] = Field(None, description="Additional message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


class SpeakRequest(BaseModel):
    """Request model for bot speech."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Message for bot to speak",
        examples=["Let me summarize the key points discussed so far."]
    )


# ===== Route Handlers =====

@router.post("/deploy", response_model=BotResponse)
async def deploy_bot(request: BotDeployRequest):
    """
    Deploy a new bot to a meeting.
    
    This endpoint creates a new bot instance and deploys it to the specified
    meeting. The bot will join the meeting and begin processing audio.
    """
    try:
        # Generate unique bot ID
        bot_id = f"bot_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Deploying bot {bot_id} to {request.meeting_url}")
        
        # Prepare WebSocket URLs for audio streaming
        ws_base = settings.get_websocket_url()
        streaming_config = {
            "input": f"{ws_base}/ws/{bot_id}/audio/in",
            "output": f"{ws_base}/ws/{bot_id}/audio/out",
            "audio_frequency": f"{settings.audio_sample_rate}hz"
        }
        
        # Deploy bot to Meeting BaaS
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{MEETING_BAAS_API_URL}/bots/",
                headers=settings.get_meeting_baas_headers(),
                json={
                    "meeting_url": str(request.meeting_url),
                    "bot_name": request.bot_name or settings.bot_default_name,
                    "entry_message": request.entry_message or settings.bot_entry_message,
                    "streaming": streaming_config,
                    "speech_to_text": {
                        "provider": "Deepgram",
                        "api_key": settings.deepgram_api_key
                    }
                }
            )
            response.raise_for_status()
            meeting_baas_data = response.json()
        
        # Store bot information
        bot_data = {
            "bot_id": bot_id,
            "meeting_baas_id": meeting_baas_data.get("bot_id"),
            "meeting_url": str(request.meeting_url),
            "bot_name": request.bot_name or settings.bot_default_name,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "websocket_url": f"{ws_base}/ws/{bot_id}",
        }
        
        # Store in memory
        from api.main import app_state
        app_state["active_bots"][bot_id] = bot_data
        app_state["stats"]["total_bots_deployed"] += 1
        
        logger.info(f"Bot {bot_id} deployed successfully")
        
        return BotResponse(
            success=True,
            bot_id=bot_id,
            status="active",
            message="Bot deployed successfully",
            data={
                "meeting_url": str(request.meeting_url),
                "websocket_url": bot_data["websocket_url"],
                "meeting_baas_id": meeting_baas_data.get("bot_id")
            }
        )
        
    except httpx.HTTPError as e:
        logger.error(f"HTTP error during bot deployment: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to deploy bot: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during bot deployment: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Bot deployment failed: {str(e)}"
        )


@router.get("/{bot_id}", response_model=BotResponse)
async def get_bot_status(bot_id: str):
    """
    Get bot status and information.
    """
    from api.main import app_state
    bot_data = app_state["active_bots"].get(bot_id)
    
    if not bot_data:
        raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
    
    # Update last activity
    bot_data["last_activity"] = datetime.utcnow().isoformat()
    
    return BotResponse(
        success=True,
        bot_id=bot_id,
        status=bot_data["status"],
        message=f"Bot is {bot_data['status']}",
        data=bot_data
    )


@router.post("/{bot_id}/speak", response_model=BotResponse)
async def queue_bot_speech(bot_id: str, request: SpeakRequest):
    """
    Queue a message for the bot to speak.
    """
    from api.main import app_state
    
    if bot_id not in app_state["active_bots"]:
        raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
    
    # Create speech task
    speech_task = {
        "task_id": str(uuid.uuid4()),
        "bot_id": bot_id,
        "message": request.message,
        "created_at": datetime.utcnow().isoformat(),
        "status": "queued"
    }
    
    # Store in bot's speech queue
    if "speech_queue" not in app_state["active_bots"][bot_id]:
        app_state["active_bots"][bot_id]["speech_queue"] = []
    
    app_state["active_bots"][bot_id]["speech_queue"].append(speech_task)
    
    logger.info(f"Speech queued for bot {bot_id}: {request.message}")
    
    return BotResponse(
        success=True,
        bot_id=bot_id,
        status="speech_queued",
        message="Message queued for speech",
        data={"task_id": speech_task["task_id"]}
    )


@router.delete("/{bot_id}", response_model=BotResponse)
async def remove_bot(bot_id: str):
    """
    Remove bot from meeting.
    """
    from api.main import app_state
    bot_data = app_state["active_bots"].get(bot_id)
    
    if not bot_data:
        raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
    
    try:
        # Remove from Meeting BaaS
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{MEETING_BAAS_API_URL}/bots/{bot_data['meeting_baas_id']}",
                headers=settings.get_meeting_baas_headers()
            )
            response.raise_for_status()
        
        # Clean up local state
        del app_state["active_bots"][bot_id]
        
        logger.info(f"Bot {bot_id} removed successfully")
        
        return BotResponse(
            success=True,
            bot_id=bot_id,
            status="removed",
            message="Bot removed from meeting",
            data={"removed_at": datetime.utcnow().isoformat()}
        )
        
    except Exception as e:
        logger.error(f"Failed to remove bot {bot_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove bot: {str(e)}"
        )


@router.get("", response_model=Dict[str, Any])
async def list_active_bots():
    """
    List all active bots.
    """
    from api.main import app_state
    
    bots = list(app_state["active_bots"].values())
    
    return {
        "total": len(bots),
        "bots": bots
    }
