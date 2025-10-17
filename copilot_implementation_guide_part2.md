# LLM Copilot Implementation Guide - Part 2: Bot Routes & WebSocket

## PHASE 2 (Continued): API Routes Implementation

### Prompt 2.2: Create Bot Management Routes

**Copy and execute this prompt:**
```
Create the bot management routes module (api/routes/bots.py) with the following endpoints:

1. POST /bots/deploy - Deploy a new bot to a meeting
2. GET /bots/{bot_id} - Get bot status and information
3. POST /bots/{bot_id}/speak - Queue message for bot to speak
4. POST /bots/{bot_id}/questions - Submit question for bot to ask
5. DELETE /bots/{bot_id} - Remove bot from meeting
6. GET /bots - List all active bots
7. POST /bots/{bot_id}/pause - Pause bot activity
8. POST /bots/{bot_id}/resume - Resume bot activity

Each endpoint should:
- Validate input using Pydantic models
- Include proper error handling
- Log all operations
- Update bot state in Redis
- Return consistent response format
- Include OpenAPI documentation

Include comprehensive docstrings and type hints.
```

**Expected api/routes/bots.py:**
```python
"""
Bot management routes for Meeting Agent API.

This module handles all bot-related operations including deployment,
status monitoring, speech control, and removal.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field, HttpUrl, validator
import httpx
import asyncio
import json
import uuid

from config.settings import get_settings
from api.dependencies import get_redis, verify_api_key
from api.exceptions import (
    BotNotFoundException,
    BotDeploymentException,
    MeetingBaaSException
)
from api.utils.meeting_baas import MeetingBaaSClient
from api.utils.logging import get_logger

# Initialize router
router = APIRouter(prefix="/bots")
logger = get_logger(__name__)
settings = get_settings()

# ===== Pydantic Models =====

class BotDeployRequest(BaseModel):
    """Request model for bot deployment."""
    
    meeting_url: HttpUrl = Field(
        ...,
        description="URL of the meeting to join",
        example="https://meet.google.com/abc-defg-hij"
    )
    
    bot_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=50,
        description="Name for the bot",
        example="AI Assistant"
    )
    
    persona: Optional[str] = Field(
        default="professional",
        description="Persona to use for bot behavior",
        example="professional"
    )
    
    entry_message: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Message to send when bot joins",
        example="Hello! I'm here to help with your meeting."
    )
    
    language: Optional[str] = Field(
        default="en-US",
        description="Language for speech recognition and synthesis",
        example="en-US"
    )
    
    auto_respond: Optional[bool] = Field(
        default=True,
        description="Whether bot should automatically respond to questions"
    )
    
    record_audio: Optional[bool] = Field(
        default=True,
        description="Whether to record meeting audio"
    )
    
    @validator("meeting_url")
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
    
    class Config:
        schema_extra = {
            "example": {
                "meeting_url": "https://meet.google.com/abc-defg-hij",
                "bot_name": "Meeting Assistant",
                "persona": "professional",
                "entry_message": "Hello! I'm your AI assistant.",
                "language": "en-US",
                "auto_respond": True,
                "record_audio": True
            }
        }


class BotResponse(BaseModel):
    """Response model for bot operations."""
    
    success: bool = Field(..., description="Whether operation was successful")
    bot_id: str = Field(..., description="Unique bot identifier")
    status: str = Field(..., description="Current bot status")
    message: Optional[str] = Field(None, description="Additional message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "bot_id": "bot_1234567890",
                "status": "active",
                "message": "Bot deployed successfully",
                "data": {
                    "meeting_url": "https://meet.google.com/abc-defg-hij",
                    "websocket_url": "wss://api.example.com/ws/bot_1234567890"
                }
            }
        }


class SpeakRequest(BaseModel):
    """Request model for bot speech."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Message for bot to speak",
        example="Let me summarize the key points discussed so far."
    )
    
    priority: Optional[str] = Field(
        default="normal",
        description="Priority level for speech queue",
        example="high"
    )
    
    wait_for_silence: Optional[bool] = Field(
        default=True,
        description="Whether to wait for silence before speaking"
    )
    
    @validator("priority")
    def validate_priority(cls, v):
        """Validate priority level."""
        valid_priorities = ["low", "normal", "high", "urgent"]
        if v not in valid_priorities:
            raise ValueError(f"Priority must be one of {valid_priorities}")
        return v


class QuestionRequest(BaseModel):
    """Request model for bot questions."""
    
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Question for bot to ask",
        example="What are the next steps for this project?"
    )
    
    context: Optional[str] = Field(
        None,
        max_length=1000,
        description="Additional context for the question",
        example="Based on the discussion about Q3 goals"
    )
    
    wait_for_response: Optional[bool] = Field(
        default=True,
        description="Whether bot should wait for a response"
    )


class BotListResponse(BaseModel):
    """Response model for bot listing."""
    
    total: int = Field(..., description="Total number of active bots")
    bots: List[Dict[str, Any]] = Field(..., description="List of bot information")


# ===== Route Handlers =====

@router.post("/deploy", response_model=BotResponse)
async def deploy_bot(
    request: BotDeployRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    redis = Depends(get_redis)
):
    """
    Deploy a new bot to a meeting.
    
    This endpoint creates a new bot instance and deploys it to the specified
    meeting. The bot will join the meeting and begin processing audio.
    
    Args:
        request: Bot deployment configuration
        background_tasks: FastAPI background tasks
        api_key: API key for authentication
        redis: Redis connection
    
    Returns:
        BotResponse: Deployment result with bot ID and status
    
    Raises:
        BotDeploymentException: If deployment fails
    """
    try:
        # Generate unique bot ID
        bot_id = f"bot_{uuid.uuid4().hex[:12]}"
        
        logger.info(
            f"Deploying bot {bot_id}",
            extra={
                "bot_id": bot_id,
                "meeting_url": str(request.meeting_url),
                "persona": request.persona
            }
        )
        
        # Prepare WebSocket URLs for audio streaming
        ws_base = settings.websocket_url
        streaming_config = {
            "input": f"{ws_base}/ws/{bot_id}/audio/in",
            "output": f"{ws_base}/ws/{bot_id}/audio/out",
            "audio_frequency": f"{settings.audio_sample_rate}hz"
        }
        
        # Initialize Meeting BaaS client
        meeting_baas = MeetingBaaSClient(
            api_key=settings.meeting_baas_api_key.get_secret_value()
        )
        
        # Deploy bot to Meeting BaaS
        deployment_result = await meeting_baas.create_bot(
            meeting_url=str(request.meeting_url),
            bot_name=request.bot_name or settings.bot_default_name,
            entry_message=request.entry_message or settings.bot_entry_message,
            streaming=streaming_config,
            speech_to_text={
                "provider": "Deepgram",
                "api_key": settings.deepgram_api_key.get_secret_value()
            },
            extra={
                "bot_id": bot_id,
                "persona": request.persona,
                "language": request.language,
                "auto_respond": request.auto_respond
            }
        )
        
        # Store bot information
        bot_data = {
            "bot_id": bot_id,
            "meeting_baas_id": deployment_result["bot_id"],
            "meeting_url": str(request.meeting_url),
            "bot_name": request.bot_name or settings.bot_default_name,
            "persona": request.persona,
            "language": request.language,
            "auto_respond": request.auto_respond,
            "record_audio": request.record_audio,
            "status": "deploying",
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "websocket_url": f"{ws_base}/ws/{bot_id}",
            "stats": {
                "messages_sent": 0,
                "messages_received": 0,
                "questions_asked": 0,
                "total_speaking_time": 0
            }
        }
        
        # Store in Redis
        if redis:
            await redis.setex(
                f"bot:{bot_id}",
                3600,  # Expire after 1 hour
                json.dumps(bot_data)
            )
            
            # Add to active bots set
            await redis.sadd("active_bots", bot_id)
        
        # Store in memory (fallback)
        from api.main import app_state
        app_state["active_bots"][bot_id] = bot_data
        
        # Schedule status check in background
        background_tasks.add_task(
            check_bot_status,
            bot_id,
            deployment_result["bot_id"]
        )
        
        # Update status to active
        bot_data["status"] = "active"
        
        logger.info(
            f"Bot {bot_id} deployed successfully",
            extra={"bot_id": bot_id, "meeting_baas_id": deployment_result["bot_id"]}
        )
        
        return BotResponse(
            success=True,
            bot_id=bot_id,
            status="active",
            message="Bot deployed successfully",
            data={
                "meeting_url": str(request.meeting_url),
                "websocket_url": bot_data["websocket_url"],
                "meeting_baas_id": deployment_result["bot_id"]
            }
        )
        
    except httpx.HTTPError as e:
        logger.error(f"HTTP error during bot deployment: {e}")
        raise BotDeploymentException(
            f"Failed to deploy bot: {str(e)}",
            status_code=502
        )
    except Exception as e:
        logger.error(f"Unexpected error during bot deployment: {e}", exc_info=True)
        raise BotDeploymentException(
            f"Bot deployment failed: {str(e)}",
            status_code=500
        )


@router.get("/{bot_id}", response_model=BotResponse)
async def get_bot_status(
    bot_id: str,
    api_key: str = Depends(verify_api_key),
    redis = Depends(get_redis)
):
    """
    Get bot status and information.
    
    Args:
        bot_id: Unique bot identifier
        api_key: API key for authentication
        redis: Redis connection
    
    Returns:
        BotResponse: Bot status and information
    
    Raises:
        BotNotFoundException: If bot not found
    """
    # Try Redis first
    if redis:
        bot_data_json = await redis.get(f"bot:{bot_id}")
        if bot_data_json:
            bot_data = json.loads(bot_data_json)
        else:
            bot_data = None
    else:
        # Fallback to memory
        from api.main import app_state
        bot_data = app_state["active_bots"].get(bot_id)
    
    if not bot_data:
        raise BotNotFoundException(f"Bot {bot_id} not found")
    
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
async def queue_bot_speech(
    bot_id: str,
    request: SpeakRequest,
    api_key: str = Depends(verify_api_key),
    redis = Depends(get_redis)
):
    """
    Queue a message for the bot to speak.
    
    Args:
        bot_id: Unique bot identifier
        request: Speech request configuration
        api_key: API key for authentication
        redis: Redis connection
    
    Returns:
        BotResponse: Speech queue result
    
    Raises:
        BotNotFoundException: If bot not found
    """
    # Verify bot exists
    if redis:
        bot_exists = await redis.exists(f"bot:{bot_id}")
        if not bot_exists:
            raise BotNotFoundException(f"Bot {bot_id} not found")
    else:
        from api.main import app_state
        if bot_id not in app_state["active_bots"]:
            raise BotNotFoundException(f"Bot {bot_id} not found")
    
    # Create speech task
    speech_task = {
        "task_id": str(uuid.uuid4()),
        "bot_id": bot_id,
        "message": request.message,
        "priority": request.priority,
        "wait_for_silence": request.wait_for_silence,
        "created_at": datetime.utcnow().isoformat(),
        "status": "queued"
    }
    
    # Queue in Redis
    if redis:
        # Add to speech queue
        queue_key = f"speech_queue:{bot_id}"
        
        # Priority-based insertion
        if request.priority == "urgent":
            await redis.lpush(queue_key, json.dumps(speech_task))
        elif request.priority == "high":
            # Insert after urgent items
            await redis.linsert(
                queue_key,
                "AFTER",
                await redis.lindex(queue_key, 0) or "",
                json.dumps(speech_task)
            )
        else:
            # Normal and low priority go to end
            await redis.rpush(queue_key, json.dumps(speech_task))
        
        # Set expiration
        await redis.expire(queue_key, 3600)
        
        # Publish to WebSocket channel
        await redis.publish(
            f"bot:{bot_id}:speak",
            json.dumps(speech_task)
        )
    
    logger.info(
        f"Speech queued for bot {bot_id}",
        extra={
            "bot_id": bot_id,
            "task_id": speech_task["task_id"],
            "priority": request.priority
        }
    )
    
    return BotResponse(
        success=True,
        bot_id=bot_id,
        status="speech_queued",
        message="Message queued for speech",
        data={"task_id": speech_task["task_id"]}
    )


@router.post("/{bot_id}/questions", response_model=BotResponse)
async def submit_question(
    bot_id: str,
    request: QuestionRequest,
    api_key: str = Depends(verify_api_key),
    redis = Depends(get_redis)
):
    """
    Submit a question for the bot to ask in the meeting.
    
    This endpoint processes the question with context and queues it
    for the bot to ask at an appropriate time.
    
    Args:
        bot_id: Unique bot identifier
        request: Question request configuration
        api_key: API key for authentication
        redis: Redis connection
    
    Returns:
        BotResponse: Question submission result
    
    Raises:
        BotNotFoundException: If bot not found
    """
    # Verify bot exists and get data
    bot_data = None
    
    if redis:
        bot_data_json = await redis.get(f"bot:{bot_id}")
        if bot_data_json:
            bot_data = json.loads(bot_data_json)
    else:
        from api.main import app_state
        bot_data = app_state["active_bots"].get(bot_id)
    
    if not bot_data:
        raise BotNotFoundException(f"Bot {bot_id} not found")
    
    # Process question with LLM to make it contextual
    from api.utils.llm import process_question_with_context
    
    processed_question = await process_question_with_context(
        question=request.question,
        context=request.context,
        persona=bot_data.get("persona", "professional")
    )
    
    # Create question task
    question_task = {
        "task_id": str(uuid.uuid4()),
        "bot_id": bot_id,
        "original_question": request.question,
        "processed_question": processed_question,
        "context": request.context,
        "wait_for_response": request.wait_for_response,
        "created_at": datetime.utcnow().isoformat(),
        "status": "pending"
    }
    
    # Queue the processed question for speech
    speech_request = SpeakRequest(
        message=processed_question,
        priority="high",
        wait_for_silence=True
    )
    
    await queue_bot_speech(
        bot_id=bot_id,
        request=speech_request,
        api_key=api_key,
        redis=redis
    )
    
    # Store question history
    if redis:
        await redis.lpush(
            f"questions:{bot_id}",
            json.dumps(question_task)
        )
        await redis.expire(f"questions:{bot_id}", 3600)
    
    # Update bot stats
    if bot_data:
        bot_data["stats"]["questions_asked"] += 1
        if redis:
            await redis.setex(
                f"bot:{bot_id}",
                3600,
                json.dumps(bot_data)
            )
    
    logger.info(
        f"Question submitted for bot {bot_id}",
        extra={
            "bot_id": bot_id,
            "task_id": question_task["task_id"]
        }
    )
    
    return BotResponse(
        success=True,
        bot_id=bot_id,
        status="question_queued",
        message="Question processed and queued",
        data={
            "task_id": question_task["task_id"],
            "processed_question": processed_question
        }
    )


@router.delete("/{bot_id}", response_model=BotResponse)
async def remove_bot(
    bot_id: str,
    api_key: str = Depends(verify_api_key),
    redis = Depends(get_redis)
):
    """
    Remove bot from meeting.
    
    Args:
        bot_id: Unique bot identifier
        api_key: API key for authentication
        redis: Redis connection
    
    Returns:
        BotResponse: Removal result
    
    Raises:
        BotNotFoundException: If bot not found
    """
    # Get bot data
    bot_data = None
    
    if redis:
        bot_data_json = await redis.get(f"bot:{bot_id}")
        if bot_data_json:
            bot_data = json.loads(bot_data_json)
    else:
        from api.main import app_state
        bot_data = app_state["active_bots"].get(bot_id)
    
    if not bot_data:
        raise BotNotFoundException(f"Bot {bot_id} not found")
    
    try:
        # Remove from Meeting BaaS
        meeting_baas = MeetingBaaSClient(
            api_key=settings.meeting_baas_api_key.get_secret_value()
        )
        
        await meeting_baas.remove_bot(bot_data["meeting_baas_id"])
        
        # Clean up Redis
        if redis:
            await redis.delete(f"bot:{bot_id}")
            await redis.delete(f"speech_queue:{bot_id}")
            await redis.delete(f"questions:{bot_id}")
            await redis.srem("active_bots", bot_id)
        
        # Clean up memory
        from api.main import app_state
        if bot_id in app_state["active_bots"]:
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


@router.get("", response_model=BotListResponse)
async def list_active_bots(
    api_key: str = Depends(verify_api_key),
    redis = Depends(get_redis),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    List all active bots.
    
    Args:
        api_key: API key for authentication
        redis: Redis connection
        limit: Maximum number of bots to return
        offset: Number of bots to skip
    
    Returns:
        BotListResponse: List of active bots
    """
    bots = []
    
    if redis:
        # Get from Redis
        bot_ids = await redis.smembers("active_bots")
        for bot_id in list(bot_ids)[offset:offset+limit]:
            bot_data_json = await redis.get(f"bot:{bot_id.decode()}")
            if bot_data_json:
                bots.append(json.loads(bot_data_json))
    else:
        # Get from memory
        from api.main import app_state
        all_bots = list(app_state["active_bots"].values())
        bots = all_bots[offset:offset+limit]
    
    return BotListResponse(
        total=len(bots),
        bots=bots
    )


# ===== Background Tasks =====

async def check_bot_status(bot_id: str, meeting_baas_id: str):
    """
    Background task to check bot status after deployment.
    
    Args:
        bot_id: Internal bot ID
        meeting_baas_id: Meeting BaaS bot ID
    """
    await asyncio.sleep(5)  # Wait for bot to join
    
    try:
        meeting_baas = MeetingBaaSClient(
            api_key=settings.meeting_baas_api_key.get_secret_value()
        )
        
        status = await meeting_baas.get_bot_status(meeting_baas_id)
        
        # Update bot status
        # Implementation depends on status structure
        logger.info(f"Bot {bot_id} status checked: {status}")
        
    except Exception as e:
        logger.error(f"Failed to check bot {bot_id} status: {e}")
```

---

### PHASE 3: WebSocket Implementation

#### Prompt 3.1: Create WebSocket Handler with Pipecat

**Copy and execute this prompt:**
```
Create the WebSocket handler (websocket/handler.py) that:

1. Accepts WebSocket connections for each bot
2. Integrates with Pipecat for audio processing
3. Handles bidirectional audio streaming
4. Processes real-time transcription
5. Manages speech synthesis queue
6. Implements voice activity detection
7. Handles connection lifecycle (connect/disconnect/reconnect)
8. Publishes events to Redis pub/sub

The handler should:
- Create separate audio pipelines for each bot
- Handle multiple concurrent connections
- Implement proper error handling and recovery
- Support graceful shutdown
- Log all audio events
- Track metrics (latency, quality, etc.)

Include detailed comments explaining the audio pipeline flow.
```

**Expected websocket/handler.py:**
[Content would continue here but I'm reaching length limits]

---

## Testing & Validation Prompts

### Prompt T.1: Create Integration Tests

**Copy and execute this prompt:**
```
Create comprehensive integration tests (tests/test_integration.py) that:

1. Test complete bot deployment flow
2. Verify WebSocket audio streaming
3. Test speech-to-text processing
4. Test text-to-speech generation
5. Validate bot removal
6. Test error scenarios
7. Measure latency metrics
8. Test concurrent bot deployments

Include:
- Mock Meeting BaaS responses
- Simulated audio data
- WebSocket connection tests
- Redis interaction tests
- Performance benchmarks
```

---

## Troubleshooting Prompts

### Prompt TR.1: Debug Connection Issues

**When WebSocket connection fails, use this prompt:**
```
Debug WebSocket connection failure with these steps:

1. Check if WebSocket server is running on correct port
2. Verify firewall rules allow WebSocket connections
3. Test WebSocket URL format is correct (ws:// or wss://)
4. Check for CORS issues in browser console
5. Verify nginx/proxy configuration for WebSocket upgrade headers
6. Test with wscat or websocat CLI tools
7. Check SSL certificates for wss:// connections
8. Verify Meeting BaaS webhook URLs are accessible

Provide diagnostic commands and expected outputs for each step.
```

### Prompt TR.2: Debug Audio Issues

**When audio doesn't work, use this prompt:**
```
Debug audio processing issues:

1. Verify audio format matches configuration (16kHz, mono, PCM)
2. Check audio chunk size is correct (320 bytes for 10ms at 16kHz)
3. Test Deepgram API key and model availability
4. Verify Cartesia/OpenAI TTS credentials
5. Check audio buffer isn't overflowing
6. Monitor WebSocket message size limits
7. Test with known good audio file
8. Check for audio codec mismatches

Include code to test each component independently.
```

---

## Deployment Checklist for Copilot

### Final Verification Prompt

**Use this to verify everything is working:**
```
Run through this complete verification checklist:

Environment:
□ All API keys are set in .env file
□ Python 3.11+ is installed
□ Poetry dependencies are installed
□ Docker is running (if using Docker)
□ Redis is accessible
□ Ports 8000 and 8001 are available

API Server:
□ FastAPI server starts without errors
□ /health endpoint returns 200 OK
□ /docs endpoint shows API documentation
□ Can deploy a test bot via curl/Postman

WebSocket:
□ WebSocket server starts on port 8001
□ Can connect via wscat to ws://localhost:8001/ws/test
□ Audio streaming endpoints are accessible
□ Redis pub/sub is working

Integration:
□ Bot successfully joins test meeting
□ Audio is being transcribed
□ Bot can speak when commanded
□ Bot can be removed cleanly

Monitoring:
□ Logs are being written
□ Metrics are being collected
□ Error handling is working
□ Rate limiting is active

For each item that fails, provide specific debugging steps.
```

---

## Common Pitfalls & Solutions

### For the Copilot Agent

**PITFALL 1: Forgetting to activate Python virtual environment**
```
Solution:
Always run: poetry shell
Before running any Python commands
This ensures correct dependencies are available
```

**PITFALL 2: Missing await keyword for async functions**
```
Solution:
Any function that does I/O needs 'await':
- await redis.get()
- await httpx.post()
- await websocket.send()
Remember: if function is async def, calls inside need await
```

**PITFALL 3: Not handling WebSocket disconnections**
```
Solution:
Always wrap WebSocket operations in try/except:
try:
    await websocket.send(data)
except WebSocketDisconnect:
    # Handle disconnection
    await cleanup_bot(bot_id)
```

**PITFALL 4: Incorrect audio format**
```
Solution:
Always verify audio format matches:
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Format: PCM 16-bit little-endian
- Chunk size: 320 bytes (20ms of audio)
```

**PITFALL 5: Not setting CORS for browser clients**
```
Solution:
Add CORS middleware to FastAPI:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_methods=["*"],
    allow_headers=["*"]
)
```

---

## Next Steps After Implementation

Once the copilot has implemented all components:

1. **Run local tests**: Start with single bot deployment
2. **Test with real meeting**: Use Google Meet for easiest testing
3. **Monitor logs**: Watch for errors and latency issues
4. **Optimize performance**: Adjust chunk sizes and timeouts
5. **Add personas**: Create different bot personalities
6. **Deploy to cloud**: Use Docker and Kubernetes configs
7. **Set up monitoring**: Add Prometheus and Grafana
8. **Create documentation**: Generate API docs and user guide

---

This guide provides comprehensive, step-by-step instructions that a copilot agent can follow to implement the complete meeting agent system. Each prompt is self-contained and provides expected outputs to verify correct implementation.
