"""
WebSocket handler for Meeting Agent.

Handles bidirectional audio streaming between meetings and the agent.
"""

import logging
import sys
import json
import asyncio
from typing import Dict
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from config.settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meeting Agent WebSocket")

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


@app.websocket("/ws/{bot_id}")
async def websocket_endpoint(websocket: WebSocket, bot_id: str):
    """
    Main WebSocket endpoint for bidirectional audio.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for bot {bot_id}")
    
    # Store connection
    active_connections[bot_id] = websocket
    
    try:
        while True:
            # Receive data from meeting
            try:
                data = await websocket.receive()
                
                if "bytes" in data:
                    # Audio data
                    audio_bytes = data["bytes"]
                    logger.debug(f"Received {len(audio_bytes)} bytes of audio for bot {bot_id}")
                    
                    # TODO: Process audio through STT/LLM/TTS pipeline
                    # For now, just acknowledge receipt
                    await websocket.send_json({
                        "type": "audio_received",
                        "bot_id": bot_id,
                        "bytes_received": len(audio_bytes),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                elif "text" in data:
                    # JSON message
                    message = json.loads(data["text"])
                    logger.info(f"Received message for bot {bot_id}: {message}")
                    
                    # Handle different message types
                    if message.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                
            await asyncio.sleep(0.01)  # Small delay to prevent tight loop
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for bot {bot_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket for bot {bot_id}: {e}", exc_info=True)
    finally:
        # Cleanup
        if bot_id in active_connections:
            del active_connections[bot_id]
        logger.info(f"WebSocket cleanup complete for bot {bot_id}")


@app.websocket("/ws/{bot_id}/audio/in")
async def audio_input_endpoint(websocket: WebSocket, bot_id: str):
    """
    Dedicated endpoint for Meeting BaaS audio input.
    """
    await websocket.accept()
    logger.info(f"Audio input connection established for bot {bot_id}")
    
    try:
        while True:
            # Receive audio data from meeting
            data = await websocket.receive_bytes()
            logger.debug(f"Received {len(data)} bytes on audio input for bot {bot_id}")
            
            # TODO: Process audio through STT
            # For now, just log
            
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        logger.info(f"Audio input disconnected for bot {bot_id}")
    except Exception as e:
        logger.error(f"Error in audio input for bot {bot_id}: {e}", exc_info=True)


@app.websocket("/ws/{bot_id}/audio/out")
async def audio_output_endpoint(websocket: WebSocket, bot_id: str):
    """
    Dedicated endpoint for Meeting BaaS audio output.
    """
    await websocket.accept()
    logger.info(f"Audio output connection established for bot {bot_id}")
    
    try:
        while True:
            # Wait for audio to send to meeting
            # TODO: Get audio from TTS queue
            # For now, just keep connection alive
            
            await asyncio.sleep(1.0)
            
            # Send keepalive
            try:
                await websocket.send_json({
                    "type": "keepalive",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except:
                # Connection might be closed
                break
            
    except WebSocketDisconnect:
        logger.info(f"Audio output disconnected for bot {bot_id}")
    except Exception as e:
        logger.error(f"Error in audio output for bot {bot_id}: {e}", exc_info=True)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_connections": len(active_connections),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Meeting Agent WebSocket Server",
        "version": "0.1.0",
        "active_connections": len(active_connections)
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "websocket.handler:app",
        host=settings.server_host,
        port=settings.websocket_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
