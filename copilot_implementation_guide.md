# LLM Copilot Implementation Guide: Meeting Agent with Two-Way Audio

## Table of Contents
1. [Understanding the Project](#understanding-the-project)
2. [Prerequisites Checklist](#prerequisites-checklist)
3. [Step-by-Step Implementation Prompts](#step-by-step-implementation-prompts)
4. [Troubleshooting Prompts](#troubleshooting-prompts)
5. [Testing & Validation Prompts](#testing-validation-prompts)

---

## Understanding the Project

### What You're Building
You are implementing a **Meeting Agent** that can:
1. **Join live meetings** on Zoom, Google Meet, or Microsoft Teams
2. **Listen to conversations** and transcribe them in real-time
3. **Speak in meetings** using synthesized voice
4. **Answer questions** based on meeting context
5. **Accept commands** from users to ask questions in the meeting

### Core Concepts You Need to Understand

#### 1. Meeting Bot
- **Definition**: A virtual participant that joins online meetings programmatically
- **How it works**: Uses meeting platform APIs to appear as a participant with audio/video capabilities
- **Your implementation**: Uses Meeting BaaS API to handle platform-specific complexities

#### 2. Two-Way Audio
- **Input (Listening)**: Audio from meeting → Speech-to-Text → Text transcript
- **Output (Speaking)**: Text message → Text-to-Speech → Audio to meeting
- **Real-time requirement**: Must process audio with <2 second latency

#### 3. WebSocket
- **Definition**: Protocol for persistent, bidirectional communication between client and server
- **Why needed**: HTTP is request-response only; WebSocket allows continuous audio streaming
- **In your system**: Carries audio data between meeting bot and your processing pipeline

#### 4. Pipeline Architecture
- **Definition**: Series of processing steps where output of one becomes input of next
- **Your pipeline**: Audio → STT → LLM → TTS → Audio
- **Framework used**: Pipecat handles the pipeline orchestration

### System Architecture Overview
```
[Meeting Platform] ←→ [Meeting BaaS Bot] ←→ [Your Server] ←→ [AI Services]
     (Zoom)            (Virtual Participant)    (FastAPI)     (OpenAI/Deepgram)
```

---

## Prerequisites Checklist

### Required Accounts and API Keys

Before starting, ensure you have:

```markdown
□ Meeting BaaS Account
  - Sign up at: https://meetingbaas.com
  - Get API key from dashboard
  - Store as: MEETING_BAAS_API_KEY

□ Deepgram Account (for Speech-to-Text)
  - Sign up at: https://deepgram.com
  - Create API key in console
  - Store as: DEEPGRAM_API_KEY

□ OpenAI Account (for LLM)
  - Sign up at: https://platform.openai.com
  - Generate API key
  - Store as: OPENAI_API_KEY

□ Cartesia Account (for Text-to-Speech)
  - Sign up at: https://cartesia.ai
  - Get API key
  - Store as: CARTESIA_API_KEY
```

### Development Environment Setup

```markdown
□ Python 3.11+ installed
  - Check: python --version
  - If not: Download from python.org

□ Poetry installed (Python package manager)
  - Check: poetry --version
  - If not: curl -sSL https://install.python-poetry.org | python3 -

□ Docker installed (for containerization)
  - Check: docker --version
  - If not: Download Docker Desktop

□ Git installed (version control)
  - Check: git --version
  - If not: Download from git-scm.com

□ Code editor (VS Code recommended)
  - Download from: code.visualstudio.com
  - Install Python extension
```

---

## Step-by-Step Implementation Prompts

### PHASE 1: Project Initialization (Day 1)

#### Prompt 1.1: Create Project Structure

**Copy and execute this prompt:**
```
Create a new Python project for a meeting agent with the following structure:

1. Create main directory: meeting-agent/
2. Inside, create these subdirectories:
   - api/ (for FastAPI server code)
   - websocket/ (for WebSocket handlers)
   - pipeline/ (for audio processing)
   - models/ (for data models)
   - config/ (for configuration)
   - tests/ (for test files)
   - docker/ (for Docker files)
   - scripts/ (for utility scripts)

3. Create these files in root:
   - pyproject.toml (Poetry configuration)
   - .env.example (environment variables template)
   - .gitignore (Git ignore patterns)
   - README.md (project documentation)
   - docker-compose.yml (Docker orchestration)

4. In each subdirectory, create an __init__.py file to make it a Python package

Output the complete directory structure and initial file contents.
```

**Expected Output Structure:**
```
meeting-agent/
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│       ├── __init__.py
│       └── bots.py
├── websocket/
│   ├── __init__.py
│   └── handler.py
├── pipeline/
│   ├── __init__.py
│   └── audio_processor.py
├── models/
│   ├── __init__.py
│   └── schemas.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.websocket
├── scripts/
│   ├── __init__.py
│   └── setup.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── docker-compose.yml
```

#### Prompt 1.2: Initialize Poetry Project

**Copy and execute this prompt:**
```
Initialize a Poetry project with the following specifications:

1. Project name: meeting-agent
2. Version: 0.1.0
3. Description: "AI-powered meeting agent with two-way audio capabilities"
4. Python version: ^3.11
5. Add these dependencies:
   - fastapi (^0.104.0) - Web framework
   - uvicorn[standard] (^0.24.0) - ASGI server
   - websockets (^12.0) - WebSocket client/server
   - pydantic (^2.5.0) - Data validation
   - python-dotenv (^1.0.0) - Environment variables
   - httpx (^0.25.0) - HTTP client
   - pipecat-ai (^0.0.39) - Audio pipeline framework
   - deepgram-sdk (^3.0.0) - Speech-to-text
   - openai (^1.6.0) - LLM and TTS
   - redis (^5.0.0) - Caching and pub/sub
   - asyncio (built-in) - Asynchronous programming

6. Add dev dependencies:
   - pytest (^7.4.0) - Testing
   - pytest-asyncio (^0.21.0) - Async testing
   - black (^23.12.0) - Code formatting
   - flake8 (^6.1.0) - Linting
   - mypy (^1.7.0) - Type checking

Generate the complete pyproject.toml file with explanations for each dependency.
```

**Expected pyproject.toml:**
```toml
[tool.poetry]
name = "meeting-agent"
version = "0.1.0"
description = "AI-powered meeting agent with two-way audio capabilities"
authors = ["Your Name <email@example.com>"]
readme = "README.md"
python = "^3.11"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.0"        # Modern web framework for building APIs
uvicorn = {extras = ["standard"], version = "^0.24.0"}  # ASGI server for FastAPI
websockets = "^12.0"        # WebSocket protocol implementation
pydantic = "^2.5.0"         # Data validation using Python type hints
python-dotenv = "^1.0.0"    # Load environment variables from .env file
httpx = "^0.25.0"           # Modern HTTP client for API calls
pipecat-ai = "^0.0.39"      # Framework for audio pipeline processing
deepgram-sdk = "^3.0.0"     # Deepgram API for speech-to-text
openai = "^1.6.0"           # OpenAI API for LLM and TTS
redis = "^5.0.0"            # In-memory data store for caching
aioredis = "^2.0.1"         # Async Redis client

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"           # Testing framework
pytest-asyncio = "^0.21.0"  # Pytest support for asyncio
black = "^23.12.0"          # Code formatter
flake8 = "^6.1.0"           # Style guide enforcement
mypy = "^1.7.0"             # Static type checker
ipython = "^8.18.0"         # Enhanced Python shell for debugging

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88
target-version = ['py311']

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
strict_optional = true
```

#### Prompt 1.3: Create Environment Variables Template

**Copy and execute this prompt:**
```
Create a comprehensive .env.example file with all required environment variables for the meeting agent system. Include:

1. API Keys section (all required external services)
2. Server Configuration (ports, hosts, URLs)
3. Audio Settings (sample rates, formats)
4. Bot Behavior Settings (timeouts, limits)
5. Database Configuration (Redis, PostgreSQL)
6. Logging Configuration

For each variable:
- Provide a descriptive comment
- Show example/default value
- Indicate if required or optional
- Explain valid value ranges where applicable

Also create a corresponding config/settings.py file that loads and validates these environment variables using Pydantic.
```

**Expected .env.example:**
```bash
# ============================================
# MEETING AGENT CONFIGURATION
# ============================================
# Copy this file to .env and fill in your values

# --------------------------------------------
# REQUIRED: External Service API Keys
# --------------------------------------------

# Meeting BaaS API Key (Required)
# Get from: https://meetingbaas.com/dashboard
MEETING_BAAS_API_KEY=your_meeting_baas_api_key_here

# Deepgram API Key (Required for Speech-to-Text)
# Get from: https://console.deepgram.com
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# OpenAI API Key (Required for LLM and optional TTS)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your_openai_api_key_here

# Cartesia API Key (Optional - alternative TTS)
# Get from: https://app.cartesia.ai
CARTESIA_API_KEY=your_cartesia_api_key_here

# --------------------------------------------
# Server Configuration
# --------------------------------------------

# API Server Port (Default: 8000)
API_PORT=8000

# WebSocket Server Port (Default: 8001)
WEBSOCKET_PORT=8001

# Server Host (Default: 0.0.0.0 for all interfaces)
SERVER_HOST=0.0.0.0

# Base URL for production (Required in production)
# Example: https://your-domain.com
BASE_URL=http://localhost:8000

# WebSocket URL (Auto-generated from BASE_URL if not set)
# Example: wss://your-domain.com
WEBSOCKET_URL=

# --------------------------------------------
# Audio Configuration
# --------------------------------------------

# Audio Sample Rate in Hz (Default: 16000)
# Valid values: 8000, 16000, 24000, 48000
AUDIO_SAMPLE_RATE=16000

# Audio Channels (Default: 1 for mono)
# Valid values: 1 (mono), 2 (stereo)
AUDIO_CHANNELS=1

# Audio Chunk Size in bytes (Default: 320)
# Should be multiple of 160 for 16kHz mono
AUDIO_CHUNK_SIZE=320

# Audio Format (Default: pcm_s16le)
# Valid values: pcm_s16le, pcm_f32le
AUDIO_FORMAT=pcm_s16le

# --------------------------------------------
# Bot Behavior Configuration
# --------------------------------------------

# Bot Default Name (Default: AI Assistant)
BOT_DEFAULT_NAME=AI Assistant

# Bot Entry Message (Default: greeting)
BOT_ENTRY_MESSAGE=Hello! I'm your AI assistant. How can I help you today?

# Bot Response Timeout in seconds (Default: 30)
BOT_RESPONSE_TIMEOUT=30

# Maximum Speaking Duration in seconds (Default: 60)
BOT_MAX_SPEAKING_DURATION=60

# Voice Activity Detection Threshold (0.0-1.0, Default: 0.5)
VAD_THRESHOLD=0.5

# Minimum Speech Duration in seconds (Default: 0.3)
MIN_SPEECH_DURATION=0.3

# Maximum Silence Duration in seconds (Default: 1.0)
MAX_SILENCE_DURATION=1.0

# --------------------------------------------
# LLM Configuration
# --------------------------------------------

# OpenAI Model (Default: gpt-4)
# Valid values: gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo
OPENAI_MODEL=gpt-4

# Maximum Tokens for LLM Response (Default: 150)
MAX_LLM_TOKENS=150

# LLM Temperature (0.0-2.0, Default: 0.7)
LLM_TEMPERATURE=0.7

# --------------------------------------------
# Speech Services Configuration
# --------------------------------------------

# Deepgram Model (Default: nova-2)
# Valid values: nova-2, nova, enhanced, base
DEEPGRAM_MODEL=nova-2

# Deepgram Language (Default: en-US)
DEEPGRAM_LANGUAGE=en-US

# TTS Provider (Default: cartesia)
# Valid values: cartesia, openai, elevenlabs
TTS_PROVIDER=cartesia

# TTS Voice ID (Provider-specific)
# Cartesia: professional-female, casual-male, etc.
# OpenAI: alloy, echo, fable, onyx, nova, shimmer
TTS_VOICE_ID=professional-female

# --------------------------------------------
# Database Configuration
# --------------------------------------------

# Redis Host (Default: localhost)
REDIS_HOST=localhost

# Redis Port (Default: 6379)
REDIS_PORT=6379

# Redis Password (Optional)
REDIS_PASSWORD=

# Redis Database Number (Default: 0)
REDIS_DB=0

# PostgreSQL Connection String (Optional)
# Format: postgresql://user:password@host:port/database
DATABASE_URL=postgresql://meeting_agent:password@localhost:5432/meeting_agent_db

# --------------------------------------------
# Logging Configuration
# --------------------------------------------

# Log Level (Default: INFO)
# Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Log Format (Default: json)
# Valid values: json, text
LOG_FORMAT=json

# Log File Path (Optional, logs to stdout if not set)
LOG_FILE_PATH=logs/meeting_agent.log

# --------------------------------------------
# Security Configuration
# --------------------------------------------

# API Key for your service (Required for production)
# Generate with: openssl rand -hex 32
API_SECRET_KEY=your_secret_key_here

# CORS Allowed Origins (comma-separated, Default: *)
CORS_ALLOWED_ORIGINS=*

# Rate Limiting (requests per minute, Default: 60)
RATE_LIMIT_PER_MINUTE=60

# --------------------------------------------
# Development/Debug Settings
# --------------------------------------------

# Debug Mode (Default: false)
# Set to true for development
DEBUG=false

# Hot Reload (Default: false)
# Set to true for development
HOT_RELOAD=false

# Mock External APIs (Default: false)
# Set to true for testing without real API calls
MOCK_EXTERNAL_APIS=false
```

#### Prompt 1.4: Create Configuration Loader

**Copy and execute this prompt:**
```
Create a Python configuration module (config/settings.py) that:

1. Uses Pydantic BaseSettings to load and validate environment variables
2. Provides type hints for all configuration values
3. Implements validation for ranges and formats
4. Provides sensible defaults where appropriate
5. Raises clear errors for missing required values
6. Implements a singleton pattern for configuration access
7. Includes helper methods for common configuration tasks

The module should:
- Load from .env file automatically
- Validate all API keys are present
- Ensure ports are valid integers
- Validate URLs are properly formatted
- Check audio settings are within valid ranges
- Provide methods to get WebSocket URL from base URL

Include comprehensive docstrings and type hints.
```

**Expected config/settings.py:**
```python
"""
Configuration management for Meeting Agent.
Loads and validates environment variables using Pydantic.
"""

from typing import Optional, List, Literal
from pydantic import BaseSettings, Field, validator, HttpUrl, SecretStr
from functools import lru_cache
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Attributes are automatically loaded from environment variables
    with the same name (case-insensitive). Values are validated
    according to their type hints and custom validators.
    """
    
    # ===== API Keys =====
    meeting_baas_api_key: SecretStr = Field(
        ...,
        description="Meeting BaaS API key for bot deployment"
    )
    
    deepgram_api_key: SecretStr = Field(
        ...,
        description="Deepgram API key for speech-to-text"
    )
    
    openai_api_key: SecretStr = Field(
        ...,
        description="OpenAI API key for LLM and optional TTS"
    )
    
    cartesia_api_key: Optional[SecretStr] = Field(
        None,
        description="Cartesia API key for text-to-speech"
    )
    
    # ===== Server Configuration =====
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port"
    )
    
    websocket_port: int = Field(
        default=8001,
        ge=1,
        le=65535,
        description="WebSocket server port"
    )
    
    server_host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    
    base_url: Optional[HttpUrl] = Field(
        None,
        description="Base URL for production deployment"
    )
    
    websocket_url: Optional[str] = Field(
        None,
        description="WebSocket URL (auto-generated if not provided)"
    )
    
    # ===== Audio Configuration =====
    audio_sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz"
    )
    
    audio_channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of audio channels"
    )
    
    audio_chunk_size: int = Field(
        default=320,
        gt=0,
        description="Audio chunk size in bytes"
    )
    
    audio_format: Literal["pcm_s16le", "pcm_f32le"] = Field(
        default="pcm_s16le",
        description="Audio format"
    )
    
    # ===== Bot Behavior =====
    bot_default_name: str = Field(
        default="AI Assistant",
        description="Default name for bots"
    )
    
    bot_entry_message: str = Field(
        default="Hello! I'm your AI assistant. How can I help you today?",
        description="Default entry message for bots"
    )
    
    bot_response_timeout: int = Field(
        default=30,
        gt=0,
        le=300,
        description="Bot response timeout in seconds"
    )
    
    bot_max_speaking_duration: int = Field(
        default=60,
        gt=0,
        le=300,
        description="Maximum speaking duration in seconds"
    )
    
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice activity detection threshold"
    )
    
    min_speech_duration: float = Field(
        default=0.3,
        gt=0.0,
        description="Minimum speech duration in seconds"
    )
    
    max_silence_duration: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum silence duration in seconds"
    )
    
    # ===== LLM Configuration =====
    openai_model: str = Field(
        default="gpt-4",
        description="OpenAI model to use"
    )
    
    max_llm_tokens: int = Field(
        default=150,
        gt=0,
        le=4000,
        description="Maximum tokens for LLM response"
    )
    
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation"
    )
    
    # ===== Speech Services =====
    deepgram_model: str = Field(
        default="nova-2",
        description="Deepgram model for STT"
    )
    
    deepgram_language: str = Field(
        default="en-US",
        description="Language for Deepgram STT"
    )
    
    tts_provider: Literal["cartesia", "openai", "elevenlabs"] = Field(
        default="cartesia",
        description="TTS provider to use"
    )
    
    tts_voice_id: str = Field(
        default="professional-female",
        description="Voice ID for TTS"
    )
    
    # ===== Database Configuration =====
    redis_host: str = Field(
        default="localhost",
        description="Redis host"
    )
    
    redis_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port"
    )
    
    redis_password: Optional[SecretStr] = Field(
        None,
        description="Redis password"
    )
    
    redis_db: int = Field(
        default=0,
        ge=0,
        description="Redis database number"
    )
    
    database_url: Optional[str] = Field(
        None,
        description="PostgreSQL connection string"
    )
    
    # ===== Logging Configuration =====
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log format"
    )
    
    log_file_path: Optional[Path] = Field(
        None,
        description="Log file path"
    )
    
    # ===== Security Configuration =====
    api_secret_key: SecretStr = Field(
        default="change-me-in-production",
        description="Secret key for API authentication"
    )
    
    cors_allowed_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    rate_limit_per_minute: int = Field(
        default=60,
        gt=0,
        description="Rate limit per minute"
    )
    
    # ===== Development Settings =====
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    hot_reload: bool = Field(
        default=False,
        description="Enable hot reload"
    )
    
    mock_external_apis: bool = Field(
        default=False,
        description="Mock external API calls"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("audio_sample_rate")
    def validate_sample_rate(cls, v):
        """Validate audio sample rate is a standard value."""
        valid_rates = [8000, 16000, 24000, 48000]
        if v not in valid_rates:
            raise ValueError(f"Sample rate must be one of {valid_rates}")
        return v
    
    @validator("audio_chunk_size")
    def validate_chunk_size(cls, v, values):
        """Validate chunk size is appropriate for sample rate."""
        if "audio_sample_rate" in values:
            sample_rate = values["audio_sample_rate"]
            # Chunk size should be a multiple of sample_rate / 100
            chunk_samples = sample_rate // 100
            if v % chunk_samples != 0:
                raise ValueError(
                    f"Chunk size should be multiple of {chunk_samples} "
                    f"for sample rate {sample_rate}"
                )
        return v
    
    @validator("websocket_url", always=True)
    def generate_websocket_url(cls, v, values):
        """Generate WebSocket URL from base URL if not provided."""
        if v:
            return v
        
        if "base_url" in values and values["base_url"]:
            base = str(values["base_url"])
            ws_url = base.replace("https://", "wss://")
            ws_url = ws_url.replace("http://", "ws://")
            return ws_url
        
        # Fallback to localhost
        if "websocket_port" in values:
            return f"ws://localhost:{values['websocket_port']}"
        
        return "ws://localhost:8001"
    
    @validator("cors_allowed_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        password = self.redis_password.get_secret_value() if self.redis_password else ""
        if password:
            return f"redis://:{password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def get_meeting_baas_headers(self) -> dict:
        """Get headers for Meeting BaaS API calls."""
        return {
            "x-meeting-baas-api-key": self.meeting_baas_api_key.get_secret_value(),
            "Content-Type": "application/json"
        }
    
    def get_deepgram_config(self) -> dict:
        """Get Deepgram configuration dictionary."""
        return {
            "api_key": self.deepgram_api_key.get_secret_value(),
            "model": self.deepgram_model,
            "language": self.deepgram_language,
            "sample_rate": self.audio_sample_rate,
            "channels": self.audio_channels,
            "encoding": "linear16" if self.audio_format == "pcm_s16le" else "float32"
        }
    
    def get_openai_config(self) -> dict:
        """Get OpenAI configuration dictionary."""
        return {
            "api_key": self.openai_api_key.get_secret_value(),
            "model": self.openai_model,
            "max_tokens": self.max_llm_tokens,
            "temperature": self.llm_temperature
        }
    
    def get_tts_config(self) -> dict:
        """Get TTS configuration based on provider."""
        config = {
            "provider": self.tts_provider,
            "voice_id": self.tts_voice_id,
            "sample_rate": self.audio_sample_rate
        }
        
        if self.tts_provider == "cartesia" and self.cartesia_api_key:
            config["api_key"] = self.cartesia_api_key.get_secret_value()
        elif self.tts_provider == "openai":
            config["api_key"] = self.openai_api_key.get_secret_value()
        
        return config
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug and self.base_url is not None
    
    def validate_required_services(self) -> List[str]:
        """
        Validate all required external services are configured.
        Returns list of missing services.
        """
        missing = []
        
        if not self.meeting_baas_api_key:
            missing.append("Meeting BaaS API key")
        
        if not self.deepgram_api_key:
            missing.append("Deepgram API key")
        
        if not self.openai_api_key:
            missing.append("OpenAI API key")
        
        if self.tts_provider == "cartesia" and not self.cartesia_api_key:
            missing.append("Cartesia API key (required for selected TTS provider)")
        
        return missing


@lru_cache()
def get_settings() -> Settings:
    """
    Get settings singleton instance.
    
    This function uses LRU cache to ensure only one instance
    of settings is created and reused throughout the application.
    
    Returns:
        Settings: Application settings instance
    
    Raises:
        ValidationError: If required environment variables are missing
        or have invalid values
    """
    return Settings()


# Convenience function for importing
settings = get_settings()
```

---

### PHASE 2: Core API Implementation (Day 2-3)

#### Prompt 2.1: Create FastAPI Main Application

**Copy and execute this prompt:**
```
Create the main FastAPI application (api/main.py) with the following requirements:

1. Initialize FastAPI with proper metadata (title, version, description)
2. Configure CORS middleware for cross-origin requests
3. Set up exception handlers for common errors
4. Implement health check endpoint
5. Include API versioning (v1)
6. Add request ID middleware for tracking
7. Set up structured logging
8. Include startup and shutdown events
9. Add rate limiting middleware
10. Include API documentation customization

The application should:
- Load configuration from settings
- Initialize connections to Redis
- Set up background tasks
- Include comprehensive error handling
- Provide detailed OpenAPI documentation
- Support both development and production modes

Include detailed comments explaining each component.
```

**Expected api/main.py:**
```python
"""
Main FastAPI application for Meeting Agent.

This module initializes the FastAPI application with all necessary
middleware, routers, and configurations for running the meeting agent API.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

import aioredis
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import get_settings
from api.routes import bots, meetings, health
from api.exceptions import MeetingAgentException, BotNotFoundException
from api.utils.logging import setup_logging, get_logger

# Get settings and logger
settings = get_settings()
logger = get_logger(__name__)

# Global state
app_state: Dict[str, Any] = {
    "redis": None,
    "active_bots": {},
    "stats": {
        "total_requests": 0,
        "total_bots_deployed": 0,
        "active_connections": 0
    }
}


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracking."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add to logs
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown"
            }
        )
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log completion
        logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time
            }
        )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, calls: int = 60, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)
        
        # Check rate limit
        now = time.time()
        if client not in self.clients:
            self.clients[client] = []
        
        # Remove old entries
        self.clients[client] = [
            timestamp for timestamp in self.clients[client]
            if timestamp > now - self.period
        ]
        
        # Check if limit exceeded
        if len(self.clients[client]) >= self.calls:
            logger.warning(f"Rate limit exceeded for {client}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.calls} requests per {self.period} seconds"
                }
            )
        
        # Add current request
        self.clients[client].append(now)
        
        # Process request
        response = await call_next(request)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    This context manager handles startup and shutdown tasks,
    ensuring proper initialization and cleanup of resources.
    """
    # Startup
    logger.info("Starting Meeting Agent API...")
    
    try:
        # Validate configuration
        missing_services = settings.validate_required_services()
        if missing_services:
            logger.error(f"Missing required services: {missing_services}")
            if settings.is_production():
                raise RuntimeError(f"Cannot start in production without: {missing_services}")
            else:
                logger.warning("Running in development mode with missing services")
        
        # Initialize Redis connection
        if not settings.mock_external_apis:
            try:
                app_state["redis"] = await aioredis.create_redis_pool(
                    settings.get_redis_url(),
                    minsize=5,
                    maxsize=10
                )
                logger.info("Connected to Redis successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                if settings.is_production():
                    raise
                logger.warning("Continuing without Redis in development mode")
        
        # Initialize background tasks
        asyncio.create_task(cleanup_inactive_bots())
        asyncio.create_task(collect_metrics())
        
        logger.info(
            f"Meeting Agent API started successfully",
            extra={
                "environment": "production" if settings.is_production() else "development",
                "debug": settings.debug,
                "api_port": settings.api_port,
                "websocket_port": settings.websocket_port
            }
        )
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down Meeting Agent API...")
        
        # Close Redis connection
        if app_state["redis"]:
            app_state["redis"].close()
            await app_state["redis"].wait_closed()
            logger.info("Redis connection closed")
        
        # Clean up active bots
        for bot_id in list(app_state["active_bots"].keys()):
            try:
                # Attempt to remove bot from meeting
                logger.info(f"Cleaning up bot {bot_id}")
                # TODO: Call Meeting BaaS API to remove bot
            except Exception as e:
                logger.error(f"Failed to clean up bot {bot_id}: {e}")
        
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
    
    ## Authentication
    Include `X-API-Key` header with your API key for all requests.
    """,
    version="0.1.0",
    docs_url="/docs" if not settings.is_production() else None,
    redoc_url="/redoc" if not settings.is_production() else None,
    openapi_url="/openapi.json" if not settings.is_production() else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    calls=settings.rate_limit_per_minute,
    period=60
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.is_production():
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[settings.base_url.host if settings.base_url else "*"]
    )


# Exception handlers
@app.exception_handler(MeetingAgentException)
async def meeting_agent_exception_handler(request: Request, exc: MeetingAgentException):
    """Handle custom Meeting Agent exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_type,
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


@app.exception_handler(BotNotFoundException)
async def bot_not_found_handler(request: Request, exc: BotNotFoundException):
    """Handle bot not found exceptions."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "bot_not_found",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        f"Unexpected error: {exc}",
        exc_info=True,
        extra={"request_id": getattr(request.state, "request_id", "unknown")}
    )
    
    if settings.debug:
        # In debug mode, return detailed error
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": str(exc),
                "type": type(exc).__name__,
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )
    else:
        # In production, return generic error
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(
    bots.router,
    prefix="/api/v1",
    tags=["Bots"]
)
app.include_router(
    meetings.router,
    prefix="/api/v1",
    tags=["Meetings"]
)


# Background tasks
async def cleanup_inactive_bots():
    """Periodically clean up inactive bots."""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            
            current_time = time.time()
            inactive_bots = []
            
            for bot_id, bot_data in app_state["active_bots"].items():
                # Check if bot has been inactive for more than 30 minutes
                if current_time - bot_data.get("last_activity", current_time) > 1800:
                    inactive_bots.append(bot_id)
            
            for bot_id in inactive_bots:
                logger.info(f"Removing inactive bot {bot_id}")
                # TODO: Call removal logic
                del app_state["active_bots"][bot_id]
            
            if inactive_bots:
                logger.info(f"Cleaned up {len(inactive_bots)} inactive bots")
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")


async def collect_metrics():
    """Collect and store metrics."""
    while True:
        try:
            await asyncio.sleep(60)  # Collect every minute
            
            # Update metrics
            app_state["stats"]["active_bots"] = len(app_state["active_bots"])
            
            # Store in Redis if available
            if app_state["redis"]:
                await app_state["redis"].setex(
                    "metrics:active_bots",
                    60,
                    len(app_state["active_bots"])
                )
            
            logger.debug(
                f"Metrics collected",
                extra={"stats": app_state["stats"]}
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Meeting Agent API",
        "version": "0.1.0",
        "status": "running",
        "documentation": "/docs" if not settings.is_production() else "Disabled in production",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Setup logging
    setup_logging(
        level=settings.log_level,
        format=settings.log_format,
        log_file=settings.log_file_path
    )
    
    # Run server
    uvicorn.run(
        "api.main:app",
        host=settings.server_host,
        port=settings.api_port,
        reload=settings.hot_reload,
        log_level=settings.log_level.lower()
    )
```

---

### Continue with more prompts...

Due to length limits, I'll create the next part of the guide in a separate file. Let me continue with the bot routes and WebSocket implementation.
