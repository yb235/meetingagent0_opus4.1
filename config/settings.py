"""
Configuration management for Meeting Agent.
Loads and validates environment variables using Pydantic.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # API Keys
    meeting_baas_api_key: str = Field(
        ...,
        description="Meeting BaaS API key for bot deployment"
    )
    
    deepgram_api_key: str = Field(
        ...,
        description="Deepgram API key for speech-to-text"
    )
    
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for LLM and TTS"
    )
    
    # Server Configuration
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
    
    base_url: Optional[str] = Field(
        default="http://localhost:8000",
        description="Base URL for the application"
    )
    
    # Audio Configuration
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
    
    # Bot Behavior
    bot_default_name: str = Field(
        default="AI Assistant",
        description="Default name for bots"
    )
    
    bot_entry_message: str = Field(
        default="Hello! I'm your AI assistant.",
        description="Default entry message for bots"
    )
    
    # Development Settings
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def get_websocket_url(self) -> str:
        """Get WebSocket URL from base URL."""
        if not self.base_url:
            return f"ws://localhost:{self.websocket_port}"
        
        base = self.base_url
        ws_url = base.replace("https://", "wss://")
        ws_url = ws_url.replace("http://", "ws://")
        
        # Handle port if using localhost
        if "localhost" in ws_url and str(self.api_port) in ws_url:
            ws_url = ws_url.replace(str(self.api_port), str(self.websocket_port))
        
        return ws_url
    
    def get_meeting_baas_headers(self) -> dict:
        """Get headers for Meeting BaaS API calls."""
        return {
            "x-meeting-baas-api-key": self.meeting_baas_api_key,
            "Content-Type": "application/json"
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get settings singleton instance.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()
