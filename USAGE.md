# Meeting Agent Usage Guide

## Getting Started

### Prerequisites

1. **Python 3.11+** - Check with `python --version`
2. **Poetry** - Install with: `curl -sSL https://install.python-poetry.org | python3 -`
3. **API Keys** - You'll need:
   - Meeting BaaS API key
   - Deepgram API key (for Speech-to-Text)
   - OpenAI API key (for LLM and TTS)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yb235/meetingagent0_opus4.1.git
   cd meetingagent0_opus4.1
   ```

2. **Install dependencies:**
   ```bash
   poetry install --no-root
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env file with your actual API keys
   nano .env  # or use your favorite editor
   ```

4. **Run the API server:**
   ```bash
   poetry run python api/main.py
   ```
   
   The API server will start on http://localhost:8000
   
5. **In a separate terminal, run the WebSocket server:**
   ```bash
   poetry run python websocket/handler.py
   ```
   
   The WebSocket server will start on http://localhost:8001

## Using the API

### 1. Check API Health

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "meeting-agent"
}
```

### 2. Deploy a Bot to a Meeting

```bash
curl -X POST http://localhost:8000/api/v1/bots/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_url": "https://meet.google.com/abc-defg-hij",
    "bot_name": "Meeting Assistant",
    "entry_message": "Hello! I'\''m here to help with your meeting."
  }'
```

**Response:**
```json
{
  "success": true,
  "bot_id": "bot_abc123def456",
  "status": "active",
  "message": "Bot deployed successfully",
  "data": {
    "meeting_url": "https://meet.google.com/abc-defg-hij",
    "websocket_url": "ws://localhost:8001/ws/bot_abc123def456",
    "meeting_baas_id": "mb_bot_xyz789"
  }
}
```

### 3. Get Bot Status

```bash
curl http://localhost:8000/api/v1/bots/{bot_id}
```

Replace `{bot_id}` with the actual bot ID from the deployment response.

**Response:**
```json
{
  "success": true,
  "bot_id": "bot_abc123def456",
  "status": "active",
  "message": "Bot is active",
  "data": {
    "bot_id": "bot_abc123def456",
    "meeting_baas_id": "mb_bot_xyz789",
    "meeting_url": "https://meet.google.com/abc-defg-hij",
    "bot_name": "Meeting Assistant",
    "status": "active",
    "created_at": "2024-01-15T10:30:00Z",
    "last_activity": "2024-01-15T10:35:00Z",
    "websocket_url": "ws://localhost:8001/ws/bot_abc123def456"
  }
}
```

### 4. Make Bot Speak

```bash
curl -X POST http://localhost:8000/api/v1/bots/{bot_id}/speak \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Thank you everyone for joining today'\''s meeting."
  }'
```

**Response:**
```json
{
  "success": true,
  "bot_id": "bot_abc123def456",
  "status": "speech_queued",
  "message": "Message queued for speech",
  "data": {
    "task_id": "task_xyz789"
  }
}
```

### 5. List All Active Bots

```bash
curl http://localhost:8000/api/v1/bots
```

**Response:**
```json
{
  "total": 2,
  "bots": [
    {
      "bot_id": "bot_abc123def456",
      "meeting_url": "https://meet.google.com/abc-defg-hij",
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z"
    },
    {
      "bot_id": "bot_def456ghi789",
      "meeting_url": "https://zoom.us/j/123456789",
      "status": "active",
      "created_at": "2024-01-15T10:32:00Z"
    }
  ]
}
```

### 6. Remove Bot from Meeting

```bash
curl -X DELETE http://localhost:8000/api/v1/bots/{bot_id}
```

**Response:**
```json
{
  "success": true,
  "bot_id": "bot_abc123def456",
  "status": "removed",
  "message": "Bot removed from meeting",
  "data": {
    "removed_at": "2024-01-15T10:40:00Z"
  }
}
```

## Using with Docker

### Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

The services will be available at:
- API: http://localhost:8000
- WebSocket: http://localhost:8001

## API Documentation

When running in debug mode (`DEBUG=true` in .env), you can access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Environment Variables

Key configuration options in `.env`:

```bash
# Required API Keys
MEETING_BAAS_API_KEY=your_key_here
DEEPGRAM_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Server Configuration
API_PORT=8000
WEBSOCKET_PORT=8001
SERVER_HOST=0.0.0.0
BASE_URL=http://localhost:8000

# Audio Settings
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1

# Bot Behavior
BOT_DEFAULT_NAME=AI Assistant
BOT_ENTRY_MESSAGE=Hello! I'm your AI assistant.

# Development
DEBUG=true
LOG_LEVEL=INFO
```

## Supported Meeting Platforms

The bot can join meetings on:
- ✅ **Google Meet** - `https://meet.google.com/xxx-yyyy-zzz`
- ✅ **Zoom** - `https://zoom.us/j/123456789`
- ✅ **Microsoft Teams** - `https://teams.microsoft.com/l/meetup-join/...`

## Troubleshooting

### Bot fails to deploy

**Issue:** HTTP 502 error when deploying bot

**Solution:**
1. Check Meeting BaaS API key is correct
2. Verify meeting URL is valid and accessible
3. Check Meeting BaaS service status
4. Review API server logs: `docker-compose logs api`

### WebSocket connection fails

**Issue:** Cannot connect to WebSocket

**Solution:**
1. Verify WebSocket server is running: `curl http://localhost:8001/health`
2. Check firewall allows port 8001
3. Ensure `WEBSOCKET_PORT` matches in both .env files
4. Review WebSocket server logs

### Bot joins but doesn't speak

**Issue:** Bot appears in meeting but stays silent

**Solution:**
1. Check audio configuration in .env
2. Verify TTS API keys (OpenAI/Cartesia)
3. Ensure speech queue is processing
4. Check WebSocket audio output connection
5. Review logs for TTS errors

### Audio quality issues

**Issue:** Robot voice or choppy audio

**Solution:**
1. Verify `AUDIO_SAMPLE_RATE=16000` (standard for speech)
2. Check `AUDIO_CHANNELS=1` (mono)
3. Ensure stable internet connection
4. Monitor CPU/memory usage
5. Consider using different TTS provider

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run specific test
poetry run pytest tests/test_api.py::test_health_check
```

### Code Formatting

```bash
# Format code with Black
poetry run black .

# Check formatting without changes
poetry run black --check .
```

### Linting

```bash
# Run Flake8
poetry run flake8 api/ websocket/ config/ tests/
```

## Next Steps

1. **Try deploying a bot** to a test meeting
2. **Experiment with speech** using the `/speak` endpoint
3. **Monitor bot activity** via status endpoint
4. **Review logs** to understand the flow
5. **Customize bot behavior** by modifying the code

## Support

For issues or questions:
- Review the documentation files in the repository
- Check existing issues on GitHub
- Create a new issue with detailed information

## License

MIT License - See LICENSE file for details
