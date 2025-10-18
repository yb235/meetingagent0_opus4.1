# Meeting Agent

AI-powered meeting agent with two-way audio capabilities. Deploy bots to Zoom, Google Meet, and Microsoft Teams meetings for real-time transcription and interaction.

## Features

- 🤖 Deploy bots to live meetings
- 🎤 Real-time audio transcription
- 🗣️ Two-way audio interaction
- 💬 Context-aware responses
- ❓ Question relay system

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (Python package manager)
- API keys for:
  - Meeting BaaS
  - Deepgram (Speech-to-Text)
  - OpenAI (LLM and TTS)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yb235/meetingagent0_opus4.1.git
cd meetingagent0_opus4.1
```

2. Install dependencies:
```bash
poetry install
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the API server:
```bash
poetry run python api/main.py
```

5. In a separate terminal, run the WebSocket server:
```bash
poetry run python websocket/handler.py
```

## API Endpoints

### Deploy Bot
```bash
POST /api/v1/bots/deploy
{
  "meeting_url": "https://meet.google.com/abc-defg-hij",
  "bot_name": "AI Assistant"
}
```

### Get Bot Status
```bash
GET /api/v1/bots/{bot_id}
```

### Queue Speech
```bash
POST /api/v1/bots/{bot_id}/speak
{
  "message": "Hello everyone!"
}
```

### Remove Bot
```bash
DELETE /api/v1/bots/{bot_id}
```

## Development

### Run Tests
```bash
poetry run pytest
```

### Format Code
```bash
poetry run black .
```

### Lint Code
```bash
poetry run flake8 .
```

## Documentation

See the comprehensive documentation files for detailed implementation guides:
- `meeting_agent_reference.md` - Complete API reference
- `meeting_agent_architecture (2).md` - System architecture
- `meeting_agent_implementation (1).md` - Implementation guide
- `copilot_implementation_guide.md` - Step-by-step guide

## License

MIT License
