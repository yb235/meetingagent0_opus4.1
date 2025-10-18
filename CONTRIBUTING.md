# Contributing to Meeting Agent

Thank you for your interest in contributing to the Meeting Agent project!

## Architecture Overview

The Meeting Agent system consists of three main components:

### 1. API Server (`api/`)
- **FastAPI** application handling HTTP requests
- **Bot management** (deploy, status, remove)
- **Speech queuing** for bot responses
- **State management** in memory (extensible to Redis)

### 2. WebSocket Server (`websocket/`)
- **Real-time audio streaming** between meetings and agent
- **Bidirectional communication** for audio input/output
- **Connection lifecycle management**

### 3. Configuration (`config/`)
- **Pydantic Settings** for environment variable management
- **Validation** of all configuration values
- **Helper methods** for API integration

## Code Structure

```
meeting-agent/
├── api/                    # FastAPI server
│   ├── main.py            # Application entry point
│   └── routes/            # API route handlers
│       ├── health.py      # Health check endpoints
│       └── bots.py        # Bot management endpoints
├── websocket/             # WebSocket server
│   └── handler.py         # WebSocket handlers
├── config/                # Configuration management
│   └── settings.py        # Settings with Pydantic
├── tests/                 # Test suite
│   └── test_api.py        # API integration tests
├── scripts/               # Utility scripts
│   └── setup.py           # Development setup
├── pyproject.toml         # Poetry dependencies
├── docker-compose.yml     # Docker orchestration
└── .env.example           # Environment template
```

## Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/meetingagent0_opus4.1.git
   cd meetingagent0_opus4.1
   ```

3. **Install dependencies:**
   ```bash
   poetry install --no-root
   ```

4. **Create .env file:**
   ```bash
   cp .env.example .env
   # Add your test API keys
   ```

5. **Run tests:**
   ```bash
   poetry run pytest
   ```

## Coding Standards

### Python Style
- **PEP 8** compliance (enforced by Black)
- **Type hints** for all function parameters and returns
- **Docstrings** for all public functions and classes
- **Line length** max 88 characters (Black default)

### Code Formatting
```bash
# Format code before committing
poetry run black .

# Check formatting
poetry run black --check .
```

### Linting
```bash
# Run linter
poetry run flake8 api/ websocket/ config/ tests/
```

## Testing Guidelines

### Writing Tests
- Use **pytest** for all tests
- Mark async tests with `@pytest.mark.asyncio`
- Test both success and failure cases
- Mock external API calls

### Example Test
```python
@pytest.mark.asyncio
async def test_deploy_bot():
    """Test bot deployment endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/bots/deploy",
            json={"meeting_url": "https://meet.google.com/test"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
```

### Running Tests
```bash
# All tests
poetry run pytest

# Specific file
poetry run pytest tests/test_api.py

# With coverage
poetry run pytest --cov=api --cov=websocket
```

## Making Changes

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Follow coding standards
- Add tests for new features
- Update documentation

### 3. Test Your Changes
```bash
# Run tests
poetry run pytest

# Format code
poetry run black .

# Check linting
poetry run flake8 .
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "Description of your changes"
```

Use clear commit messages:
- `feat: Add new feature`
- `fix: Fix bug in bot deployment`
- `docs: Update API documentation`
- `test: Add tests for WebSocket handler`
- `refactor: Restructure configuration module`

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- **Clear title** describing the change
- **Description** of what and why
- **Test results** showing all tests pass
- **Screenshots** if UI changes

## Areas for Contribution

### High Priority
1. **Audio Pipeline Integration**
   - Integrate Pipecat framework
   - Add Deepgram STT processing
   - Add OpenAI/Cartesia TTS generation
   - Implement voice activity detection

2. **State Management**
   - Add Redis integration
   - Implement pub/sub for bot events
   - Add persistent storage for transcripts

3. **Error Handling**
   - Improve error messages
   - Add retry logic for API calls
   - Better WebSocket reconnection

### Medium Priority
4. **Monitoring**
   - Add Prometheus metrics
   - Implement health checks
   - Add performance tracking

5. **Documentation**
   - API usage examples
   - Architecture diagrams
   - Deployment guides

6. **Testing**
   - Increase test coverage
   - Add load tests
   - Add end-to-end tests

### Low Priority
7. **Features**
   - Bot personas
   - Multiple TTS voices
   - Meeting summaries
   - Action item extraction

## API Design Principles

When adding new endpoints:

1. **RESTful** - Use standard HTTP methods
2. **Versioned** - All routes under `/api/v1/`
3. **Documented** - Add docstrings and examples
4. **Validated** - Use Pydantic models
5. **Consistent** - Follow existing response format

### Response Format
```python
{
  "success": bool,
  "data": dict,
  "message": str,  # optional
  "error": str     # optional, on failure
}
```

## WebSocket Protocol

When modifying WebSocket handlers:

1. **Accept connections** immediately
2. **Handle disconnections** gracefully
3. **Log all events** for debugging
4. **Send keepalives** to prevent timeouts
5. **Validate messages** before processing

### Message Format
```json
{
  "type": "message_type",
  "bot_id": "bot_123",
  "data": {},
  "timestamp": "ISO-8601 timestamp"
}
```

## Security Considerations

- **Never commit** API keys or secrets
- **Validate all input** from external sources
- **Use environment variables** for configuration
- **Sanitize logs** to avoid leaking sensitive data
- **Rate limit** API endpoints in production

## Documentation

Update these files when making changes:

- **README.md** - For major features
- **USAGE.md** - For API changes
- **Code comments** - For complex logic
- **Docstrings** - For all public functions
- **.env.example** - For new config options

## Questions?

- Check existing documentation
- Review similar code in the project
- Ask in GitHub discussions
- Create an issue for clarification

## Code Review Process

All contributions go through code review:

1. **Automated checks** must pass (tests, linting)
2. **Manual review** by maintainers
3. **Feedback** may be provided for improvements
4. **Approval** required before merging

## Thank You!

Your contributions help make Meeting Agent better for everyone. We appreciate your time and effort!
