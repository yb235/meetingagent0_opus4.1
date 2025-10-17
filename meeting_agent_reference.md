# Meeting Agent Reference Documentation

## Complete API & Concept Reference for LLM Implementation

This document serves as a comprehensive reference for all concepts, APIs, and technical details needed to implement the meeting agent system.

---

## 1. CORE CONCEPTS EXPLAINED

### 1.1 What is a Meeting Bot?

**Definition**: A meeting bot is a programmatic participant that joins video conferences to perform automated tasks.

**How it works**:
```
1. Bot receives meeting URL (e.g., https://meet.google.com/abc-defg-hij)
2. Bot uses platform API to join as participant
3. Bot appears in participant list with name/avatar
4. Bot can access audio/video streams
5. Bot can send audio/video back to meeting
```

**Technical Implementation**:
- **Zoom**: Uses Zoom SDK or Meeting SDK
- **Google Meet**: Uses browser automation or Meet API
- **Microsoft Teams**: Uses Graph API or Teams SDK

**Key Capabilities**:
- Join/leave meetings programmatically
- Access real-time audio/video streams
- Send messages in chat
- Share screen content
- Speak using synthesized voice
- Record meeting content

### 1.2 Audio Processing Pipeline

**The Complete Audio Flow**:
```
Meeting Audio → Bot Receives → Speech-to-Text → LLM Processing → Text-to-Speech → Bot Sends → Meeting Hears
```

**Detailed Breakdown**:

1. **Audio Capture** (from meeting):
   - Format: PCM (Pulse Code Modulation)
   - Sample Rate: 16000 Hz (16,000 samples per second)
   - Bit Depth: 16-bit (2 bytes per sample)
   - Channels: Mono (1 channel)
   - Chunk Size: 320 bytes (20ms of audio)

2. **Speech-to-Text (STT)**:
   - Service: Deepgram or Google Speech
   - Input: Raw audio chunks
   - Output: Text transcription with timestamps
   - Features: Speaker diarization, punctuation

3. **LLM Processing**:
   - Service: OpenAI GPT-4 or Claude
   - Input: Transcribed text + context
   - Output: Generated response text
   - Context window: Recent transcript history

4. **Text-to-Speech (TTS)**:
   - Service: Cartesia, ElevenLabs, or OpenAI
   - Input: Text to speak
   - Output: Audio data (same format as input)
   - Voice selection: Different personas/styles

### 1.3 WebSocket Protocol

**What is WebSocket?**
- Full-duplex communication protocol
- Persistent connection (unlike HTTP)
- Real-time bidirectional data transfer
- Lower latency than polling

**WebSocket Lifecycle**:
```javascript
// 1. Connection Establishment
const ws = new WebSocket('ws://localhost:8001/ws/bot_123');

// 2. Connection Opened
ws.onopen = () => {
    console.log('Connected');
    // Can now send/receive data
};

// 3. Receiving Data
ws.onmessage = (event) => {
    // Handle incoming data
    const data = JSON.parse(event.data);
};

// 4. Sending Data
ws.send(JSON.stringify({ type: 'audio', data: audioBuffer }));

// 5. Connection Closed
ws.onclose = () => {
    console.log('Disconnected');
    // Implement reconnection logic
};
```

**Binary vs Text Messages**:
- Text: JSON messages for control/metadata
- Binary: Raw audio data for streaming

### 1.4 Meeting Platforms Specifics

**Google Meet**:
- URL Format: `https://meet.google.com/xxx-yyyy-zzz`
- Bot appears as regular participant
- Requires browser automation or API
- Audio quality: Up to 48kHz

**Zoom**:
- URL Format: `https://zoom.us/j/MEETING_ID?pwd=PASSWORD`
- Can use Zoom SDK for deeper integration
- Supports raw audio streaming
- Meeting passwords may be required

**Microsoft Teams**:
- URL Format: `https://teams.microsoft.com/l/meetup-join/...`
- Requires Microsoft Graph API access
- Azure AD authentication needed
- More complex permission model

---

## 2. EXTERNAL APIS DOCUMENTATION

### 2.1 Meeting BaaS API

**Base URL**: `https://api.meetingbaas.com`

**Authentication**:
```http
x-meeting-baas-api-key: YOUR_API_KEY
```

**Core Endpoints**:

#### Deploy Bot
```http
POST /bots/
Content-Type: application/json

{
  "meeting_url": "https://meet.google.com/abc-defg-hij",
  "bot_name": "AI Assistant",
  "entry_message": "Hello, I've joined the meeting",
  "streaming": {
    "input": "wss://your-server.com/ws/bot_123/audio/in",
    "output": "wss://your-server.com/ws/bot_123/audio/out",
    "audio_frequency": "16khz"
  },
  "speech_to_text": {
    "provider": "Deepgram",
    "api_key": "YOUR_DEEPGRAM_KEY"
  }
}

Response:
{
  "bot_id": "mb_bot_abc123",
  "status": "joining",
  "meeting_url": "https://meet.google.com/abc-defg-hij"
}
```

#### Remove Bot
```http
DELETE /bots/{bot_id}

Response:
{
  "success": true,
  "message": "Bot removed from meeting"
}
```

#### Get Bot Status
```http
GET /bots/{bot_id}

Response:
{
  "bot_id": "mb_bot_abc123",
  "status": "in_meeting",
  "joined_at": "2024-01-01T12:00:00Z",
  "participants": 5
}
```

### 2.2 Deepgram API (Speech-to-Text)

**WebSocket Endpoint**: `wss://api.deepgram.com/v1/listen`

**Connection Parameters**:
```javascript
const deepgramUrl = new URL('wss://api.deepgram.com/v1/listen');
deepgramUrl.searchParams.set('encoding', 'linear16');
deepgramUrl.searchParams.set('sample_rate', '16000');
deepgramUrl.searchParams.set('channels', '1');
deepgramUrl.searchParams.set('model', 'nova-2');
deepgramUrl.searchParams.set('punctuate', 'true');
deepgramUrl.searchParams.set('diarize', 'true');
deepgramUrl.searchParams.set('interim_results', 'true');

const ws = new WebSocket(deepgramUrl, {
    headers: {
        'Authorization': 'Token YOUR_DEEPGRAM_API_KEY'
    }
});
```

**Response Format**:
```json
{
  "type": "Results",
  "channel": {
    "alternatives": [{
      "transcript": "Hello, how are you today?",
      "confidence": 0.98,
      "words": [{
        "word": "Hello",
        "start": 0.0,
        "end": 0.5,
        "confidence": 0.99,
        "speaker": 0
      }]
    }]
  },
  "metadata": {
    "request_id": "abc123",
    "model_uuid": "xyz789"
  }
}
```

### 2.3 OpenAI API (LLM & TTS)

#### Chat Completion (LLM)
```python
import openai

client = openai.Client(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful meeting assistant."},
        {"role": "user", "content": "Summarize: [transcript here]"}
    ],
    temperature=0.7,
    max_tokens=150
)

answer = response.choices[0].message.content
```

#### Text-to-Speech
```python
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",  # or: echo, fable, onyx, nova, shimmer
    input="Hello, this is the AI assistant speaking.",
    speed=1.0,
    response_format="pcm"  # Raw PCM format
)

audio_data = response.content  # Raw audio bytes
```

### 2.4 Cartesia API (Alternative TTS)

**HTTP Endpoint**:
```http
POST https://api.cartesia.ai/tts/bytes
Content-Type: application/json
X-API-Key: YOUR_CARTESIA_KEY

{
  "model_id": "sonic-english",
  "voice": {
    "voice_id": "professional-female"
  },
  "transcript": "Text to speak",
  "output_format": {
    "container": "raw",
    "encoding": "pcm_s16le",
    "sample_rate": 16000
  }
}

Response: Raw audio bytes
```

### 2.5 Pipecat Framework

**Installation**:
```bash
pip install pipecat-ai
```

**Basic Pipeline Setup**:
```python
from pipecat.pipeline import Pipeline
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.frames import AudioRawFrame, TextFrame

# Initialize services
stt = DeepgramSTTService(api_key="KEY", model="nova-2")
llm = OpenAILLMService(api_key="KEY", model="gpt-4")
tts = CartesiaTTSService(api_key="KEY", voice_id="voice")

# Create pipeline
pipeline = Pipeline([stt, llm, tts])

# Process audio
async def process_audio(audio_bytes):
    frame = AudioRawFrame(
        audio=audio_bytes,
        sample_rate=16000,
        num_channels=1
    )
    result = await pipeline.process_frame(frame)
    return result
```

---

## 3. AUDIO TECHNICAL SPECIFICATIONS

### 3.1 Audio Format Details

**PCM (Pulse Code Modulation)**:
- Uncompressed audio format
- Direct digital representation of analog signal
- Each sample represents amplitude at point in time

**Sample Rate (16000 Hz)**:
- 16,000 samples per second
- Nyquist frequency: 8000 Hz (max representable frequency)
- Good for speech (human speech is 300-3400 Hz)

**Bit Depth (16-bit)**:
- Each sample uses 16 bits (2 bytes)
- Range: -32,768 to 32,767
- Dynamic range: ~96 dB

**Calculating Data Rates**:
```
Data rate = Sample Rate × Bit Depth × Channels
         = 16000 × 16 × 1
         = 256,000 bits/second
         = 32,000 bytes/second
         = 32 KB/second

For 20ms chunks:
Chunk size = 32,000 × 0.02 = 640 bytes
(Often 320 bytes used for 10ms chunks)
```

### 3.2 Audio Processing Code

**Converting Audio Formats**:
```python
import numpy as np
import struct

def pcm_to_float32(pcm_bytes):
    """Convert PCM S16LE to Float32."""
    # Unpack 16-bit signed integers
    samples = struct.unpack(f'{len(pcm_bytes)//2}h', pcm_bytes)
    # Convert to float32 (-1.0 to 1.0)
    float_samples = np.array(samples, dtype=np.float32) / 32768.0
    return float_samples

def float32_to_pcm(float_samples):
    """Convert Float32 to PCM S16LE."""
    # Scale to 16-bit range
    pcm_samples = (float_samples * 32767).astype(np.int16)
    # Pack as bytes
    pcm_bytes = struct.pack(f'{len(pcm_samples)}h', *pcm_samples)
    return pcm_bytes
```

**Audio Chunking**:
```python
def chunk_audio(audio_bytes, chunk_size=320):
    """Split audio into fixed-size chunks."""
    chunks = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i+chunk_size]
        # Pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk += b'\x00' * (chunk_size - len(chunk))
        chunks.append(chunk)
    return chunks
```

### 3.3 Voice Activity Detection (VAD)

**Purpose**: Detect when someone is speaking vs silence

**Implementation with Silero VAD**:
```python
import torch
import torchaudio

# Load VAD model
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# Process audio
def detect_speech(audio_tensor, sample_rate=16000):
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100
    )
    return speech_timestamps
```

---

## 4. ERROR HANDLING & RECOVERY

### 4.1 Common Errors and Solutions

#### WebSocket Connection Errors

**Error**: `WebSocket connection failed`
```python
# Solution: Implement exponential backoff reconnection
import asyncio

async def connect_with_retry(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            ws = await websockets.connect(url)
            return ws
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Connection failed, retry in {wait_time}s: {e}")
            await asyncio.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

#### Audio Format Mismatch

**Error**: `Invalid audio format`
```python
# Solution: Validate and convert audio format
def validate_audio_format(audio_data, expected_rate=16000):
    # Check if audio is in correct format
    try:
        # Attempt to decode as PCM
        samples = np.frombuffer(audio_data, dtype=np.int16)
        
        # Validate size
        if len(samples) == 0:
            raise ValueError("Empty audio data")
        
        # Check for clipping
        if np.max(np.abs(samples)) > 32000:
            print("Warning: Audio may be clipping")
        
        return True
    except Exception as e:
        print(f"Audio validation failed: {e}")
        return False
```

#### Meeting Platform Errors

**Error**: `Bot failed to join meeting`
```python
# Solution: Implement platform-specific fallbacks
async def deploy_bot_with_fallback(meeting_url):
    # Try primary method
    try:
        result = await deploy_via_meeting_baas(meeting_url)
        return result
    except Exception as e:
        print(f"Primary deployment failed: {e}")
    
    # Try alternative method
    try:
        result = await deploy_via_direct_api(meeting_url)
        return result
    except Exception as e:
        print(f"Fallback deployment failed: {e}")
    
    raise Exception("All deployment methods failed")
```

### 4.2 Debugging Techniques

#### Audio Stream Debugging
```python
import wave
import datetime

def save_audio_for_debugging(audio_bytes, sample_rate=16000):
    """Save audio to WAV file for debugging."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_audio_{timestamp}.wav"
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_bytes)
    
    print(f"Saved debug audio to {filename}")
    return filename
```

#### WebSocket Message Logging
```python
import json
from datetime import datetime

class WebSocketLogger:
    def __init__(self, bot_id):
        self.bot_id = bot_id
        self.log_file = f"ws_log_{bot_id}.jsonl"
    
    def log_message(self, direction, message):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "bot_id": self.bot_id,
            "direction": direction,  # "sent" or "received"
            "message": message if isinstance(message, dict) else str(message)
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

#### Performance Monitoring
```python
import time
from collections import deque

class LatencyMonitor:
    def __init__(self, window_size=100):
        self.latencies = deque(maxlen=window_size)
    
    def measure(self, func):
        """Decorator to measure function latency."""
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            latency = (time.perf_counter() - start) * 1000  # ms
            
            self.latencies.append(latency)
            
            if len(self.latencies) == self.latencies.maxlen:
                avg = sum(self.latencies) / len(self.latencies)
                p95 = sorted(self.latencies)[int(0.95 * len(self.latencies))]
                print(f"Latency - Avg: {avg:.2f}ms, P95: {p95:.2f}ms")
            
            return result
        return wrapper
```

---

## 5. TESTING UTILITIES

### 5.1 Mock Services for Testing

#### Mock Meeting BaaS
```python
from fastapi import FastAPI
import uuid

app = FastAPI()

mock_bots = {}

@app.post("/bots/")
async def create_mock_bot(request: dict):
    bot_id = f"mock_bot_{uuid.uuid4().hex[:8]}"
    mock_bots[bot_id] = {
        "bot_id": bot_id,
        "meeting_url": request["meeting_url"],
        "status": "in_meeting"
    }
    return {"bot_id": bot_id}

@app.delete("/bots/{bot_id}")
async def delete_mock_bot(bot_id: str):
    if bot_id in mock_bots:
        del mock_bots[bot_id]
    return {"success": True}

# Run with: uvicorn mock_meeting_baas:app --port 9000
```

#### Mock Audio Generator
```python
import numpy as np

def generate_test_audio(duration_seconds=1, frequency=440, sample_rate=16000):
    """Generate sine wave test audio."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Convert to PCM S16LE
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()

def generate_speech_like_audio(duration_seconds=1, sample_rate=16000):
    """Generate audio that mimics speech patterns."""
    samples = int(sample_rate * duration_seconds)
    
    # Mix of frequencies common in speech
    frequencies = [200, 500, 1000, 2000]
    audio = np.zeros(samples)
    
    for freq in frequencies:
        t = np.linspace(0, duration_seconds, samples)
        # Random amplitude for each frequency
        amplitude = np.random.uniform(0.1, 0.3)
        audio += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add some noise
    audio += np.random.normal(0, 0.01, samples)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Convert to PCM
    pcm = (audio * 32767 * 0.5).astype(np.int16)  # 50% volume
    return pcm.tobytes()
```

### 5.2 Load Testing

#### WebSocket Load Test
```python
import asyncio
import websockets
import time

async def bot_client(bot_id, duration=60):
    """Simulate a single bot connection."""
    uri = f"ws://localhost:8001/ws/{bot_id}"
    
    async with websockets.connect(uri) as websocket:
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Send test audio
            audio = generate_test_audio(0.02)  # 20ms
            await websocket.send(audio)
            
            # Wait for response
            try:
                response = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                pass
            
            await asyncio.sleep(0.02)  # 20ms intervals

async def load_test(num_bots=10, duration=60):
    """Run load test with multiple bots."""
    print(f"Starting load test with {num_bots} bots for {duration}s")
    
    tasks = [
        bot_client(f"test_bot_{i}", duration)
        for i in range(num_bots)
    ]
    
    start = time.time()
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start
    
    print(f"Load test completed in {elapsed:.2f}s")

# Run with: asyncio.run(load_test(50, 120))
```

---

## 6. PRODUCTION CHECKLIST

### 6.1 Pre-Deployment Verification

```markdown
## Environment Setup
□ All API keys configured and validated
□ SSL certificates installed for WebSocket
□ Domain name configured with DNS
□ Firewall rules allow required ports
□ Redis cluster configured for high availability
□ Database migrations completed
□ Backup strategy implemented

## Security
□ API keys stored in secure vault (not in code)
□ Rate limiting configured
□ Input validation on all endpoints
□ SQL injection prevention
□ XSS protection enabled
□ CORS properly configured
□ Authentication middleware active
□ Audit logging enabled

## Performance
□ Load testing completed (target: 100 concurrent bots)
□ Audio latency < 2 seconds end-to-end
□ Memory usage stable under load
□ CPU usage < 70% at peak
□ WebSocket reconnection tested
□ Database connection pooling configured
□ CDN configured for static assets

## Monitoring
□ Application metrics exposed (Prometheus)
□ Log aggregation configured (ELK stack)
□ Alerts configured for critical errors
□ Uptime monitoring active
□ Performance dashboards created (Grafana)
□ Error tracking configured (Sentry)
□ Health check endpoints tested

## Disaster Recovery
□ Backup schedule configured
□ Recovery procedure documented
□ Failover tested
□ Data retention policy defined
□ Incident response plan created
□ Runbook for common issues
```

### 6.2 Post-Deployment Monitoring

**Key Metrics to Track**:
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
bots_deployed = Counter('bots_deployed_total', 'Total bots deployed')
bots_failed = Counter('bots_failed_total', 'Failed bot deployments')
active_bots = Gauge('active_bots', 'Currently active bots')
meeting_minutes = Counter('meeting_minutes_total', 'Total meeting minutes processed')

# Performance metrics
audio_latency = Histogram('audio_latency_seconds', 'Audio processing latency')
stt_latency = Histogram('stt_latency_seconds', 'Speech-to-text latency')
tts_latency = Histogram('tts_latency_seconds', 'Text-to-speech latency')
websocket_messages = Counter('websocket_messages_total', 'WebSocket messages', ['direction'])

# System metrics
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
redis_connections = Gauge('redis_connections', 'Active Redis connections')
```

---

## 7. QUICK REFERENCE CARDS

### 7.1 API Endpoints Cheat Sheet

```
Meeting Agent API
├── GET  /                          # API info
├── GET  /health                    # Health check
├── GET  /docs                      # API documentation
│
├── POST   /api/v1/bots/deploy      # Deploy new bot
├── GET    /api/v1/bots/{bot_id}   # Get bot status
├── POST   /api/v1/bots/{bot_id}/speak     # Queue speech
├── POST   /api/v1/bots/{bot_id}/questions # Submit question
├── DELETE /api/v1/bots/{bot_id}   # Remove bot
├── GET    /api/v1/bots             # List all bots
│
└── WS   /ws/{bot_id}               # WebSocket connection
    ├── /ws/{bot_id}/audio/in       # Audio input stream
    └── /ws/{bot_id}/audio/out      # Audio output stream
```

### 7.2 Environment Variables Quick Reference

```bash
# Required
MEETING_BAAS_API_KEY    # Meeting platform integration
DEEPGRAM_API_KEY        # Speech-to-text
OPENAI_API_KEY          # LLM and optional TTS
BASE_URL                # Production server URL

# Optional but Recommended
CARTESIA_API_KEY        # Alternative TTS
REDIS_HOST              # Cache and pub/sub
API_SECRET_KEY          # API authentication
LOG_LEVEL               # DEBUG|INFO|WARNING|ERROR

# Audio Configuration
AUDIO_SAMPLE_RATE=16000 # Don't change without updating pipeline
AUDIO_CHANNELS=1        # Must be mono
AUDIO_CHUNK_SIZE=320    # 20ms at 16kHz
```

### 7.3 Common Commands

```bash
# Development
poetry install              # Install dependencies
poetry shell               # Activate environment
python api/main.py         # Run API server
python websocket/handler.py # Run WebSocket server

# Testing
pytest tests/              # Run all tests
pytest -v tests/test_api.py # Run specific test
python -m tests.load_test  # Run load tests

# Docker
docker-compose up -d       # Start all services
docker-compose logs -f api # View API logs
docker-compose down        # Stop all services

# Production
kubectl apply -f k8s/      # Deploy to Kubernetes
kubectl get pods           # Check pod status
kubectl logs -f pod-name   # View pod logs

# Debugging
wscat -c ws://localhost:8001/ws/test  # Test WebSocket
curl http://localhost:8000/health     # Check API health
redis-cli ping            # Check Redis connection
```

---

This comprehensive reference documentation provides all the technical details, code examples, and troubleshooting information needed for an LLM copilot to successfully implement the meeting agent system. Each section is self-contained and can be referenced independently during implementation.
