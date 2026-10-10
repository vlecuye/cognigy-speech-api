# Cognigy Gemini Speech Adapter

Unified custom speech (TTS and STT) adapter for **[Cognigy Voice Gateway (VG)](https://www.cognigy.com/platform/cognigy-voice-gateway)**, powered by Google Gemini speech synthesis and **Gemini 3.5 Transcribe Live**.

This microservice implements the [Cognigy Custom Speech HTTP and WebSocket Contracts](https://github.com/Cognigy/custom-speech-example) to deliver natural voice generation and real-time streaming speech recognition to telephony and contact center callers.

---

## Architecture & Endpoints

Cognigy Voice Gateway communicates with this adapter via HTTP POST for speech synthesis and WebSocket for streaming transcription:

| Capability | Protocol | Endpoint | Backend Engine | Latency / Format |
|---|---|---|---|---|
| **STT (Streaming)** | WebSocket | `wss://<HOST>/transcribe/gemini` (or `/transcribe`) | `gemini-3.5-transcribe-live` on Vertex AI (`eu`) | Real-time interim & final transcripts; turns auto-detected |
| **TTS (Streaming Live)** | HTTP POST | `POST /synthesize/gemini-live` | `gemini-live-2.5-flash-native-audio` (Live WebSocket API) | Chunked 8000 Hz Linear16 WAV (~300ms Time-To-First-Byte) |
| **TTS (Standard Batch)** | HTTP POST | `POST /synthesize/gemini` | `gemini-2.5-flash-tts` on Vertex AI | Buffered 8000 Hz Linear16 WAV (~1s) |
| **TTS (Unified)** | HTTP POST | `POST /synthesize` | Dynamic dispatch (`?mode=live` or payload `stream: true`) | Auto-selected based on request |

---

## The Cognigy Custom Speech Contract

### Request Specification
During a phone call, Cognigy Voice Gateway issues an HTTP POST request to your configured custom TTS URL:

```http
POST /synthesize/gemini-live HTTP/1.1
Authorization: Bearer <YOUR_API_KEY>
Content-Type: application/json
Accept: audio/wav

{
  "text": "Hello, thank you for calling. How can I help you today?",
  "voice": "Puck",
  "language": "en-US",
  "type": "text",
  "encoding": "linear16",
  "sample_rate": 8000
}
```

### Response Specification (Streaming Mode)
```http
HTTP/1.1 200 OK
Content-Type: audio/wav
Transfer-Encoding: chunked

<44-byte RIFF/WAV header (0xFFFFFFFF data size)><linear16 PCM @ 8000 Hz mono chunks...>
```
Cognigy Voice Gateway parses the 44-byte header and begins playing audio over the phone call on the very first 8kHz chunk received.

---

## The Cognigy Custom STT WebSocket Contract

During a phone call, Cognigy Voice Gateway connects to the WebSocket endpoint:
`wss://<YOUR_HOST>/transcribe/gemini`

1. **Handshake Auth**: Voice Gateway passes `Authorization: Bearer <YOUR_API_KEY>` in upgrade headers.
2. **Start Control Frame**: Voice Gateway sends JSON initialization:
   ```json
   {
     "type": "start",
     "language": "en-US",
     "format": "raw",
     "encoding": "LINEAR16",
     "sampleRateHz": 8000,
     "interimResults": true
   }
   ```
3. **Audio Streaming**: Voice Gateway continuously streams raw Linear16 PCM binary frames (8kHz mono).
4. **Transcription Delivery**: The adapter streams back JSON events:
   - Partial hypotheses: `{"type": "transcription", "is_final": false, "alternatives": [{"transcript": "hello"}]}`
   - Turn completion: `{"type": "transcription", "is_final": true, "alternatives": [{"confidence": 0.98, "transcript": "hello world"}]}`
5. **Teardown**: Voice Gateway sends `{"type": "stop"}` when the turn completes.

---

## Voices & Customization

Standard prebuilt Gemini voices:
- `Puck` — Male, warm, conversational
- `Kore` — Female, natural narration
- `Fenrir` — Male, deep, resonant
- `Aoede` — Female, soft, expressive
- `Charon` — Male, calm, deep
- `Zephyr` — Female, friendly, dynamic

You can also pass inline emotional tags in prompts when using Gemini 3.8 TTS:
`"I can certainly help you with that! [cheerful] Let me pull up your account."`

---

## Configuration

Environment variables (configure in `.env` or Cloud Run environment):

| Variable | Description | Default |
|---|---|---|
| `PROJECT_ID` | Google Cloud Project ID (Vertex AI) | `all-in-demo` |
| `LOCATION` | Vertex AI region for TTS | `europe-west1` |
| `TRANSCRIBE_LOCATION` | Vertex AI region for Gemini 3.5 Transcribe Live | `eu` (EU Multi-region) |
| `COGNIGY_API_KEY` | Secret Bearer token expected from Cognigy VG | `None` (open if unset) |
| `STANDARD_TTS_MODEL` | Standard Gemini TTS model | `gemini-2.5-flash-tts` |
| `LIVE_TTS_MODEL` | Real-time Live API TTS model | `gemini-live-2.5-flash-native-audio` |
| `TRANSCRIBE_MODEL` | Real-time Live API STT model | `gemini-3.5-transcribe-live` |
| `TRANSCRIBE_LANGUAGE` | Default transcription language | `en-US` |
| `DEFAULT_VOICE` | Default voice if omitted in request | `Puck` |
| `DEFAULT_SAMPLE_RATE` | Default telephony sample rate | `8000` |
| `PORT` | HTTP port | `8080` |

---

## Local Development & Testing

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Run Locally
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8080 --reload
```

### 4. Test Synthesis with cURL
```bash
# Standard batch synthesis (saves complete 8kHz WAV)
curl -X POST http://localhost:8080/synthesize/gemini \
  -H "Content-Type: application/json" \
  -d '{"text":"Welcome to Circle K.","voice":"Puck","sample_rate":8000}' \
  --output test_batch.wav

# Streaming synthesis (chunked transfer)
curl -X POST -N http://localhost:8080/synthesize/gemini-live \
  -H "Content-Type: application/json" \
  -d '{"text":"Welcome to Circle K live audio stream.","voice":"Kore","sample_rate":8000}' \
  --output test_live.wav
```

---

## Deploying to Google Cloud Run

Deploy this service independently to Cloud Run in your GCP project:

```bash
CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=false \
gcloud run deploy cognigy-gemini-speech \
  --source . \
  --project all-in-demo \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 1 \
  --set-env-vars "PROJECT_ID=all-in-demo,LOCATION=global,COGNIGY_API_KEY=YOUR_SECURE_TOKEN"
```

> [!TIP]
> Setting `--min-instances 1` ensures at least one container is always warm, eliminating cold-start latency when callers connect to your phone bot.

---

## Configuring in Cognigy Voice Gateway

1. In the **Cognigy Management Console**, navigate to **Voice Gateway** > **Credentials**.
2. Click **Add Credential** and choose **Custom Speech**.
3. Fill in the connection settings:
   - **TTS URL**:
     - For streaming: `https://<YOUR_CLOUD_RUN_URL>/synthesize/gemini-live`
     - For standard: `https://<YOUR_CLOUD_RUN_URL>/synthesize/gemini`
   - **Voice**: `Puck` (or `Kore`)
   - **Auth Token**: Enter your `COGNIGY_API_KEY` value (if configured).
   - **Enable text-to-speech streaming**: Checked `[x]` (when using the live streaming endpoint).
4. Save the credential.
5. In your Cognigy Conversation Flow:
   - Attach the credential to the endpoint.
   - Use standard **Say** nodes to prompt callers!
