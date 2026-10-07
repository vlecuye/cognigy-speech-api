# Cognigy Gemini Speech Adapter

Custom speech (Text-to-Speech) adapter for **[Cognigy Voice Gateway (VG)](https://www.cognigy.com/platform/cognigy-voice-gateway)**, powered by Google Gemini 3.8 and the Gemini Live API.

This microservice implements the [Cognigy Custom Speech HTTP Contract](https://github.com/Cognigy/custom-speech-example) to deliver natural, human-like voice synthesis to telephony and contact center callers.

---

## Architecture & Modes

Cognigy Voice Gateway supports both **non-streaming** and **streaming** TTS over HTTP POST:

| Mode | Endpoint | Backend Engine | Telephony Audio Output | Latency Profile |
|---|---|---|---|---|
| **Standard (Batch)** | `POST /synthesize/gemini` | `gemini-3.8-flash-tts` on Vertex AI | linear16 PCM @ 8000 Hz WAV (buffered) | Full synthesis buffer (~1–2s) |
| **Streaming (Live)** | `POST /synthesize/gemini-live` | `gemini-3.1-flash-live-preview` (Live WebSocket API) | linear16 PCM @ 8000 Hz WAV (chunked HTTP stream) | **Sub-350ms Time-To-First-Byte** |
| **Unified** | `POST /synthesize` | Dynamic dispatch (`?mode=live` or payload `stream: true`) | Auto-selected based on request | Flexible |

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
| `LOCATION` | Vertex AI region | `global` |
| `COGNIGY_API_KEY` | Secret Bearer token expected from Cognigy VG | `None` (open if unset) |
| `STANDARD_TTS_MODEL` | Standard Gemini TTS model | `gemini-3.8-flash-tts` |
| `LIVE_TTS_MODEL` | Real-time Live API model | `gemini-3.1-flash-live-preview` |
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
