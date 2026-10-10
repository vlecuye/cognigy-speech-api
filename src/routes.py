"""Cognigy Voice Gateway Custom Speech HTTP and WebSocket routes."""

import asyncio
import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import Settings, settings
from src.audio_utils import build_wav_header
from src.gemini_tts import GeminiTTSClient
from src.gemini_live import GeminiLiveClient
from src.gemini_stt import GeminiSTTClient

logger = logging.getLogger(__name__)

router = APIRouter()


class SynthesizeRequest(BaseModel):
    """Cognigy Voice Gateway Custom Speech Request Schema."""
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice identifier, e.g. Puck, Kore")
    language: Optional[str] = Field("en-US", description="BCP-47 language tag")
    type: Optional[str] = Field("text", description="Input type: 'text' or 'ssml'")
    encoding: Optional[str] = Field("linear16", description="Audio encoding (linear16)")
    sample_rate: Optional[int] = Field(8000, description="Sample rate in Hz (telephony standard is 8000)")
    style: Optional[str] = Field(None, description="Optional Gemini speech style instruction")
    model: Optional[str] = Field(None, description="Override Gemini model name")
    stream: Optional[bool] = Field(None, description="Explicit stream toggle")


def get_settings() -> Settings:
    return settings


def verify_auth(request: Request, app_settings: Settings = Depends(get_settings)):
    """Enforce Bearer token authorization if cognigy_api_key is configured."""
    expected_key = app_settings.cognigy_api_key
    if not expected_key:
        return

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer authorization header")

    token = auth_header[7:].strip()
    if token != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")


@router.get("/healthz")
async def healthz():
    """Health check endpoint for Cloud Run and load balancers."""
    return {"status": "ok", "service": "cognigy-gemini-speech"}


@router.post(
    "/synthesize/gemini",
    dependencies=[Depends(verify_auth)],
    summary="Standard Gemini 3.8 batch TTS for Cognigy Voice Gateway",
)
async def synthesize_gemini_standard(
    req: SynthesizeRequest,
    app_settings: Settings = Depends(get_settings),
):
    """Standard (buffered) custom TTS endpoint for Cognigy Voice Gateway.

    Returns full linear16 WAV audio at the requested sample rate (default 8000 Hz)
    with Content-Length header.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty")

    sample_rate = req.sample_rate or app_settings.default_sample_rate
    client = GeminiTTSClient(app_settings)

    try:
        wav_bytes = client.synthesize_to_wav(
            text=req.text,
            voice=req.voice,
            style=req.style,
            target_sample_rate=sample_rate,
            model=req.model,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Length": str(len(wav_bytes)),
                "X-Audio-Sample-Rate": str(sample_rate),
                "X-Audio-Encoding": "linear16",
            },
        )
    except Exception as e:
        logger.exception("Error in standard synthesis: %s", e)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@router.post(
    "/synthesize/gemini-live",
    dependencies=[Depends(verify_auth)],
    summary="Ultra-low latency streaming TTS for Cognigy Voice Gateway via Gemini Live API",
)
async def synthesize_gemini_live(
    req: SynthesizeRequest,
    app_settings: Settings = Depends(get_settings),
):
    """Real-time streaming custom TTS endpoint for Cognigy Voice Gateway.

    Streams chunked linear16 WAV audio at 8000 Hz (mono) starting with a 44-byte
    RIFF/WAV header (0xFFFFFFFF data size) followed immediately by linear16 PCM chunks.
    Playback begins on the phone call as soon as chunk 1 arrives.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty")

    sample_rate = req.sample_rate or app_settings.default_sample_rate
    live_client = GeminiLiveClient(app_settings)

    async def generate_chunked_wav():
        # 1. Immediately emit standard 44-byte streaming WAV header
        header = build_wav_header(
            sample_rate=sample_rate,
            channels=app_settings.default_channels,
            bits_per_sample=app_settings.default_bits_per_sample,
            total_pcm_len=None,  # 0xFFFFFFFF for chunked stream
        )
        yield header

        # 2. Progressively stream resampled PCM chunks from Gemini Live
        try:
            async for pcm_chunk in live_client.stream_speech(
                text=req.text,
                voice=req.voice,
                target_sample_rate=sample_rate,
                model=req.model,
            ):
                yield pcm_chunk
        except Exception as e:
            logger.exception("Error during live streaming synthesis: %s", e)
            # If headers are already sent, dropping connection ends the audio stream
            return

    return StreamingResponse(
        generate_chunked_wav(),
        media_type="audio/wav",
        headers={
            "Transfer-Encoding": "chunked",
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Encoding": "linear16",
        },
    )


@router.post(
    "/synthesize",
    dependencies=[Depends(verify_auth)],
    summary="Unified Cognigy Voice Gateway custom TTS endpoint",
)
async def synthesize_unified(
    req: SynthesizeRequest,
    mode: Optional[str] = None,
    app_settings: Settings = Depends(get_settings),
):
    """Unified endpoint that routes to either standard or live streaming synthesis.

    If mode='live', req.stream is True, or req.model contains 'live', dispatches to Gemini Live.
    Otherwise dispatches to standard Gemini 3.8 TTS.
    """
    use_live = (
        mode == "live"
        or req.stream is True
        or (req.model and "live" in req.model.lower())
    )

    if use_live:
        return await synthesize_gemini_live(req=req, app_settings=app_settings)
    return await synthesize_gemini_standard(req=req, app_settings=app_settings)


@router.websocket("/transcribe/gemini")
@router.websocket("/transcribe")
async def transcribe_websocket(
    websocket: WebSocket,
    app_settings: Settings = Depends(get_settings),
):
    """Cognigy Voice Gateway Custom Speech WebSocket STT Endpoint.

    Streams bidirectional real-time audio and transcription events:
    1. Authenticates via Authorization: Bearer <API_KEY> on the WS connection handshake.
    2. Receives initial JSON {"type": "start", "language": "...", "sampleRateHz": 8000, ...}.
    3. Concurrently receives binary Linear16 PCM chunks and streams to Gemini 3.5 Transcribe Live.
    4. Streams interim (is_final=False) and final (is_final=True) JSON transcription events to Cognigy.
    5. Clean teardown upon receiving {"type": "stop"} or client disconnect.
    """
    # 1. Enforce API key authentication if configured
    expected_key = app_settings.cognigy_api_key
    if expected_key:
        auth_header = websocket.headers.get("authorization", "")
        token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
        if token != expected_key:
            logger.warning("Rejected unauthorized WebSocket connection attempt")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await websocket.accept()
    logger.info("Cognigy Voice Gateway connected to STT WebSocket")

    stt_client = GeminiSTTClient(app_settings)
    live_session = None
    session_ctx = None
    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    try:
        # 2. Receive and parse initial "start" message
        first_msg = await websocket.receive()
        if "text" not in first_msg or not first_msg["text"]:
            logger.warning("Expected initial JSON 'start' message, got binary or empty")
            await websocket.send_json({"type": "error", "error": "Expected initial JSON start message"})
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        try:
            start_payload = json.loads(first_msg["text"])
        except Exception:
            await websocket.send_json({"type": "error", "error": "Invalid JSON in start message"})
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        if start_payload.get("type") != "start":
            await websocket.send_json({"type": "error", "error": f"Expected type 'start', got '{start_payload.get('type')}'"})
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        language = start_payload.get("language") or app_settings.transcribe_language
        sample_rate_hz = int(start_payload.get("sampleRateHz") or app_settings.transcribe_sample_rate)
        logger.info("Starting transcription session (lang=%s, rate=%d Hz)", language, sample_rate_hz)

        # 3. Connect to upstream Gemini 3.5 Transcribe Live
        live_session, session_ctx = await stt_client.connect_transcribe_session(
            language=language,
            sample_rate_hz=sample_rate_hz,
        )

        stop_event = asyncio.Event()

        # Task A: Read binary audio chunks and control frames from Cognigy
        async def cognigy_reader():
            try:
                while not stop_event.is_set():
                    msg = await websocket.receive()
                    msg_type = msg.get("type")

                    if msg_type == "websocket.disconnect":
                        logger.info("Cognigy WebSocket disconnected")
                        break

                    # Binary frame: Audio PCM chunk
                    if "bytes" in msg and msg["bytes"]:
                        await audio_queue.put(msg["bytes"])

                    # Text frame: Control message (e.g. stop)
                    elif "text" in msg and msg["text"]:
                        try:
                            ctrl = json.loads(msg["text"])
                            if ctrl.get("type") == "stop":
                                logger.info("Received 'stop' teardown signal from Cognigy Voice Gateway")
                                stop_event.set()
                                break
                        except Exception as e:
                            logger.warning("Error parsing Cognigy text frame: %s", e)
            except WebSocketDisconnect:
                logger.info("Cognigy Voice Gateway disconnected normally")
            except Exception as e:
                logger.error("Error in Cognigy reader: %s", e)
            finally:
                stop_event.set()

        # Task B: Push audio PCM chunks to Gemini Live
        async def gemini_audio_pusher():
            try:
                while not stop_event.is_set():
                    try:
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        await live_session.send_audio_chunk(chunk)
                    except asyncio.TimeoutError:
                        continue
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Error in Gemini audio pusher: %s", e)
            finally:
                await live_session.send_stream_end()

        # Task C: Stream transcription results back to Cognigy Voice Gateway
        async def gemini_result_streamer():
            try:
                async for event in live_session.receive_events():
                    if stop_event.is_set():
                        break
                    await websocket.send_json(event)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Error streaming transcription results to Cognigy: %s", e)

        # Run coordinated tasks
        reader_task = asyncio.create_task(cognigy_reader())
        pusher_task = asyncio.create_task(gemini_audio_pusher())
        streamer_task = asyncio.create_task(gemini_result_streamer())

        # Wait until reader ends (stop received or client disconnected)
        await reader_task
        stop_event.set()

        # Cancel background pusher & streamer
        pusher_task.cancel()
        streamer_task.cancel()
        await asyncio.gather(pusher_task, streamer_task, return_exceptions=True)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("Error in transcription WebSocket handler: %s", e)
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        if live_session:
            live_session.close()
        if session_ctx:
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Error cleaning up Gemini Live session: %s", e)
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Transcription session closed and cleaned up")

