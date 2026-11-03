"""Cognigy Voice Gateway Custom Speech HTTP and WebSocket routes."""

import asyncio
import json
import logging
import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import Settings, settings
from src.audio_utils import build_wav_header
from src.gemini_tts import GeminiTTSClient
from src.gemini_live import GeminiLiveClient
from src.gemini_stt import GeminiSTTClient
from src.google_stt import GoogleSpeechSTTClient

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
@router.post(
    "/synthetize/gemini",
    dependencies=[Depends(verify_auth)],
    include_in_schema=False,
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
        logger.warning("[TTS-STANDARD] Rejected request with empty text field")
        raise HTTPException(status_code=400, detail="Text field cannot be empty")

    sample_rate = req.sample_rate or app_settings.default_sample_rate
    logger.info(
        "[TTS-STANDARD] Inbound request | voice=%s | style=%s | sample_rate=%d Hz | model=%s | text_len=%d | preview=%r",
        req.voice,
        req.style,
        sample_rate,
        req.model,
        len(req.text),
        req.text[:60],
    )
    client = GeminiTTSClient(app_settings)

    try:
        t0 = time.perf_counter()
        wav_bytes = client.synthesize_to_wav(
            text=req.text,
            voice=req.voice,
            style=req.style,
            target_sample_rate=sample_rate,
            model=req.model,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[TTS-STANDARD] Synthesis completed in %.1f ms | WAV size: %d bytes (approx %.2f s audio)",
            elapsed_ms,
            len(wav_bytes),
            (len(wav_bytes) - 44) / (sample_rate * 2) if len(wav_bytes) > 44 else 0.0,
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
        logger.exception("[TTS-STANDARD] Error in standard synthesis: %s", e)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@router.post(
    "/synthesize/gemini-live",
    dependencies=[Depends(verify_auth)],
    summary="Ultra-low latency streaming TTS for Cognigy Voice Gateway via Gemini Live API",
)
@router.post(
    "/synthetize/gemini-live",
    dependencies=[Depends(verify_auth)],
    include_in_schema=False,
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
        logger.warning("[TTS-LIVE] Rejected request with empty text field")
        raise HTTPException(status_code=400, detail="Text field cannot be empty")

    sample_rate = req.sample_rate or app_settings.default_sample_rate
    logger.info(
        "[TTS-LIVE] Inbound streaming request | voice=%s | sample_rate=%d Hz | model=%s | text_len=%d | preview=%r",
        req.voice,
        sample_rate,
        req.model,
        len(req.text),
        req.text[:60],
    )
    live_client = GeminiLiveClient(app_settings)

    async def generate_chunked_wav():
        t0 = time.perf_counter()
        chunks_streamed = 0
        bytes_streamed = 0

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
                chunks_streamed += 1
                bytes_streamed += len(pcm_chunk)
                if chunks_streamed == 1:
                    first_chunk_latency = (time.perf_counter() - t0) * 1000
                    logger.info(
                        "[TTS-LIVE] >> First audio chunk yielded! Time-To-First-Audio: %.1f ms (%d bytes)",
                        first_chunk_latency,
                        len(pcm_chunk),
                    )
                yield pcm_chunk

            total_elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "[TTS-LIVE] Streaming synthesis complete | Total time: %.1f ms | Chunks: %d | Total bytes: %d",
                total_elapsed,
                chunks_streamed,
                bytes_streamed,
            )
        except Exception as e:
            logger.exception("[TTS-LIVE] Error during live streaming synthesis: %s", e)
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
@router.post(
    "/synthetize",
    dependencies=[Depends(verify_auth)],
    include_in_schema=False,
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


@router.websocket("/transcribe/google")
@router.websocket("/transcribe/gemini")
@router.websocket("/transcribe")
async def transcribe_websocket(
    websocket: WebSocket,
    app_settings: Settings = Depends(get_settings),
):
    """Cognigy Voice Gateway Custom Speech WebSocket STT Endpoint.

    Streams bidirectional real-time audio and transcription events:
    1. Authenticates via Authorization: Bearer <API_KEY> on WS handshake.
    2. Dynamically selects STT provider: Google Cloud Speech-to-Text or Gemini Live.
    3. Receives initial JSON {"type": "start", "language": "...", "sampleRateHz": 8000, ...}.
    4. Concurrently receives binary Linear16 PCM chunks and streams to chosen GCP STT engine.
    5. Streams interim (is_final=False) and final (is_final=True) JSON transcription events to Cognigy.
    6. Clean teardown upon receiving {"type": "stop"} or client disconnect.
    """
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else 0
    path = websocket.url.path
    query_params = dict(websocket.query_params)
    session_start_time = time.perf_counter()

    # Determine STT Provider: path override > query param override > settings default
    if "/gemini" in path:
        provider = "gemini"
    elif "/google" in path:
        provider = "google"
    elif "provider" in query_params:
        provider = query_params["provider"].lower()
    else:
        provider = app_settings.stt_provider.lower()

    # Log incoming connection details
    sanitized_headers = {}
    for k, v in websocket.headers.items():
        if k.lower() in ("authorization", "x-api-key"):
            sanitized_headers[k] = f"{v[:10]}...[LEN={len(v)}]" if len(v) > 10 else "***"
        else:
            sanitized_headers[k] = v

    logger.info(
        "[WSS-ENTRY] Incoming WebSocket connection attempt from %s:%s on %s [provider=%s] | Query: %s | Headers: %s",
        client_host,
        client_port,
        path,
        provider.upper(),
        query_params,
        sanitized_headers,
    )

    # 1. Enforce API key authentication if configured
    expected_key = app_settings.cognigy_api_key
    if expected_key:
        auth_header = websocket.headers.get("authorization", "")
        token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
        if not token:
            token = websocket.query_params.get("token") or websocket.query_params.get("api_key") or ""

        if token != expected_key:
            logger.warning(
                "[WSS-AUTH] Rejected unauthorized WebSocket connection from %s:%s on %s (token present: %s)",
                client_host,
                client_port,
                path,
                bool(token),
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        logger.info("[WSS-AUTH] Authenticated successfully from %s:%s", client_host, client_port)
    else:
        logger.info("[WSS-AUTH] Auth check skipped (COGNIGY_API_KEY is not configured)")

    await websocket.accept()
    logger.info("[WSS-ENTRY] WebSocket connection ACCEPTED for %s:%s on %s [provider=%s]", client_host, client_port, path, provider.upper())

    if provider == "gemini":
        logger.info("[WSS-LIFECYCLE] Initializing Gemini Live STT client (model=%s, location=%s)", app_settings.transcribe_model, app_settings.transcribe_location)
        stt_client = GeminiSTTClient(app_settings)
    else:
        logger.info("[WSS-LIFECYCLE] Initializing Google Cloud Speech STT client (model=%s, location=%s)", app_settings.google_speech_model, app_settings.google_speech_location)
        stt_client = GoogleSpeechSTTClient(app_settings)

    live_session = None
    session_ctx = None
    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    # Metrics counters for session summary
    chunks_received = 0
    bytes_received = 0
    text_msgs_received = 0
    chunks_pushed = 0
    bytes_pushed = 0
    interims_sent = 0
    finals_sent = 0
    exit_status = "UNKNOWN"

    try:
        # 2. Receive and parse initial "start" message
        logger.info("[WSS-ENTRY] Waiting for initial START frame from Cognigy Voice Gateway...")
        first_msg = await websocket.receive()
        logger.info(
            "[WSS-ENTRY] Received initial message frame: type=%s, keys=%s, text_len=%d, bytes_len=%d",
            first_msg.get("type"),
            list(first_msg.keys()),
            len(first_msg.get("text") or ""),
            len(first_msg.get("bytes") or b""),
        )

        if "text" not in first_msg or not first_msg["text"]:
            logger.warning("[WSS-ENTRY] Expected initial JSON 'start' message, got binary or empty: %r", first_msg)
            await websocket.send_json({"type": "error", "error": "Expected initial JSON start message"})
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            exit_status = "ERROR_EXPECTED_JSON_START"
            return

        raw_start_text = first_msg["text"]
        logger.info("[WSS-ENTRY] Raw START message text: %s", raw_start_text)

        try:
            start_payload = json.loads(raw_start_text)
            logger.info("[WSS-ENTRY] Parsed START payload JSON successfully: %s", start_payload)
        except Exception as e:
            logger.error("[WSS-ENTRY] Invalid JSON in start message: %r (error: %s)", raw_start_text, e)
            await websocket.send_json({"type": "error", "error": "Invalid JSON in start message"})
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            exit_status = "ERROR_INVALID_START_JSON"
            return

        if start_payload.get("type") != "start":
            logger.warning("[WSS-ENTRY] Expected start frame with type='start', got %r", start_payload.get("type"))
            await websocket.send_json({"type": "error", "error": f"Expected type 'start', got '{start_payload.get('type')}'"})
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            exit_status = "ERROR_UNEXPECTED_START_TYPE"
            return

        language = start_payload.get("language") or app_settings.transcribe_language
        sample_rate_hz = int(start_payload.get("sampleRateHz") or app_settings.transcribe_sample_rate)
        encoding = start_payload.get("encoding", "LINEAR16")
        format_type = start_payload.get("format", "raw")
        logger.info(
            "[WSS-ENTRY] Configured STT session parameters: provider=%s, language=%s, rate=%d Hz, encoding=%s, format=%s",
            provider.upper(),
            language,
            sample_rate_hz,
            encoding,
            format_type,
        )

        # 3. Connect to upstream STT engine (Gemini Live or Google Cloud Speech)
        logger.info("[WSS-LIFECYCLE] Initiating upstream STT connection via %s...", stt_client.__class__.__name__)
        t_gcp_start = time.perf_counter()
        live_session, session_ctx = await stt_client.connect_transcribe_session(
            language=language,
            sample_rate_hz=sample_rate_hz,
        )
        gcp_connect_ms = (time.perf_counter() - t_gcp_start) * 1000
        logger.info(
            "[WSS-LIFECYCLE] Upstream STT session [%s] established in %.1f ms! Session ID: %s",
            provider.upper(),
            gcp_connect_ms,
            live_session.session_id,
        )

        stop_event = asyncio.Event()

        # Task A: Read binary audio chunks and control frames from Cognigy
        async def cognigy_reader():
            nonlocal chunks_received, bytes_received, text_msgs_received
            logger.info("[WSS-IN] Cognigy reader task started. Listening for incoming audio chunks & control messages...")
            try:
                while not stop_event.is_set():
                    msg = await websocket.receive()
                    msg_type = msg.get("type")

                    if msg_type == "websocket.disconnect":
                        logger.info("[WSS-IN] Cognigy WebSocket disconnected normally (client closed connection)")
                        break

                    # Binary frame: Audio PCM chunk
                    if "bytes" in msg and msg["bytes"]:
                        chunk_bytes = msg["bytes"]
                        chunks_received += 1
                        bytes_received += len(chunk_bytes)
                        if chunks_received == 1:
                            logger.info(
                                "[WSS-IN] Received FIRST binary audio chunk from Cognigy: %d bytes",
                                len(chunk_bytes),
                            )
                        elif chunks_received % 50 == 0:
                            logger.info(
                                "[WSS-IN] Cognigy audio stream active: %d chunks received (%d bytes total) | Queue depth: %d",
                                chunks_received,
                                bytes_received,
                                audio_queue.qsize(),
                            )
                        await audio_queue.put(chunk_bytes)

                    # Text frame: Control message (e.g. stop)
                    elif "text" in msg and msg["text"]:
                        text_msgs_received += 1
                        raw_ctrl = msg["text"]
                        logger.info("[WSS-IN] Received text frame #%d from Cognigy: %s", text_msgs_received, raw_ctrl)
                        try:
                            ctrl = json.loads(raw_ctrl)
                            if ctrl.get("type") == "stop":
                                logger.info("[WSS-IN] Received explicit 'stop' teardown signal from Cognigy Voice Gateway")
                                stop_event.set()
                                break
                            else:
                                logger.info("[WSS-IN] Received unhandled control frame type: %s", ctrl.get("type"))
                        except Exception as e:
                            logger.warning("[WSS-IN] Error parsing Cognigy text frame as JSON: %s (raw=%r)", e, raw_ctrl)
            except WebSocketDisconnect as e:
                logger.info(
                    "[WSS-IN] WebSocketDisconnect encountered: code=%s, reason=%r",
                    getattr(e, "code", None),
                    getattr(e, "reason", None),
                )
            except Exception as e:
                logger.error("[WSS-IN] Unexpected error in Cognigy reader: %s (%s)", type(e).__name__, e, exc_info=True)
            finally:
                logger.info(
                    "[WSS-IN] Cognigy reader finished: total_chunks=%d (%d bytes), text_msgs=%d",
                    chunks_received,
                    bytes_received,
                    text_msgs_received,
                )
                stop_event.set()

        # Task B: Push audio PCM chunks to Gemini Live
        async def gemini_audio_pusher():
            nonlocal chunks_pushed, bytes_pushed
            logger.info("[WSS-PUSHER] Gemini audio pusher task started. Forwarding queue to Gemini Live...")
            try:
                while not stop_event.is_set():
                    try:
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        await live_session.send_audio_chunk(chunk)
                        chunks_pushed += 1
                        bytes_pushed += len(chunk)
                    except asyncio.TimeoutError:
                        continue
            except asyncio.CancelledError:
                logger.info("[WSS-PUSHER] Audio pusher task cancelled")
            except Exception as e:
                logger.error("[WSS-PUSHER] Error in Gemini audio pusher: %s (%s)", type(e).__name__, e, exc_info=True)
            finally:
                logger.info(
                    "[WSS-PUSHER] Signaling stream end to Gemini Live (forwarded %d chunks, %d bytes)",
                    chunks_pushed,
                    bytes_pushed,
                )
                await live_session.send_stream_end()

        # Task C: Stream transcription results back to Cognigy Voice Gateway
        async def gemini_result_streamer():
            nonlocal interims_sent, finals_sent
            logger.info("[WSS-STREAMER] Gemini result streamer task started. Forwarding transcripts to Cognigy...")
            try:
                async for event in live_session.receive_events():
                    if stop_event.is_set():
                        logger.info("[WSS-STREAMER] Stop event set. Breaking out of streamer loop.")
                        break

                    is_final = event.get("is_final", False)
                    alt_text = event.get("alternatives", [{}])[0].get("transcript", "")
                    confidence = event.get("alternatives", [{}])[0].get("confidence", 0.0)

                    if is_final:
                        finals_sent += 1
                        logger.info(
                            "[WSS-OUT] >>> Forwarding FINAL #%d to Cognigy (conf=%.2f): %r",
                            finals_sent,
                            confidence,
                            alt_text,
                        )
                    else:
                        interims_sent += 1
                        logger.info(
                            "[WSS-OUT] >>> Forwarding INTERIM #%d to Cognigy (conf=%.2f): %r",
                            interims_sent,
                            confidence,
                            alt_text,
                        )

                    await websocket.send_json(event)
            except asyncio.CancelledError:
                logger.info("[WSS-STREAMER] Result streamer task cancelled")
            except Exception as e:
                logger.error("[WSS-STREAMER] Error streaming transcription results to Cognigy: %s (%s)", type(e).__name__, e, exc_info=True)
            finally:
                logger.info(
                    "[WSS-STREAMER] Result streamer terminated (interims_sent=%d, finals_sent=%d)",
                    interims_sent,
                    finals_sent,
                )

        # Run coordinated tasks
        reader_task = asyncio.create_task(cognigy_reader())
        pusher_task = asyncio.create_task(gemini_audio_pusher())
        streamer_task = asyncio.create_task(gemini_result_streamer())

        # Wait until reader ends (stop received or client disconnected)
        await reader_task
        stop_event.set()
        exit_status = "NORMAL_COMPLETION"

        # Cancel background pusher & streamer
        pusher_task.cancel()
        streamer_task.cancel()
        await asyncio.gather(pusher_task, streamer_task, return_exceptions=True)

    except WebSocketDisconnect:
        logger.info("[WSS-LIFECYCLE] WebSocket disconnected normally")
        exit_status = "CLIENT_DISCONNECT"
    except Exception as e:
        exit_status = f"EXCEPTION_{type(e).__name__}"
        logger.exception("[WSS-LIFECYCLE] Exception in transcription WebSocket handler: %s", e)
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
                logger.debug("[WSS-LIFECYCLE] Error cleaning up Gemini Live session: %s", e)
        try:
            await websocket.close()
        except Exception:
            pass

        duration_s = time.perf_counter() - session_start_time
        logger.info(
            """
========================= STT SESSION SUMMARY =========================
Provider:        %s
Client:          %s:%s
Path:            %s
Duration:        %.2f s
Status:          %s
From Cognigy:    %d chunks (%d bytes), %d text frames
To STT Engine:   %d chunks (%d bytes)
Transcripts out: %d interim hypotheses, %d final utterances
=======================================================================
            """.strip(),
            provider.upper(),
            client_host,
            client_port,
            path,
            duration_s,
            exit_status,
            chunks_received,
            bytes_received,
            text_msgs_received,
            chunks_pushed,
            bytes_pushed,
            interims_sent,
            finals_sent,
        )


