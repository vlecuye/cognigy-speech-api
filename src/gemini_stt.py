"""Gemini 3.5 Transcribe Live client for Cognigy Voice Gateway."""

import asyncio
import logging
import time
from typing import AsyncGenerator, Callable, Optional
from src.config import Settings

logger = logging.getLogger(__name__)


class LiveSTTSession:
    """Represents an active streaming transcription session with Gemini Live."""

    def __init__(self, session, language: str, sample_rate_hz: int):
        self.session = session
        self.language = language
        self.sample_rate_hz = sample_rate_hz
        self.ready_event = asyncio.Event()
        self._closed = False
        self.session_id = f"stt-{int(time.time() * 1000) % 1000000:06d}"
        self._chunks_sent = 0
        self._bytes_sent = 0
        self._interims_received = 0
        self._finals_received = 0
        self._start_time = time.perf_counter()
        logger.info(
            "[GCP-STT] [%s] Initialized LiveSTTSession (lang=%s, rate=%d Hz, session_type=%s)",
            self.session_id,
            self.language,
            self.sample_rate_hz,
            type(session).__name__,
        )

    async def send_audio_chunk(self, pcm_chunk: bytes):
        """Send raw linear16 PCM audio chunk to Gemini Live immediately."""
        if self._closed or not pcm_chunk:
            return
        from google.genai import types

        try:
            t_chunk_start = time.perf_counter()
            await self.session.send_realtime_input(
                audio=types.Blob(
                    data=pcm_chunk,
                    mime_type=f"audio/pcm;rate={self.sample_rate_hz}",
                )
            )
            self._chunks_sent += 1
            self._bytes_sent += len(pcm_chunk)
            send_ms = (time.perf_counter() - t_chunk_start) * 1000

            if self._chunks_sent == 1:
                logger.info(
                    "[GCP-STT] [%s] >> Sent chunk #1 to Gemini Live (%d bytes, mime=audio/pcm;rate=%d, time=%.1f ms)",
                    self.session_id,
                    len(pcm_chunk),
                    self.sample_rate_hz,
                    send_ms,
                )
            elif self._chunks_sent % 50 == 0:
                elapsed_s = time.perf_counter() - self._start_time
                logger.info(
                    "[GCP-STT] [%s] >> Audio push active: %d chunks (%d bytes total, %.1fs elapsed, last_send=%.1f ms)",
                    self.session_id,
                    self._chunks_sent,
                    self._bytes_sent,
                    elapsed_s,
                    send_ms,
                )
        except Exception as e:
            logger.error(
                "[GCP-STT] [%s] FAILED sending chunk #%d to Gemini Live: %s (%s)",
                self.session_id,
                self._chunks_sent + 1,
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise

    async def send_stream_end(self):
        """Signal end of audio stream to Gemini Live."""
        if self._closed:
            return
        logger.info(
            "[GCP-STT] [%s] Signaling audio_stream_end=True to Gemini Live (sent %d chunks, %d bytes total)",
            self.session_id,
            self._chunks_sent,
            self._bytes_sent,
        )
        try:
            await self.session.send_realtime_input(audio_stream_end=True)
            logger.info("[GCP-STT] [%s] Successfully sent audio_stream_end signal", self.session_id)
        except Exception as e:
            logger.warning("[GCP-STT] [%s] Warning signaling audio stream end: %s", self.session_id, e)

    async def receive_events(self) -> AsyncGenerator[dict, None]:
        """Iterate over responses from Gemini Live and yield Cognigy transcription events."""
        logger.info("[GCP-STT] [%s] Starting receive_events() loop listening for responses from Gemini Live...", self.session_id)
        response_count = 0

        try:
            async for response in self.session.receive():
                response_count += 1
                logger.debug("[GCP-STT] [%s] Raw response #%d received from Gemini Live: %r", self.session_id, response_count, response)

                if getattr(response, "setup_complete", None):
                    logger.info("[GCP-STT] [%s] Received setup_complete=True from Gemini Live. Session marked READY.", self.session_id)
                    self.ready_event.set()

                server_content = getattr(response, "server_content", None)
                if not server_content:
                    if getattr(response, "error", None):
                        logger.error("[GCP-STT] [%s] Error payload in response from Gemini Live: %r", self.session_id, response.error)
                    continue

                self.ready_event.set()

                # 1. Check for interim transcription hypothesis
                interim = getattr(server_content, "interim_input_transcription", None)
                if interim and getattr(interim, "text", None):
                    self._interims_received += 1
                    logger.info(
                        "[GCP-STT] [%s] <<< INTERIM #%d from Gemini Live: %r (lang=%s)",
                        self.session_id,
                        self._interims_received,
                        interim.text,
                        self.language,
                    )
                    yield {
                        "type": "transcription",
                        "is_final": False,
                        "alternatives": [
                            {
                                "confidence": 0.8,
                                "transcript": interim.text,
                            }
                        ],
                        "channel": 1,
                        "language": self.language,
                    }

                # 2. Check for final transcription (turn boundary)
                final = getattr(server_content, "input_transcription", None)
                if final and getattr(final, "text", None):
                    self._finals_received += 1
                    logger.info(
                        "[GCP-STT] [%s] <<< FINAL #%d from Gemini Live: %r (lang=%s)",
                        self.session_id,
                        self._finals_received,
                        final.text,
                        self.language,
                    )
                    yield {
                        "type": "transcription",
                        "is_final": True,
                        "alternatives": [
                            {
                                "confidence": 0.98,
                                "transcript": final.text,
                            }
                        ],
                        "channel": 1,
                        "language": self.language,
                    }

                if getattr(server_content, "turn_complete", None):
                    logger.info("[GCP-STT] [%s] Received turn_complete=True from Gemini Live", self.session_id)

        except asyncio.CancelledError:
            logger.info("[GCP-STT] [%s] receive_events() cancelled", self.session_id)
            raise
        except Exception as e:
            logger.error(
                "[GCP-STT] [%s] Error in receive_events() loop: %s (%s)",
                self.session_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise
        finally:
            logger.info(
                "[GCP-STT] [%s] receive_events() terminated (total_responses=%d, interims=%d, finals=%d)",
                self.session_id,
                response_count,
                self._interims_received,
                self._finals_received,
            )

    def close(self):
        self._closed = True
        self.ready_event.set()
        logger.info(
            "[GCP-STT] [%s] LiveSTTSession closed (total_sent=%d chunks, %d bytes, interims=%d, finals=%d)",
            self.session_id,
            self._chunks_sent,
            self._bytes_sent,
            self._interims_received,
            self._finals_received,
        )


class GeminiSTTClient:
    """Client for connecting to Gemini 3.5 Transcribe Live in Vertex AI."""

    def __init__(self, settings: Settings, client_factory: Optional[Callable] = None):
        self.settings = settings
        self._client_factory = client_factory
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self._client_factory:
                logger.info("[GCP-STT] Initializing client using custom client_factory")
                self._client = self._client_factory()
            else:
                from google import genai
                client_kwargs = {
                    "enterprise": True,
                    "location": self.settings.transcribe_location,
                }
                if self.settings.project_id:
                    client_kwargs["project"] = self.settings.project_id
                logger.info(
                    "[GCP-STT] Initializing google.genai.Client with kwargs: %s",
                    {k: v for k, v in client_kwargs.items()},
                )
                self._client = genai.Client(**client_kwargs)
        return self._client

    async def connect_transcribe_session(
        self,
        language: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
        model: Optional[str] = None,
    ):
        """Create and connect a bidirectional Gemini Live transcription session."""
        from google.genai import types

        lang = language or self.settings.transcribe_language
        sample_rate = sample_rate_hz or self.settings.transcribe_sample_rate
        model_name = model or self.settings.transcribe_model

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=[lang] if lang else ["en-US"],
            ),
        )

        logger.info(
            "[GCP-STT] Connecting to Gemini Live API | model=%s | location=%s | project=%s | lang=%s | sample_rate=%d",
            model_name,
            self.settings.transcribe_location,
            self.settings.project_id,
            lang,
            sample_rate,
        )
        logger.info(
            "[GCP-STT] LiveConnectConfig payload: response_modalities=%s, language_codes=%s",
            [types.Modality.TEXT],
            [lang] if lang else ["en-US"],
        )

        t_start = time.perf_counter()
        try:
            session_ctx = self.client.aio.live.connect(model=model_name, config=config)
            session = await session_ctx.__aenter__()
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "[GCP-STT] Successfully connected to Gemini Live in %.1f ms! Session object: %r",
                elapsed_ms,
                session,
            )
            live_session = LiveSTTSession(session=session, language=lang, sample_rate_hz=sample_rate)
            return live_session, session_ctx
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.error(
                "[GCP-STT] FAILED connecting to Gemini Live after %.1f ms! Exception [%s]: %s",
                elapsed_ms,
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise
