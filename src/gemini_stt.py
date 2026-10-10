"""Gemini 3.5 Transcribe Live client for Cognigy Voice Gateway."""

import asyncio
import logging
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

    async def send_audio_chunk(self, pcm_chunk: bytes):
        """Send raw linear16 PCM audio chunk to Gemini Live once ready."""
        if self._closed or not pcm_chunk:
            return
        from google.genai import types

        await self.ready_event.wait()
        await self.session.send_realtime_input(
            audio=types.Blob(
                data=pcm_chunk,
                mime_type=f"audio/pcm;rate={self.sample_rate_hz}",
            )
        )

    async def send_stream_end(self):
        """Signal end of audio stream to Gemini Live."""
        if self._closed:
            return
        try:
            await self.session.send_realtime_input(audio_stream_end=True)
        except Exception as e:
            logger.debug("Error signaling audio stream end: %s", e)

    async def receive_events(self) -> AsyncGenerator[dict, None]:
        """Iterate over responses from Gemini Live and yield Cognigy transcription events."""
        async for response in self.session.receive():
            if getattr(response, "setup_complete", None):
                self.ready_event.set()

            server_content = getattr(response, "server_content", None)
            if not server_content:
                continue

            self.ready_event.set()

            # 1. Check for interim transcription hypothesis
            interim = getattr(server_content, "interim_input_transcription", None)
            if interim and getattr(interim, "text", None):
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

    def close(self):
        self._closed = True
        self.ready_event.set()


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
                self._client = self._client_factory()
            else:
                from google import genai
                self._client = genai.Client(
                    enterprise=True,
                    project=self.settings.project_id,
                    location=self.settings.transcribe_location,
                )
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
            "Connecting to Gemini Transcribe Live API (model=%s, location=%s, lang=%s, rate=%d)",
            model_name,
            self.settings.transcribe_location,
            lang,
            sample_rate,
        )

        session_ctx = self.client.aio.live.connect(model=model_name, config=config)
        session = await session_ctx.__aenter__()
        live_session = LiveSTTSession(session=session, language=lang, sample_rate_hz=sample_rate)

        return live_session, session_ctx
