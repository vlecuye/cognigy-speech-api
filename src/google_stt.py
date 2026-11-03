"""Google Cloud Speech-to-Text (Telephony & Chirp) Client for Cognigy Voice Gateway."""

import asyncio
import logging
import time
import uuid
from typing import AsyncGenerator, Callable, Dict, Optional, Tuple

from google.cloud import speech_v1 as speech
from google.api_core.client_options import ClientOptions

from src.config import Settings, settings

logger = logging.getLogger(__name__)


class GoogleSTTSession:
    """Manages an active gRPC bidirectional streaming recognition session with Google Cloud Speech."""

    def __init__(
        self,
        streaming_call,
        request_queue: asyncio.Queue,
        language: str,
        sample_rate_hz: int,
        session_id: Optional[str] = None,
    ):
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.streaming_call = streaming_call
        self.request_queue = request_queue
        self.language = language
        self.sample_rate_hz = sample_rate_hz
        self._closed = False
        self._chunks_sent = 0
        self._bytes_sent = 0
        self._interims_received = 0
        self._finals_received = 0
        logger.info(
            "[GCP-SPEECH] [%s] Session initialized (lang=%s, rate=%d Hz)",
            self.session_id,
            self.language,
            self.sample_rate_hz,
        )

    async def send_audio_chunk(self, pcm_chunk: bytes) -> None:
        """Queue a raw Linear16 PCM audio chunk to be pushed via gRPC to Google Cloud Speech."""
        if self._closed or not pcm_chunk:
            return

        self._chunks_sent += 1
        self._bytes_sent += len(pcm_chunk)

        req = speech.StreamingRecognizeRequest(audio_content=pcm_chunk)
        await self.request_queue.put(req)

        if self._chunks_sent == 1 or self._chunks_sent % 50 == 0:
            logger.info(
                "[GCP-SPEECH] [%s] Audio streaming active: %d chunks sent (%d bytes total)",
                self.session_id,
                self._chunks_sent,
                self._bytes_sent,
            )

    async def send_stream_end(self) -> None:
        """Signal the end of the client audio stream to Google Cloud Speech."""
        if self._closed:
            return
        logger.info(
            "[GCP-SPEECH] [%s] End of audio stream signaled (sent %d chunks, %d bytes)",
            self.session_id,
            self._chunks_sent,
            self._bytes_sent,
        )
        # Put None sentinel to terminate the request generator
        await self.request_queue.put(None)

    async def receive_events(self) -> AsyncGenerator[Dict, None]:
        """Stream transcription results from Google Cloud Speech mapped to Cognigy format."""
        logger.info("[GCP-SPEECH] [%s] Starting receive_events() loop...", self.session_id)
        response_count = 0

        try:
            async for response in self.streaming_call:
                response_count += 1
                if not response.results:
                    continue

                for result in response.results:
                    is_final = result.is_final
                    if not result.alternatives:
                        continue

                    alternatives = []
                    for alt in result.alternatives:
                        alternatives.append(
                            {
                                "confidence": round(float(alt.confidence), 2) if alt.confidence else (0.98 if is_final else 0.8),
                                "transcript": alt.transcript,
                            }
                        )

                    primary_transcript = alternatives[0]["transcript"] if alternatives else ""

                    if is_final:
                        self._finals_received += 1
                        logger.info(
                            "[GCP-SPEECH] [%s] <<< FINAL #%d from Google STT: %r (lang=%s)",
                            self.session_id,
                            self._finals_received,
                            primary_transcript,
                            self.language,
                        )
                    else:
                        self._interims_received += 1
                        logger.info(
                            "[GCP-SPEECH] [%s] <<< INTERIM #%d from Google STT: %r (lang=%s)",
                            self.session_id,
                            self._interims_received,
                            primary_transcript,
                            self.language,
                        )

                    yield {
                        "type": "transcription",
                        "is_final": is_final,
                        "alternatives": alternatives,
                        "channel": 1,
                        "language": self.language,
                    }

        except asyncio.CancelledError:
            logger.info("[GCP-SPEECH] [%s] receive_events() cancelled", self.session_id)
            raise
        except Exception as e:
            logger.error(
                "[GCP-SPEECH] [%s] Error in receive_events() loop: %s (%s)",
                self.session_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise
        finally:
            logger.info(
                "[GCP-SPEECH] [%s] receive_events() terminated (responses=%d, interims=%d, finals=%d)",
                self.session_id,
                response_count,
                self._interims_received,
                self._finals_received,
            )

    def close(self):
        """Close the session and cleanup resources."""
        self._closed = True
        logger.info(
            "[GCP-SPEECH] [%s] GoogleSTTSession closed (total_sent=%d chunks, %d bytes, interims=%d, finals=%d)",
            self.session_id,
            self._chunks_sent,
            self._bytes_sent,
            self._interims_received,
            self._finals_received,
        )


class GoogleSpeechSTTClient:
    """Factory and client for Google Cloud Speech-to-Text streaming recognition."""

    def __init__(
        self,
        app_settings: Settings = settings,
        client_factory: Optional[Callable] = None,
    ):
        self.settings = app_settings
        self._client_factory = client_factory
        self._client: Optional[speech.SpeechAsyncClient] = None

    @property
    def client(self) -> speech.SpeechAsyncClient:
        if self._client is None:
            if self._client_factory:
                logger.info("[GCP-SPEECH] Initializing client using custom client_factory")
                self._client = self._client_factory()
            else:
                client_options = None
                if self.settings.google_speech_location and self.settings.google_speech_location.lower() == "eu":
                    client_options = ClientOptions(api_endpoint="eu-speech.googleapis.com")
                    logger.info("[GCP-SPEECH] Using European regional endpoint: eu-speech.googleapis.com")

                logger.info(
                    "[GCP-SPEECH] Initializing Google SpeechAsyncClient (location=%s, model=%s)",
                    self.settings.google_speech_location,
                    self.settings.google_speech_model,
                )
                self._client = speech.SpeechAsyncClient(client_options=client_options)
        return self._client

    async def connect_transcribe_session(
        self,
        language: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
    ) -> Tuple[GoogleSTTSession, None]:
        """Open a streaming recognize session with Google Cloud Speech."""
        lang = language or self.settings.transcribe_language
        sample_rate = sample_rate_hz or self.settings.transcribe_sample_rate
        session_id = uuid.uuid4().hex[:8]

        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=lang,
            model=self.settings.google_speech_model,
            enable_automatic_punctuation=True,
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,
            single_utterance=False,
        )

        logger.info(
            "[GCP-SPEECH] [%s] Opening streaming_recognize | model=%s | lang=%s | rate=%d Hz",
            session_id,
            self.settings.google_speech_model,
            lang,
            sample_rate,
        )

        request_queue: asyncio.Queue = asyncio.Queue()

        # The first request in the stream MUST contain the streaming_config
        initial_request = speech.StreamingRecognizeRequest(streaming_config=streaming_config)

        async def request_generator():
            yield initial_request
            while True:
                req = await request_queue.get()
                if req is None:
                    # Stream termination sentinel
                    break
                yield req

        t_start = time.perf_counter()
        try:
            streaming_call = await self.client.streaming_recognize(requests=request_generator())
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "[GCP-SPEECH] [%s] Connected to Google Cloud Speech in %.1f ms!",
                session_id,
                elapsed_ms,
            )
            live_session = GoogleSTTSession(
                streaming_call=streaming_call,
                request_queue=request_queue,
                language=lang,
                sample_rate_hz=sample_rate,
                session_id=session_id,
            )
            return live_session, None
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.error(
                "[GCP-SPEECH] [%s] Failed to connect to Google Speech in %.1f ms: %s (%s)",
                session_id,
                elapsed_ms,
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise
