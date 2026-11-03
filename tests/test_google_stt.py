"""Unit tests for Google Cloud Speech-to-Text STT engine and endpoints."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient

from src.app import app
from src.config import Settings
from src.routes import get_settings
from src.google_stt import GoogleSpeechSTTClient, GoogleSTTSession


class MockAlternative:
    def __init__(self, transcript, confidence=0.95):
        self.transcript = transcript
        self.confidence = confidence


class MockResult:
    def __init__(self, transcript, is_final=False, confidence=0.95):
        self.alternatives = [MockAlternative(transcript, confidence)]
        self.is_final = is_final


class MockStreamingResponse:
    def __init__(self, transcript, is_final=False, confidence=0.95):
        self.results = [MockResult(transcript, is_final, confidence)]


class MockStreamingCall:
    def __init__(self, responses):
        self.responses = responses

    def __aiter__(self):
        return self._generator()

    async def _generator(self):
        for resp in self.responses:
            yield resp


@pytest.fixture
def google_settings():
    return Settings(
        project_id="test-project",
        stt_provider="google",
        google_speech_model="telephony",
        google_speech_location="eu",
        transcribe_language="fr-FR",
        transcribe_sample_rate=8000,
        cognigy_api_key=None,
    )


@pytest.mark.asyncio
async def test_google_stt_session_lifecycle():
    """Verify GoogleSTTSession queues chunks and formats Cognigy transcription events."""
    mock_responses = [
        MockStreamingResponse("bonjour", is_final=False, confidence=0.8),
        MockStreamingResponse("bonjour comment allez-vous", is_final=True, confidence=0.99),
    ]
    streaming_call = MockStreamingCall(mock_responses)
    request_queue = asyncio.Queue()

    session = GoogleSTTSession(
        streaming_call=streaming_call,
        request_queue=request_queue,
        language="fr-FR",
        sample_rate_hz=8000,
    )

    # Send audio chunk
    await session.send_audio_chunk(b"\x00\x01" * 160)
    assert not request_queue.empty()
    req = await request_queue.get()
    assert len(req.audio_content) == 320

    # Signal stream end
    await session.send_stream_end()
    assert await request_queue.get() is None

    # Receive events
    events = []
    async for evt in session.receive_events():
        events.append(evt)

    assert len(events) == 2

    # Check interim
    assert events[0]["type"] == "transcription"
    assert events[0]["is_final"] is False
    assert events[0]["alternatives"][0]["transcript"] == "bonjour"
    assert events[0]["language"] == "fr-FR"

    # Check final
    assert events[1]["type"] == "transcription"
    assert events[1]["is_final"] is True
    assert events[1]["alternatives"][0]["transcript"] == "bonjour comment allez-vous"
    assert events[1]["alternatives"][0]["confidence"] == 0.99

    session.close()
    assert session._closed is True


@pytest.mark.asyncio
async def test_google_speech_stt_client_connection(google_settings):
    """Verify GoogleSpeechSTTClient initializes correctly and starts streaming."""
    mock_client = MagicMock()
    mock_streaming_call = MockStreamingCall([])
    mock_client.streaming_recognize = AsyncMock(return_value=mock_streaming_call)

    client = GoogleSpeechSTTClient(
        app_settings=google_settings,
        client_factory=lambda: mock_client,
    )

    session, session_ctx = await client.connect_transcribe_session(
        language="en-US",
        sample_rate_hz=8000,
    )

    assert session_ctx is None
    assert session.language == "en-US"
    assert session.sample_rate_hz == 8000
    mock_client.streaming_recognize.assert_called_once()


def test_google_stt_websocket_route(google_settings):
    """Test full WebSocket interaction with /transcribe/google route."""
    app.dependency_overrides[get_settings] = lambda: google_settings

    mock_responses = [
        MockStreamingResponse("hello", is_final=False),
        MockStreamingResponse("hello circle k", is_final=True, confidence=0.98),
    ]
    streaming_call = MockStreamingCall(mock_responses)
    mock_client = MagicMock()
    mock_client.streaming_recognize = AsyncMock(return_value=streaming_call)

    with patch("src.routes.GoogleSpeechSTTClient") as mock_client_cls:
        client_instance = MagicMock()

        async def fake_connect(language=None, sample_rate_hz=None):
            return GoogleSTTSession(
                streaming_call=streaming_call,
                request_queue=asyncio.Queue(),
                language=language or "en-US",
                sample_rate_hz=sample_rate_hz or 8000,
            ), None

        client_instance.connect_transcribe_session = AsyncMock(side_effect=fake_connect)
        mock_client_cls.return_value = client_instance

        try:
            client = TestClient(app)
            with client.websocket_connect("/transcribe/google") as ws:
                # 1. Send start frame
                start_msg = {
                    "type": "start",
                    "language": "en-US",
                    "format": "raw",
                    "encoding": "LINEAR16",
                    "sampleRateHz": 8000,
                }
                ws.send_text(json.dumps(start_msg))

                # 2. Send audio chunk
                ws.send_bytes(b"\x00" * 320)

                # 3. Receive interim
                msg1 = ws.receive_json()
                assert msg1["type"] == "transcription"
                assert msg1["is_final"] is False
                assert msg1["alternatives"][0]["transcript"] == "hello"

                # 4. Receive final
                msg2 = ws.receive_json()
                assert msg2["type"] == "transcription"
                assert msg2["is_final"] is True
                assert msg2["alternatives"][0]["transcript"] == "hello circle k"

                # 5. Send stop frame
                ws.send_text(json.dumps({"type": "stop"}))
        finally:
            app.dependency_overrides.clear()
