"""Unit tests for Cognigy Voice Gateway WebSocket STT endpoint."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.app import app
from src.config import Settings
from src.routes import get_settings


class MockServerContent:
    def __init__(self, interim_text=None, final_text=None):
        if interim_text:
            self.interim_input_transcription = type("Interim", (), {"text": interim_text})()
            self.input_transcription = None
        elif final_text:
            self.interim_input_transcription = None
            self.input_transcription = type("Final", (), {"text": final_text})()
        else:
            self.interim_input_transcription = None
            self.input_transcription = None


class MockLiveSession:
    """Mock Gemini Live session simulating streaming STT."""

    def __init__(self, incoming_events):
        self.incoming_events = incoming_events
        self.received_audio = []
        self.stream_ended = False

    async def send_realtime_input(self, audio=None, audio_stream_end=False, **kwargs):
        if audio:
            self.received_audio.append(audio.data)
        if audio_stream_end:
            self.stream_ended = True

    async def receive(self):
        for evt in self.incoming_events:
            yield evt


class MockAsyncContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def test_settings():
    return Settings(
        project_id="test-project",
        location="europe-west1",
        transcribe_location="eu",
        transcribe_model="gemini-3.5-transcribe-live",
        cognigy_api_key=None,
    )


def test_stt_auth_enforcement():
    """Test that missing or invalid API key closes connection when auth is enabled."""
    secure_settings = Settings(cognigy_api_key="secret-123")
    app.dependency_overrides[get_settings] = lambda: secure_settings

    try:
        client = TestClient(app)

        # 1. No Authorization header -> Rejected
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/transcribe/gemini"):
                pass
        assert exc.value.code == 1008

        # 2. Invalid Bearer token -> Rejected
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                "/transcribe/gemini",
                headers={"Authorization": "Bearer wrong-token"},
            ):
                pass
        assert exc.value.code == 1008

        # 3. Valid Bearer token -> Accepted
        # (Will fail on start message if not provided, confirming connection passed auth)
        with client.websocket_connect(
            "/transcribe/gemini",
            headers={"Authorization": "Bearer secret-123"},
        ) as ws:
            ws.send_text("not-json")
            err = ws.receive_json()
            assert err.get("type") == "error"
    finally:
        app.dependency_overrides.clear()


def test_stt_invalid_start_messages(test_settings):
    """Test that malformed start messages are reported with error frames."""
    app.dependency_overrides[get_settings] = lambda: test_settings

    try:
        client = TestClient(app)

        # Invalid start type
        with client.websocket_connect("/transcribe/gemini") as ws:
            ws.send_text(json.dumps({"type": "invalid_type"}))
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Expected type 'start'" in resp["error"]
    finally:
        app.dependency_overrides.clear()


def test_stt_transcription_lifecycle(test_settings):
    """Test full Cognigy STT lifecycle: start -> audio chunks -> interim & final transcripts -> stop."""
    app.dependency_overrides[get_settings] = lambda: test_settings

    mock_responses = [
        type("Resp", (), {"setup_complete": True, "server_content": None})(),
        type(
            "Resp",
            (),
            {"setup_complete": False, "server_content": MockServerContent(interim_text="I want")},
        )(),
        type(
            "Resp",
            (),
            {
                "setup_complete": False,
                "server_content": MockServerContent(final_text="I want a coffee"),
            },
        )(),
    ]

    mock_live_session = MockLiveSession(incoming_events=mock_responses)
    mock_ctx = MockAsyncContext(mock_live_session)

    try:
        from src.gemini_stt import LiveSTTSession
        live_session = LiveSTTSession(session=mock_live_session, language="en-US", sample_rate_hz=8000)

        with patch("src.routes.GeminiSTTClient") as MockSTTClient:
            instance = MockSTTClient.return_value
            instance.connect_transcribe_session = AsyncMock(return_value=(live_session, mock_ctx))

            client = TestClient(app)

            with client.websocket_connect("/transcribe/gemini") as ws:
                # 1. Send Cognigy start control frame
                ws.send_text(
                    json.dumps(
                        {
                            "type": "start",
                            "language": "en-US",
                            "format": "raw",
                            "encoding": "LINEAR16",
                            "sampleRateHz": 8000,
                            "interimResults": True,
                        }
                    )
                )

                # 2. Send simulated raw 8kHz linear16 PCM audio chunk (e.g. 160 bytes)
                pcm_chunk = b"\x00\x00" * 80
                ws.send_bytes(pcm_chunk)

                # 3. Receive interim result
                msg1 = ws.receive_json()
                assert msg1["type"] == "transcription"
                assert msg1["is_final"] is False
                assert msg1["alternatives"][0]["transcript"] == "I want"
                assert msg1["language"] == "en-US"

                # 4. Receive final result (ends the Cognigy conversational turn)
                msg2 = ws.receive_json()
                assert msg2["type"] == "transcription"
                assert msg2["is_final"] is True
                assert msg2["alternatives"][0]["transcript"] == "I want a coffee"
                assert msg2["alternatives"][0]["confidence"] > 0.9

                # 5. Send stop teardown message from Cognigy
                ws.send_text(json.dumps({"type": "stop"}))
    finally:
        app.dependency_overrides.clear()


def test_stt_unified_endpoint(test_settings):
    """Test that /transcribe is an alias for /transcribe/gemini."""
    app.dependency_overrides[get_settings] = lambda: test_settings

    try:
        client = TestClient(app)
        with client.websocket_connect("/transcribe") as ws:
            ws.send_text(json.dumps({"type": "invalid"}))
            resp = ws.receive_json()
            assert resp["type"] == "error"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_gemini_stt_client_connection(test_settings):
    """Verify GeminiSTTClient initializes correctly with EU settings and connects session."""
    from src.gemini_stt import GeminiSTTClient

    mock_session = MagicMock()
    mock_ctx = MockAsyncContext(mock_session)

    mock_client = MagicMock()
    mock_client.aio.live.connect.return_value = mock_ctx

    stt_client = GeminiSTTClient(test_settings, client_factory=lambda: mock_client)
    live_session, ctx = await stt_client.connect_transcribe_session(
        language="de-DE",
        sample_rate_hz=16000,
    )

    assert live_session.language == "de-DE"
    assert live_session.sample_rate_hz == 16000
    mock_client.aio.live.connect.assert_called_once()
    call_kwargs = mock_client.aio.live.connect.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.5-transcribe-live"

