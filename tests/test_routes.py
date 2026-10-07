"""Unit and integration tests for Cognigy Voice Gateway HTTP routes."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.config import Settings
from src.routes import get_settings


@pytest.fixture
def test_settings():
    return Settings(
        project_id="test-project",
        location="us-central1",
        cognigy_api_key=None,  # No auth by default in tests
        standard_tts_model="gemini-3.8-flash-tts",
        live_tts_model="gemini-3.1-flash-live-preview",
        default_voice="Puck",
        default_sample_rate=8000,
    )


@pytest.mark.asyncio
async def test_healthz():
    """Verify /healthz returns status ok."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "cognigy-gemini-speech"


@pytest.mark.asyncio
async def test_synthesize_gemini_standard_success(test_settings):
    """Verify POST /synthesize/gemini returns valid 8kHz WAV audio."""
    app.dependency_overrides[get_settings] = lambda: test_settings

    mock_wav = b"RIFF" + b"\x00" * 40 + b"\x11\x22" * 100

    with patch("src.routes.GeminiTTSClient") as MockClient:
        instance = MockClient.return_value
        instance.synthesize_to_wav.return_value = mock_wav

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/synthesize/gemini",
                json={
                    "text": "Hello Cognigy",
                    "voice": "Puck",
                    "sample_rate": 8000,
                },
            )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["content-length"] == str(len(mock_wav))
    assert response.headers["x-audio-sample-rate"] == "8000"
    assert response.content == mock_wav


@pytest.mark.asyncio
async def test_synthesize_gemini_empty_text():
    """Verify empty text produces 400 Bad Request."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/synthesize/gemini", json={"text": "   "})
    assert response.status_code == 400
    assert "Text field cannot be empty" in response.json()["detail"]


@pytest.mark.asyncio
async def test_synthesize_gemini_live_streaming(test_settings):
    """Verify POST /synthesize/gemini-live returns chunked WAV stream."""
    app.dependency_overrides[get_settings] = lambda: test_settings

    chunk1 = b"\x01\x00" * 50
    chunk2 = b"\x02\x00" * 50

    async def mock_stream_speech(text, voice, target_sample_rate, model):
        yield chunk1
        yield chunk2

    with patch("src.routes.GeminiLiveClient") as MockLiveClient:
        instance = MockLiveClient.return_value
        instance.stream_speech = mock_stream_speech

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/synthesize/gemini-live",
                json={
                    "text": "Streaming test",
                    "voice": "Kore",
                    "sample_rate": 8000,
                },
            )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    # First 44 bytes should be RIFF header
    assert response.content[:4] == b"RIFF"
    assert response.content[8:12] == b"WAVE"
    # Rest of content should contain the streamed chunks
    assert chunk1 in response.content
    assert chunk2 in response.content


@pytest.mark.asyncio
async def test_synthesize_unified_routing(test_settings):
    """Verify POST /synthesize routes dynamically based on payload or query."""
    app.dependency_overrides[get_settings] = lambda: test_settings

    mock_wav = b"RIFF" + b"\x00" * 40 + b"\x99" * 20

    with patch("src.routes.synthesize_gemini_standard") as mock_standard, \
         patch("src.routes.synthesize_gemini_live") as mock_live:

        mock_standard.return_value = "standard_called"
        mock_live.return_value = "live_called"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # 1. Standard by default
            await ac.post("/synthesize", json={"text": "default text"})
            assert mock_standard.called
            assert not mock_live.called

            mock_standard.reset_mock()
            mock_live.reset_mock()

            # 2. Live via query param ?mode=live
            await ac.post("/synthesize?mode=live", json={"text": "live text"})
            assert mock_live.called
            assert not mock_standard.called

            mock_standard.reset_mock()
            mock_live.reset_mock()

            # 3. Live via stream=true field
            await ac.post("/synthesize", json={"text": "streaming text", "stream": True})
            assert mock_live.called
            assert not mock_standard.called

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_bearer_auth_enforcement():
    """Verify Bearer token authentication when cognigy_api_key is set."""
    auth_settings = Settings(cognigy_api_key="super-secret-key")
    app.dependency_overrides[get_settings] = lambda: auth_settings

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Missing header -> 401
        res1 = await ac.post("/synthesize/gemini", json={"text": "test"})
        assert res1.status_code == 401

        # 2. Invalid token -> 403
        res2 = await ac.post(
            "/synthesize/gemini",
            json={"text": "test"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert res2.status_code == 403

        # 3. Valid token with mock synthesis -> 200
        with patch("src.routes.GeminiTTSClient") as MockClient:
            instance = MockClient.return_value
            instance.synthesize_to_wav.return_value = b"RIFF" + b"\x00" * 40
            res3 = await ac.post(
                "/synthesize/gemini",
                json={"text": "test"},
                headers={"Authorization": "Bearer super-secret-key"},
            )
            assert res3.status_code == 200

    app.dependency_overrides.clear()
