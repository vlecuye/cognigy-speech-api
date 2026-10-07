"""Cognigy Voice Gateway Custom Speech HTTP routes."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import Settings, settings
from src.audio_utils import build_wav_header
from src.gemini_tts import GeminiTTSClient
from src.gemini_live import GeminiLiveClient

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
