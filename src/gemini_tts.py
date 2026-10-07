"""Gemini 3.8 standard TTS client using Vertex AI."""

import logging
from typing import Any, Callable, Dict, Optional
from src.config import Settings
from src.audio_utils import build_wav_header, extract_pcm_from_wav, resample_pcm

logger = logging.getLogger(__name__)


class GeminiTTSClient:
    """Wrapper for Google GenAI Gemini Enterprise Text-to-Speech API."""

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
                    location=self.settings.location,
                )
        return self._client

    def synthesize_to_wav(
        self,
        text: str,
        voice: Optional[str] = None,
        style: Optional[str] = None,
        target_sample_rate: int = 8000,
        model: Optional[str] = None,
    ) -> bytes:
        """Synthesize text to telephony-compliant linear16 WAV audio.

        - Calls Gemini 3.8 TTS (Vertex AI)
        - Extracts raw PCM (typically 24 kHz)
        - Resamples to target_sample_rate (e.g. 8000 Hz)
        - Wraps in standard 44-byte RIFF header with exact Content-Length
        """
        voice_name = voice or self.settings.default_voice
        model_name = model or self.settings.standard_tts_model

        part: Dict[str, Any] = {"text": text}
        if style:
            part["speech_metadata"] = {"style": style}

        contents = [{"role": "user", "parts": [part]}]
        config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": voice_name}
                }
            },
        }

        logger.info(
            "Calling Gemini 3.8 TTS model=%s, voice=%s, text_len=%d",
            model_name,
            voice_name,
            len(text),
        )

        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        raw_audio_bytes = candidate.content.parts[0].inline_data.data

        # Extract PCM and source sample rate (defaults to 24000)
        pcm_data, src_rate = extract_pcm_from_wav(raw_audio_bytes, default_sample_rate=24000)

        # Resample to telephony target rate (default 8000 Hz)
        resampled_pcm = resample_pcm(pcm_data, orig_rate=src_rate, target_rate=target_sample_rate)

        # Build 44-byte WAV header with exact byte size
        wav_header = build_wav_header(
            sample_rate=target_sample_rate,
            channels=self.settings.default_channels,
            bits_per_sample=self.settings.default_bits_per_sample,
            total_pcm_len=len(resampled_pcm),
        )

        return wav_header + resampled_pcm
