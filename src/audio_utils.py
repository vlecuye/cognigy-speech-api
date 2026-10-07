"""Audio processing utilities for telephony streaming and resampling."""

import io
import math
import struct
import wave
from typing import Optional, Tuple
import numpy as np
from scipy import signal


def build_wav_header(
    sample_rate: int = 8000,
    channels: int = 1,
    bits_per_sample: int = 16,
    total_pcm_len: Optional[int] = None,
) -> bytes:
    """Generate a standard 44-byte RIFF/WAV header.

    If total_pcm_len is None, data size and RIFF size are set to 0xFFFFFFFF
    for streaming chunked transfer as required by Cognigy Voice Gateway / FreeSWITCH.
    """
    bytes_per_sample = bits_per_sample // 8
    block_align = channels * bytes_per_sample
    byte_rate = sample_rate * block_align

    if total_pcm_len is not None:
        data_size = total_pcm_len
        riff_size = 36 + total_pcm_len
    else:
        # Standard placeholder for streaming audio with unknown length
        data_size = 0xFFFFFFFF
        riff_size = 0xFFFFFFFF

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM subchunk size
        1,   # AudioFormat (1 = PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def resample_pcm(
    pcm_bytes: bytes,
    orig_rate: int = 24000,
    target_rate: int = 8000,
) -> bytes:
    """Resample raw 16-bit signed little-endian PCM audio.

    Uses polyphase filtering with anti-aliasing for fast and clean conversion.
    """
    if not pcm_bytes:
        return b""

    if orig_rate == target_rate:
        return pcm_bytes

    # Convert bytes to int16 numpy array
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(audio_data) == 0:
        return b""

    g = math.gcd(orig_rate, target_rate)
    up = target_rate // g
    down = orig_rate // g

    # Resample using polyphase filter
    resampled = signal.resample_poly(audio_data, up=up, down=down)
    # Clip to prevent int16 overflow/wrap-around
    clipped = np.clip(resampled, -32768, 32767).astype(np.int16)

    return clipped.tobytes()


def extract_pcm_from_wav(wav_bytes: bytes, default_sample_rate: int = 24000) -> Tuple[bytes, int]:
    """Extract raw PCM bytes and sample rate from WAV bytes.

    If wav_bytes is already raw PCM (no RIFF header), returns (wav_bytes, default_sample_rate).
    """
    if not wav_bytes:
        return b"", default_sample_rate

    if wav_bytes[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sample_rate = wf.getframerate()
                pcm_data = wf.readframes(wf.getnframes())
                return pcm_data, sample_rate
        except Exception:
            # Fallback if standard wave parser encounters non-standard header: strip first 44 bytes
            if len(wav_bytes) > 44:
                return wav_bytes[44:], default_sample_rate
            return wav_bytes, default_sample_rate

    return wav_bytes, default_sample_rate
