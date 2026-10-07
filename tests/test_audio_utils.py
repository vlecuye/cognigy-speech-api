"""Unit tests for audio processing and WAV header utilities."""

import struct
import numpy as np
from src.audio_utils import build_wav_header, extract_pcm_from_wav, resample_pcm


def test_build_wav_header_static():
    """Verify standard WAV header generation with known length."""
    header = build_wav_header(sample_rate=8000, channels=1, bits_per_sample=16, total_pcm_len=1600)
    assert len(header) == 44
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    assert header[36:40] == b"data"

    riff_magic, riff_size, wave_magic, fmt_magic, fmt_size, format_code, channels, rate, byte_rate, align, bits, data_magic, data_size = struct.unpack(
        "<4sI4s4sIHHIIHH4sI", header
    )
    assert riff_magic == b"RIFF"
    assert wave_magic == b"WAVE"
    assert fmt_magic == b"fmt "
    assert fmt_size == 16
    assert data_magic == b"data"
    assert riff_size == 36 + 1600
    assert format_code == 1  # PCM
    assert channels == 1
    assert rate == 8000
    assert bits == 16
    assert byte_rate == 8000 * 2
    assert align == 2
    assert data_size == 1600


def test_build_wav_header_streaming():
    """Verify streaming WAV header with 0xFFFFFFFF placeholder data size."""
    header = build_wav_header(sample_rate=8000, channels=1, bits_per_sample=16, total_pcm_len=None)
    assert len(header) == 44
    riff_size = struct.unpack("<I", header[4:8])[0]
    data_size = struct.unpack("<I", header[40:44])[0]
    assert riff_size == 0xFFFFFFFF
    assert data_size == 0xFFFFFFFF


def test_resample_pcm_24k_to_8k():
    """Verify 3:1 downsampling ratio from 24kHz to 8kHz."""
    # 24,000 samples of 440Hz sine wave = 1 second at 24kHz = 48,000 bytes
    sample_count = 24000
    t = np.linspace(0, 1, sample_count, endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    pcm_24k = sine.tobytes()

    pcm_8k = resample_pcm(pcm_24k, orig_rate=24000, target_rate=8000)

    # 1 second at 8kHz should be 8,000 samples = 16,000 bytes
    assert len(pcm_8k) == 8000 * 2
    samples_8k = np.frombuffer(pcm_8k, dtype=np.int16)
    assert len(samples_8k) == 8000


def test_resample_pcm_identity():
    """Verify identical sample rates return unchanged bytes."""
    data = b"\x01\x00\x02\x00\x03\x00"
    assert resample_pcm(data, orig_rate=8000, target_rate=8000) == data


def test_resample_pcm_empty():
    """Verify empty input returns empty output."""
    assert resample_pcm(b"", orig_rate=24000, target_rate=8000) == b""


def test_extract_pcm_from_wav():
    """Verify extracting raw PCM from a generated WAV file."""
    pcm_payload = b"\x10\x20" * 400
    header = build_wav_header(sample_rate=8000, channels=1, bits_per_sample=16, total_pcm_len=len(pcm_payload))
    wav_bytes = header + pcm_payload

    extracted_pcm, rate = extract_pcm_from_wav(wav_bytes)
    assert rate == 8000
    assert extracted_pcm == pcm_payload


def test_extract_pcm_from_raw():
    """Verify raw PCM bytes without header are returned as-is."""
    raw_pcm = b"\x05\x06\x07\x08"
    extracted, rate = extract_pcm_from_wav(raw_pcm, default_sample_rate=24000)
    assert extracted == raw_pcm
    assert rate == 24000
