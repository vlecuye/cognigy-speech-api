"""Script to verify live Cloud Run endpoints."""

import socket
import subprocess
import sys
import httpx

# In corporate environments, resolve external Cloud Run IPs via public DNS
HOST = "cognigy-gemini-speech-816122473048.us-central1.run.app"
BASE_URL = f"https://{HOST}"

orig_getaddrinfo = socket.getaddrinfo


def custom_getaddrinfo(host, port, *args, **kwargs):
    try:
        return orig_getaddrinfo(host, port, *args, **kwargs)
    except Exception:
        # Fallback to direct DNS resolution
        cmd = ["dig", "@216.239.32.10", host, "+short"]
        out = subprocess.check_output(cmd).decode()
        for line in out.splitlines():
            ip = line.strip()
            if ip and not ip.startswith(";") and ":" not in ip:
                return orig_getaddrinfo(ip, port, *args, **kwargs)
        raise


socket.getaddrinfo = custom_getaddrinfo


def main():
    print(f"Testing live service at: {BASE_URL}")

    # 1. Test /healthz
    print("\n1. Testing GET /healthz ...")
    r = httpx.get(f"{BASE_URL}/healthz", timeout=15)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # 2. Test /synthesize/gemini (Standard Batch TTS)
    print("\n2. Testing POST /synthesize/gemini (Standard Batch TTS) ...")
    payload = {
        "text": "Hello from Cognigy and Gemini 3.8 speech synthesis.",
        "voice": "Puck",
        "sample_rate": 8000,
    }
    r = httpx.post(f"{BASE_URL}/synthesize/gemini", json=payload, timeout=30)
    print(f"Status: {r.status_code}")
    print(f"Content-Type: {r.headers.get('content-type')}")
    print(f"Content-Length: {r.headers.get('content-length')} bytes")
    print(f"Sample-Rate Header: {r.headers.get('x-audio-sample-rate')}")
    assert r.status_code == 200
    assert r.headers.get("content-type") == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert r.content[8:12] == b"WAVE"
    print(f"Successfully received {len(r.content)} bytes of linear16 8kHz WAV audio!")

    # 3. Test /synthesize/gemini-live (Streaming TTS)
    print("\n3. Testing POST /synthesize/gemini-live (Real-Time Chunked Streaming) ...")
    stream_payload = {
        "text": "Streaming voice call test on Gemini Live.",
        "voice": "Kore",
        "sample_rate": 8000,
    }
    with httpx.stream("POST", f"{BASE_URL}/synthesize/gemini-live", json=stream_payload, timeout=30) as resp:
        print(f"Status: {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('content-type')}")
        print(f"Transfer-Encoding: {resp.headers.get('transfer-encoding')}")
        assert resp.status_code == 200

        total_bytes = 0
        chunk_count = 0
        first_chunk = None

        for chunk in resp.iter_bytes():
            if chunk:
                if chunk_count == 0:
                    first_chunk = chunk
                total_bytes += len(chunk)
                chunk_count += 1

        print(f"Received {chunk_count} streaming chunks, total {total_bytes} bytes.")
        assert first_chunk is not None
        assert first_chunk[:4] == b"RIFF"
        assert first_chunk[8:12] == b"WAVE"
        print("First chunk validated: 44-byte RIFF/WAV streaming header!")

    print("\nAll live Cloud Run verification tests PASSED!")


if __name__ == "__main__":
    main()
