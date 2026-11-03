"""Interactive CLI tool to test Cognigy Voice Gateway WebSocket STT endpoint."""

import argparse
import asyncio
import inspect
import json
import time
import wave
import websockets


async def run_test(url: str, token: str = None, wav_path: str = None, language: str = "en-US", verbose: bool = False):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"\n🔌 Connecting to: {url}")
    t_start = time.perf_counter()
    try:
        connect_kwargs = {}
        if headers:
            if "additional_headers" in inspect.signature(websockets.connect).parameters:
                connect_kwargs["additional_headers"] = headers
            else:
                connect_kwargs["extra_headers"] = headers

        async with websockets.connect(url, **connect_kwargs) as ws:
            conn_elapsed = (time.perf_counter() - t_start) * 1000
            print(f"✅ WebSocket connection established in {conn_elapsed:.1f} ms!")

            # 1. Send Cognigy Start frame
            sample_rate = 8000
            if wav_path:
                try:
                    with wave.open(wav_path, "rb") as wf:
                        sample_rate = wf.getframerate()
                        print(f"🎵 Using WAV file '{wav_path}' with sample rate: {sample_rate} Hz")
                except Exception as e:
                    print(f"⚠️ Could not read WAV header: {e}. Defaulting to 8000 Hz.")

            start_msg = {
                "type": "start",
                "language": language,
                "format": "raw",
                "encoding": "LINEAR16",
                "sampleRateHz": sample_rate,
                "interimResults": True,
            }
            print(f"📤 Sending START frame: {json.dumps(start_msg)}")
            t_start_sent = time.perf_counter()
            await ws.send(json.dumps(start_msg))

            # Background listener task for server responses
            stop_listening = asyncio.Event()
            events_received = 0

            async def listen_server():
                nonlocal events_received
                try:
                    async for raw_msg in ws:
                        events_received += 1
                        recv_ts = time.perf_counter() - t_start_sent
                        try:
                            data = json.loads(raw_msg)
                            msg_type = data.get("type")
                            if msg_type == "transcription":
                                is_final = data.get("is_final")
                                alts = data.get("alternatives", [{}])
                                text = alts[0].get("transcript", "")
                                if is_final:
                                    conf = alts[0].get("confidence", 1.0)
                                    print(f"\n🎯 [FINAL @ +{recv_ts:.2f}s] (conf: {conf:.2f}): \033[1;32m{text}\033[0m")
                                else:
                                    print(f"⏳ [Interim @ +{recv_ts:.2f}s]: \033[33m{text}\033[0m", end="\r")
                            elif msg_type == "error":
                                print(f"\n❌ [SERVER ERROR @ +{recv_ts:.2f}s]: {data.get('error')}")
                            else:
                                print(f"\n📩 [Server Frame @ +{recv_ts:.2f}s]: {data}")
                        except Exception:
                            print(f"\n📩 [Raw text @ +{recv_ts:.2f}s]: {raw_msg}")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    if verbose:
                        print(f"\n⚠️ Listener notice: {e}")
                finally:
                    stop_listening.set()

            listen_task = asyncio.create_task(listen_server())

            # 2. Stream audio
            chunks_sent = 0
            if wav_path:
                print(f"🎙️ Streaming audio from {wav_path} in 20ms chunks...")
                with wave.open(wav_path, "rb") as wf:
                    chunk_size = int(sample_rate * 2 * 0.02)  # 20ms of 16-bit mono PCM
                    while True:
                        data = wf.readframes(chunk_size // 2)
                        if not data:
                            break
                        chunks_sent += 1
                        await ws.send(data)
                        if verbose and chunks_sent % 50 == 0:
                            print(f"   [Sent {chunks_sent} chunks / {(chunks_sent * 0.02):.1f}s]")
                        await asyncio.sleep(0.02)
            else:
                print("🎙️ Streaming 3 seconds of dummy silence PCM frames (8000 Hz)...")
                # 20ms of 8000Hz 16-bit mono = 320 bytes per frame
                dummy_pcm = b"\x00" * 320
                for _ in range(150):  # 150 * 20ms = 3.0s
                    chunks_sent += 1
                    await ws.send(dummy_pcm)
                    await asyncio.sleep(0.02)

            print(f"\n⏸️ Finished streaming audio ({chunks_sent} chunks sent). Waiting 2s for responses...")
            await asyncio.sleep(2.0)

            # 3. Send Stop frame
            stop_msg = {"type": "stop"}
            print(f"📤 Sending STOP frame: {json.dumps(stop_msg)}")
            await ws.send(json.dumps(stop_msg))

            await asyncio.sleep(0.5)
            await ws.close()
            listen_task.cancel()
            print(f"🏁 Test completed! Total responses received: {events_received}\n")

    except Exception as e:
        status_code = getattr(e, "status_code", None) or (
            hasattr(e, "response") and getattr(e.response, "status_code", None)
        )
        if status_code:
            print(f"\n❌ Handshake failed: HTTP Status Code {status_code}")
            if status_code == 404:
                print("   Hint: 404 means the path /transcribe/gemini was not found on the backend.")
            elif status_code in (1008, 401, 403):
                print("   Hint: Authentication failed. Did you supply --token?")
        else:
            print(f"\n❌ Connection error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test Cognigy Voice Gateway WebSocket STT endpoint")
    parser.add_argument(
        "--provider",
        choices=["google", "gemini"],
        default="google",
        help="STT engine to test: 'google' (Google Cloud Speech / Chirp) or 'gemini' (Gemini Live) (default: google)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="WebSocket URL (default auto-computed based on --provider)",
    )
    parser.add_argument("--token", default=None, help="Optional Bearer token for authentication")
    parser.add_argument("--wav", default=None, help="Path to a WAV file to stream (16-bit PCM)")
    parser.add_argument("--lang", default="en-US", help="Language code (default: en-US)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose client logs")

    args = parser.parse_args()
    url = args.url
    if not url:
        url = f"wss://207.175.38.3.sslip.io/transcribe/{args.provider}"

    asyncio.run(
        run_test(
            url=url,
            token=args.token,
            wav_path=args.wav,
            language=args.lang,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()

