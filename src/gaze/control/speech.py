"""Speaking back: Claude's replies (and questions that need an answer) read aloud with Cartesia Sonic.

Audio is streamed and played as it arrives, so speech starts in a fraction of a second. `stop()` cuts it off at
once: the controller calls it when the user starts talking (barge-in), which it detects from their lips moving
rather than from sound, since the Mac's own speaker would otherwise trigger it.
"""
from __future__ import annotations

import json
import threading
import time

from . import config

TTS_URL = "https://api.cartesia.ai/tts/bytes"
TTS_VERSION = "2026-08-14"
RATE = 24000
CHUNK_BYTES = 4800  # 100 ms of 16-bit mono at RATE


class Speaker:
    def __init__(self, on_state=lambda speaking: None):
        self.on_state = on_state  # called with True when speech starts, False when it ends or is cut off
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.text = ""  # what's being said

    @property
    def available(self) -> bool:
        return bool(config.SPEAK and config.CARTESIA_API_KEY)

    @property
    def speaking(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def say(self, text: str) -> None:
        """Speak `text`, cutting off anything already being said."""
        text = text.strip()
        if not text or not self.available:
            return
        self.stop()
        self._stop = threading.Event()
        self.text = text
        self._thread = threading.Thread(target=self._run, args=(text, self._stop), name="speak", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _run(self, text: str, stop: threading.Event) -> None:
        started = False
        try:
            import ssl
            import urllib.request

            import certifi
            import sounddevice as sd

            body = {"model_id": config.TTS_MODEL, "transcript": text, "voice": {"mode": "id", "id": config.TTS_VOICE},
                    "language": "en", "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": RATE}}
            request = urllib.request.Request(TTS_URL, data=json.dumps(body).encode(), method="POST", headers={
                "Authorization": f"Bearer {config.CARTESIA_API_KEY}", "X-API-Key": config.CARTESIA_API_KEY,
                "Cartesia-Version": TTS_VERSION, "Content-Type": "application/json"})
            t0 = time.perf_counter()
            tls = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(request, timeout=10, context=tls) as response, \
                    sd.RawOutputStream(samplerate=RATE, channels=1, dtype="int16") as out:
                carry = b""
                while not stop.is_set():
                    data = response.read(CHUNK_BYTES)  # streamed: plays while the rest is still being made
                    if not data:
                        break
                    if not started:
                        started = True
                        self.on_state(True)
                        print(f"[speech] first audio after {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)
                    data, carry = carry + data, b""
                    if len(data) % 2:  # a read can split a 16-bit sample: keep the odd byte for the next one
                        data, carry = data[:-1], data[-1:]
                    out.write(data)
                if stop.is_set():
                    out.abort()
        except Exception as e:  # speaking is a nicety: never take voice control down with it
            detail = e.read()[:200] if hasattr(e, "read") else e
            print(f"[speech] failed: {detail}", flush=True)
        finally:
            if started:
                self.on_state(False)
