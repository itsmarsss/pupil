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
VOICES_URL = "https://api.cartesia.ai/voices?limit=100&language=en"
MENU_VOICES = 24  # shown in the ◉ menu (Cartesia lists its featured voices first)
VOICE_FILE = "voice.json"  # the chosen voice, in the app's support folder
TTS_VERSION = "2026-08-14"
RATE = 24000
CHUNK_BYTES = 4800  # 100 ms of 16-bit mono at RATE


def _headers():
    return {"Authorization": f"Bearer {config.CARTESIA_API_KEY}", "X-API-Key": config.CARTESIA_API_KEY,
            "Cartesia-Version": TTS_VERSION, "Content-Type": "application/json"}


def _tls():
    import ssl

    import certifi

    return ssl.create_default_context(cafile=certifi.where())


class Speaker:
    def __init__(self, on_state=lambda speaking: None):
        self.on_state = on_state  # called with True when speech starts, False when it ends or is cut off
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.text = ""  # what's being said
        self.voice = self._saved_voice() or config.TTS_VOICE  # Cartesia voice id
        self.voices: list[dict] = []  # Cartesia's voices ({"id", "name", "tagline"}), fetched in the background
        if self.available:
            threading.Thread(target=self._fetch_voices, name="voices", daemon=True).start()

    # ---- voices ----------------------------------------------------------------------------

    @staticmethod
    def _voice_path():
        from ..core.paths import APP_SUPPORT

        return APP_SUPPORT / VOICE_FILE

    def _saved_voice(self):
        try:
            return json.loads(self._voice_path().read_text()).get("id")
        except (OSError, ValueError):
            return None

    def _fetch_voices(self):
        import urllib.request

        try:
            request = urllib.request.Request(VOICES_URL, headers=_headers())
            data = json.loads(urllib.request.urlopen(request, timeout=10, context=_tls()).read())
            self.voices = [{"id": v["id"], "name": v["name"], "tagline": v.get("tagline") or ""}
                           for v in data.get("data", []) if v.get("status", "active") == "active"]
        except Exception as e:
            print(f"[speech] couldn't list voices: {e}", flush=True)

    @property
    def voice_name(self) -> str:
        return next((v["name"] for v in self.voices if v["id"] == self.voice), "")

    def menu_voices(self) -> list[dict]:
        """The voices to offer: the first MENU_VOICES, plus the current one if it isn't among them."""
        shown = self.voices[:MENU_VOICES]
        current = next((v for v in self.voices if v["id"] == self.voice), None)
        return shown + ([current] if current is not None and current not in shown else [])

    def set_voice(self, voice_id: str, preview: bool = True) -> None:
        """Use `voice_id` from now on (remembered across restarts), and say hello in it."""
        self.voice = voice_id
        try:
            self._voice_path().parent.mkdir(parents=True, exist_ok=True)
            self._voice_path().write_text(json.dumps({"id": voice_id}))
        except OSError:
            pass
        if preview and self.voice_name:
            self.say(f"Hi, I'm {self.voice_name}.")

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
            import urllib.request

            import sounddevice as sd

            body = {"model_id": config.TTS_MODEL, "transcript": text, "voice": {"mode": "id", "id": self.voice},
                    "language": "en", "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": RATE}}
            request = urllib.request.Request(TTS_URL, data=json.dumps(body).encode(), method="POST", headers=_headers())
            t0 = time.perf_counter()
            with urllib.request.urlopen(request, timeout=10, context=_tls()) as response, \
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
