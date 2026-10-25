"""Always-on mic (Apple voice processing when possible) -> transcript with start/end times.

CartesiaVoice streams to Cartesia Ink-2, which transcribes while you speak and detects the end of the turn itself.
Voice is the on-device fallback: Silero speech detection, optional lip gate, then Whisper.
"""

import ctypes
import ctypes.util
import json
import queue
import threading
import time
import urllib.parse
from collections import deque

import numpy as np
import sounddevice as sd

from . import audio, config

CHUNK_S = audio.CHUNK / audio.RATE
SPEECH_START, SPEECH_END = 0.5, 0.35  # Silero probability to start / keep an utterance (hysteresis)
START_CHUNKS = 3  # ~100 ms of speech starts one: a click or a cough doesn't
PREROLL_S = 1.5  # audio kept from before the lip gate opens, so a late open loses no words
PREROLL_CHUNKS = int(PREROLL_S / CHUNK_S)
END_SILENCE_S = 0.6
MAX_UTTERANCE_S = 15
HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "bye"}  # Whisper's favorite outputs on noise


# Inputs that are usually silent unless something feeds them; never pick these automatically.
VIRTUAL_INPUTS = ("virtual", "blackhole", "loopback", "soundflower", "zoomaudio", "microsoft teams")

# CoreAudio, via ctypes: which input macOS currently uses by default (changes when AirPods connect).
_coreaudio = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreAudio"))


class _PropertyAddress(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


def _fourcc(code):
    return int.from_bytes(code.encode(), "big")


_DEFAULT_INPUT = _PropertyAddress(_fourcc("dIn "), _fourcc("glob"), 0)  # kAudioHardwarePropertyDefaultInputDevice


def default_input_id():
    """CoreAudio id of the system default input (cheap; used to notice when it changes)."""
    device, size = ctypes.c_uint32(0), ctypes.c_uint32(4)
    err = _coreaudio.AudioObjectGetPropertyData(1, ctypes.byref(_DEFAULT_INPUT), 0, None,  # 1 = system object
                                                ctypes.byref(size), ctypes.byref(device))
    return device.value if err == 0 else None


def find_mic(name):
    """Index of the first input device whose name contains `name`, else None."""
    for i, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0 and name.lower() in device["name"].lower():
            return i
    return None


def input_devices():
    """Names of the microphones PortAudio knows about (refreshed whenever the voice thread switches mics)."""
    names = []
    for device in sd.query_devices():
        if device["max_input_channels"] > 0 and device["name"] not in names:
            names.append(device["name"])
    return names


def pick_mic(preferred=None):
    """(device index, name): `preferred` if it's connected, else the system default input, unless that's a
    silent virtual device, in which case the built-in mic."""
    if preferred:
        index = find_mic(preferred)
        if index is not None:
            return index, sd.query_devices(index)["name"]
    try:
        default = sd.query_devices(kind="input")
        if not any(v in default["name"].lower() for v in VIRTUAL_INPUTS):
            return default["index"], default["name"]
    except sd.PortAudioError:
        pass  # no default input right now
    index = find_mic("MacBook") or find_mic("Built-in")
    return index, (sd.query_devices(index)["name"] if index is not None else "system default")


class Voice(threading.Thread):
    def __init__(self, on_utterance, on_state, on_device=lambda name: None, on_partial=lambda text: None, gate=None):
        """on_utterance(text, t_start, t_end) with unix timestamps; on_state("listening" | "transcribing" | "idle");
        on_device(name) whenever the microphone in use changes; on_partial(text) with words so far (Cartesia only).
        gate: a lips.LipGate; speech only counts while it is open."""
        super().__init__(daemon=True)
        self.on_utterance = on_utterance
        self.on_state = on_state
        self.on_device = on_device
        self.on_partial = on_partial
        self.gate = gate
        self.choice = config.MIC_NAME or None  # None: follow the system default input
        self.isolation = config.VOICE_PROCESSING  # Apple voice processing on the default input
        self.device = ""  # name of the mic in use
        self._restart = threading.Event()  # reopen the stream (another mic chosen, or turned on/off)
        self._enabled = threading.Event()
        self._enabled.set()

    def use(self, name):
        """Switch to the mic named `name`, or None to follow the system default (callable from any thread)."""
        self.choice = name
        self._restart.set()

    def set_isolation(self, on):
        """Apple voice processing on or off (reopens the mic)."""
        self.isolation = on
        self._restart.set()

    def set_enabled(self, on):
        """Off closes the stream, so macOS shows the mic as unused; Whisper stays loaded."""
        (self._enabled.set if on else self._enabled.clear)()
        self._restart.set()

    def _open(self):
        """(input, name): Apple's voice processing when following the system default, else the raw device."""
        mic, name = pick_mic(self.choice)
        try:
            is_default = self.choice is None and mic == sd.query_devices(kind="input")["index"]
        except sd.PortAudioError:
            is_default = False
        if self.isolation and is_default:
            source = audio.AppleVoiceInput()
            try:
                source.start()
                return source, f"{name} · {source.label}"
            except Exception as e:
                print(f"Voice isolation off ({e}); using the raw mic.", flush=True)
        source = audio.PortAudioInput(mic)
        source.start()
        return source, name

    def _wait_enabled(self):
        """Block while voice control is off (the mic is closed then)."""
        if not self._enabled.is_set():
            self.device = ""
            self.on_device("")
            self.on_state("off")
            self._enabled.wait()
        self._restart.clear()

    def run(self):
        from faster_whisper import WhisperModel  # slow import; keep it off the UI thread

        model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        vad = audio.SpeechDetector()
        while True:
            self._wait_enabled()
            # PortAudio caches the device list at startup; re-initialize so newly connected devices show up.
            sd._terminate()
            sd._initialize()
            watched = default_input_id() if self.choice is None else None  # only follow the default when unset
            try:
                source, name = self._open()
            except Exception as e:  # no usable input right now (e.g. AirPods mid-connect)
                print(f"Microphone error ({e}); retrying...", flush=True)
                time.sleep(1)
                continue
            print(f"Microphone: {name}", flush=True)
            self.device = name
            self.on_device(name)
            try:
                self.on_state("idle")
                vad.reset()
                self._listen(model, vad, source.blocks, watched)
            except sd.PortAudioError as e:  # device vanished mid-stream (e.g. AirPods taken out)
                print(f"Microphone error ({e}); switching...", flush=True)
                time.sleep(1)
            finally:
                try:
                    source.stop()
                except Exception:
                    pass

    def _chunks(self, blocks, watched):
        """32 ms chunks from the input, until a restart is asked for or the system default input changes."""
        pending = np.zeros(0, np.float32)
        last_check = time.time()
        while not self._restart.is_set():
            if watched is not None and time.time() - last_check > 1.0:
                last_check = time.time()
                if default_input_id() != watched:
                    return
            try:
                pending = np.concatenate([pending, blocks.get(timeout=0.5)])
            except queue.Empty:
                continue  # stream stalled (device switching); the checks above handle it
            while len(pending) >= audio.CHUNK:
                yield pending[:audio.CHUNK]
                pending = pending[audio.CHUNK:]

    def _listen(self, model, vad, blocks, watched):
        """Speech (by Silero, not loudness) starts an utterance; END_SILENCE_S without it ends one."""
        preroll = deque(maxlen=PREROLL_CHUNKS)
        speaking, voiced, silence, buf, started = False, 0, 0, [], 0.0
        for chunk in self._chunks(blocks, watched):
            p = vad(chunk)
            if not speaking:
                preroll.append(chunk)
                voiced = voiced + 1 if p >= SPEECH_START else 0
                if voiced >= START_CHUNKS and (self.gate is None or self.gate.is_open()):
                    speaking, buf, silence = True, list(preroll), 0
                    started = time.time() - len(preroll) * CHUNK_S
                    self.on_state("listening")
                continue

            buf.append(chunk)
            silence = silence + 1 if p < SPEECH_END else 0
            if silence * CHUNK_S < END_SILENCE_S and len(buf) * CHUNK_S < MAX_UTTERANCE_S:
                continue

            speaking, ended = False, time.time() - silence * CHUNK_S
            preroll.clear()
            audio_clip = np.concatenate(buf)
            self.on_state("transcribing")
            segments, _ = model.transcribe(audio_clip, language="en", beam_size=1, condition_on_previous_text=False)
            text = " ".join(s.text for s in segments).strip()
            if text.lower().strip(" .!?,") not in HALLUCINATIONS:
                self.on_utterance(text, started, ended)
            self.on_state("idle")


CARTESIA_URL = "wss://api.cartesia.ai/stt/turns/websocket"
CARTESIA_VERSION = "2026-03-01"
SEND_S = 0.1  # Cartesia recommends ~100 ms audio chunks
SEND_SAMPLES = int(audio.RATE * SEND_S)
SPEECH_LEAD_S = 0.3  # turn.start arrives a little after speech begins
KEEPALIVE_S = 60  # gate closed = nothing sent, and Cartesia drops a connection idle for 180 s: 0.1 s of silence this often
SILENCE = bytes(SEND_SAMPLES * 2)
MAX_FAILURES = 3  # consecutive connection failures before falling back to Whisper
# Phrases the recognizer should expect (Cartesia keyterms: up to 100 terms, 1200 characters). Running and installed
# app names fill the rest, so "open Blender" isn't heard as "open blunder".
KEYTERMS = [
    "click", "double click", "right click", "scroll up", "scroll down", "scroll left", "scroll right", "a lot",
    "a little", "go back", "go forward", "new tab", "close tab", "reopen tab", "next tab", "previous tab",
    "new window", "full screen", "zoom in", "zoom out", "select all", "undo", "redo", "copy", "paste", "refresh",
    "minimize", "Spotlight", "Mission Control", "stop", "cancel", "never mind", "confirm", "calibrate", "type",
]


def keyterms():
    from .commands import available_apps  # AppKit; running apps come first

    terms = list(dict.fromkeys(KEYTERMS + ([config.WAKE_WORD] if config.WAKE_WORD else [])))
    for name in available_apps():
        if len(terms) >= 100:
            break
        if sum(map(len, terms)) + len(name) <= 1200:
            terms.append(name)
    return terms


def _pcm16(samples):
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class CartesiaVoice(Voice):
    """Streams the mic to Cartesia Ink-2 (same callbacks as Voice). No local silence wait: Ink-2 sends the final
    transcript as soon as it decides the turn is over. Falls back to on-device Whisper if Cartesia is unreachable.
    Audio comes from the same inputs as Voice (Apple voice processing on the default mic), as 16-bit PCM."""

    def run(self):
        import ssl

        import certifi  # python.org builds ship without CA certificates
        from websockets.sync.client import connect

        tls = ssl.create_default_context(cafile=certifi.where())
        self._failures = 0
        while self._failures < MAX_FAILURES:
            self._wait_enabled()
            sd._terminate()  # see Voice.run: pick up newly connected mics
            sd._initialize()
            watched = default_input_id() if self.choice is None else None
            try:
                source, name = self._open()
            except Exception as e:  # no usable input right now
                print(f"Microphone error ({e}); retrying...", flush=True)
                time.sleep(1)
                continue
            print(f"Microphone: {name} (Cartesia {config.CARTESIA_MODEL})", flush=True)
            self.device = name
            self.on_device(name)
            try:
                with connect(self._url(), additional_headers={"X-API-Key": config.CARTESIA_API_KEY,
                                                              "Cartesia-Version": CARTESIA_VERSION},
                             ssl=tls, open_timeout=5) as ws:
                    self._stream(ws, source.blocks, watched)
            except Exception as e:  # network down, bad key: retry, then fall back
                self._failures += 1
                print(f"Cartesia error ({e}); retrying...", flush=True)
                time.sleep(1)
            finally:
                try:
                    source.stop()
                except Exception:
                    pass
        print("Cartesia unavailable; switching to on-device Whisper.", flush=True)
        super().run()

    def _url(self):
        params = [("model", config.CARTESIA_MODEL), ("encoding", "pcm_s16le"), ("sample_rate", str(audio.RATE)),
                  ("turn_end_threshold", config.CARTESIA_END_THRESHOLD)]
        params += [("keyterm", term) for term in keyterms()]
        return f"{CARTESIA_URL}?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"

    def _stream(self, ws, blocks, watched):
        """Send audio and handle turn events until the connection drops, another mic is picked or voice control is
        turned off (return), or the default mic changes.

        Audio only goes out while the lip gate is open or a turn is in progress (Cartesia bills every second sent).
        While closed it's kept in a pre-roll buffer, sent first when the gate opens, so the start isn't lost."""
        stop, failed = threading.Event(), threading.Event()
        in_turn = threading.Event()
        self._opened_at = None  # when the gate last opened: speech started about then

        def send_audio():  # off the audio callback, which must never block on the network
            pending = np.zeros(0, np.float32)
            preroll = deque(maxlen=int(PREROLL_S / SEND_S))
            sending, last_sent = False, time.time()
            while not stop.is_set():
                try:
                    pending = np.concatenate([pending, blocks.get(timeout=0.5)])
                except queue.Empty:
                    continue
                while len(pending) >= SEND_SAMPLES:
                    block, pending = _pcm16(pending[:SEND_SAMPLES]), pending[SEND_SAMPLES:]
                    try:
                        if self.gate is None or self.gate.is_open() or in_turn.is_set():
                            if not sending:
                                sending, self._opened_at = True, time.time() - len(preroll) * SEND_S
                                for b in preroll:
                                    ws.send(b)
                                preroll.clear()
                            ws.send(block)
                            last_sent = time.time()
                        else:
                            sending = False
                            preroll.append(block)
                            if KEEPALIVE_S and time.time() - last_sent > KEEPALIVE_S:
                                ws.send(SILENCE)
                                last_sent = time.time()
                    except Exception:
                        failed.set()
                        stop.set()
                        return

        threading.Thread(target=send_audio, daemon=True).start()
        self.on_state("idle")
        started = last_word = None
        try:
            while not stop.is_set() and not self._restart.is_set():
                try:
                    msg = json.loads(ws.recv(timeout=1.0))
                except TimeoutError:
                    if started is None and watched is not None and default_input_id() != watched:
                        return
                    continue
                kind = msg.get("type")
                if kind == "connected":
                    self._failures = 0
                elif kind == "turn.start":
                    in_turn.set()
                    started, last_word = time.time() - SPEECH_LEAD_S, time.time()
                    if self._opened_at is not None:  # pre-roll arrives in a burst: speech began at the gate
                        started = max(min(started, self._opened_at + PREROLL_S), self._opened_at)
                    self.on_state("listening")
                elif kind == "turn.update":
                    last_word = time.time()
                    if msg.get("transcript"):
                        self.on_partial(msg["transcript"])
                elif kind == "turn.end":
                    in_turn.clear()
                    text = msg.get("transcript", "").strip()
                    now = time.time()
                    if text:
                        self.on_utterance(text, started or now - 1.0, last_word or now)
                    started = last_word = None
                    self.on_state("idle")
                elif kind == "error":
                    raise RuntimeError(f"{msg.get('error_code', 'error')}: {msg.get('message')}")
        finally:
            stop.set()
        if failed.is_set():
            raise ConnectionError("audio upload stopped")


class TypedInput(threading.Thread):
    """Dev mode: type commands in the terminal instead of speaking (gaze is taken at the moment you hit Enter)."""

    def __init__(self, on_utterance):
        super().__init__(daemon=True)
        self.on_utterance = on_utterance

    def run(self):
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                return
            if text:
                now = time.time()
                self.on_utterance(text, now - 0.3, now)
