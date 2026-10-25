"""Always-on mic (Apple voice processing when possible) -> Silero speech detection -> optional lip gate ->
on-device Whisper transcript, with start/end times."""

import ctypes
import ctypes.util
import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

from . import audio, config

CHUNK_S = audio.CHUNK / audio.RATE
SPEECH_START, SPEECH_END = 0.5, 0.35  # Silero probability to start / keep an utterance (hysteresis)
START_CHUNKS = 3  # ~100 ms of speech starts one: a click or a cough doesn't
PREROLL_CHUNKS = 10  # ~320 ms kept from before the start, so the first syllable isn't cut
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
    def __init__(self, on_utterance, on_state, on_device=lambda name: None, accept=None, on_rejected=None):
        """on_utterance(text, t_start, t_end) with unix timestamps; on_state("listening" | "transcribing" | "idle");
        on_device(name) whenever the microphone in use changes. accept(t_start, t_end) -> (bool, info) runs before
        transcription (the lip gate); rejected speech goes to on_rejected(t_start, t_end, info) untranscribed."""
        super().__init__(daemon=True)
        self.on_utterance = on_utterance
        self.on_state = on_state
        self.on_device = on_device
        self.accept = accept
        self.on_rejected = on_rejected or (lambda *a: None)
        self.choice = config.MIC_NAME or None  # None: follow the system default input
        self.device = ""  # name of the mic in use
        self._restart = threading.Event()  # reopen the stream (another mic chosen, or turned on/off)
        self._enabled = threading.Event()
        self._enabled.set()

    def use(self, name):
        """Switch to the mic named `name`, or None to follow the system default (callable from any thread)."""
        self.choice = name
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
        if config.VOICE_PROCESSING and is_default:
            source = audio.AppleVoiceInput()
            try:
                source.start()
                return source, f"{name} · {source.label}"
            except Exception as e:
                print(f"Voice isolation off ({e}); using the raw mic.", flush=True)
        source = audio.PortAudioInput(mic)
        source.start()
        return source, name

    def run(self):
        from faster_whisper import WhisperModel  # slow import; keep it off the UI thread

        model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        vad = audio.SpeechDetector()
        while True:
            if not self._enabled.is_set():
                self.device = ""
                self.on_device("")
                self.on_state("off")
                self._enabled.wait()
            self._restart.clear()
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
                if voiced >= START_CHUNKS:
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
            if self.accept is not None:
                ok, info = self.accept(started, ended)
                if not ok:  # e.g. your lips weren't moving: someone else, or the TV
                    self.on_rejected(started, ended, info)
                    self.on_state("idle")
                    continue
            audio_clip = np.concatenate(buf)
            self.on_state("transcribing")
            segments, _ = model.transcribe(audio_clip, language="en", beam_size=1, condition_on_previous_text=False)
            text = " ".join(s.text for s in segments).strip()
            if text.lower().strip(" .!?,") not in HALLUCINATIONS:
                self.on_utterance(text, started, ended)
            self.on_state("idle")


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
