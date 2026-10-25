"""Always-on mic -> energy-based utterance detection -> on-device Whisper transcript (with start/end times)."""

import ctypes
import ctypes.util
import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

from . import config

RATE = 16000
BLOCK_S = 0.03
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
    def __init__(self, on_utterance, on_state, on_device=lambda name: None):
        """on_utterance(text, t_start, t_end) with unix timestamps; on_state("listening" | "transcribing" | "idle");
        on_device(name) whenever the microphone in use changes."""
        super().__init__(daemon=True)
        self.on_utterance = on_utterance
        self.on_state = on_state
        self.on_device = on_device
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

    def run(self):
        from faster_whisper import WhisperModel  # slow import; keep it off the UI thread

        model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
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
            mic, name = pick_mic(self.choice)
            watched = default_input_id() if self.choice is None else None  # only follow the default when unset
            print(f"Microphone: {name}", flush=True)
            self.device = name
            self.on_device(name)
            blocks = queue.Queue()
            try:
                with sd.InputStream(device=mic, samplerate=RATE, channels=1, dtype="float32",
                                    blocksize=int(RATE * BLOCK_S),
                                    callback=lambda data, *_: blocks.put(data[:, 0].copy())):
                    self.on_state("idle")
                    self._listen(model, blocks, watched)
            except sd.PortAudioError as e:  # device vanished mid-stream (e.g. AirPods taken out)
                print(f"Microphone error ({e}); switching...", flush=True)
                time.sleep(1)

    def _listen(self, model, blocks, watched):
        """Detect utterances until the system default input changes or a restart is asked for (then return so
        run() switches mics)."""
        noise = 0.01
        preroll = deque(maxlen=10)
        speaking, voiced, silence, buf, started = False, 0, 0, [], 0.0
        last_check = time.time()
        while True:
            if self._restart.is_set():
                return
            if watched is not None and not speaking and time.time() - last_check > 1.0:
                last_check = time.time()
                if default_input_id() != watched:
                    return
            try:
                block = blocks.get(timeout=0.5)
            except queue.Empty:
                continue  # stream stalled (device switching); the default-input check above handles it
            rms = float(np.sqrt(np.mean(block ** 2)))
            threshold = max(0.015, noise * 3)
            if not speaking:
                if rms < threshold:
                    noise = 0.95 * noise + 0.05 * rms
                preroll.append(block)
                voiced = voiced + 1 if rms > threshold else 0
                if voiced >= 3:
                    speaking, buf, silence = True, list(preroll), 0
                    started = time.time() - len(preroll) * BLOCK_S
                    self.on_state("listening")
                continue

            buf.append(block)
            silence = silence + 1 if rms < threshold * 0.8 else 0
            if silence * BLOCK_S < END_SILENCE_S and len(buf) * BLOCK_S < MAX_UTTERANCE_S:
                continue

            speaking, ended = False, time.time() - silence * BLOCK_S
            audio = np.concatenate(buf)
            self.on_state("transcribing")
            segments, _ = model.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False)
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
