"""What the dashboard shows: the live tracker, or a recorded session played back.

A feed produces the messages the server sends: `hello()` for a client that just connected, `poll()` for
the regular broadcast (~30 Hz), and handles its own commands (`command()`; e.g. seeking a replay).
Text messages are JSON strings; images are bytes prefixed with IMAGE_PREFIX.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np

from ..core.device import load_device
from ..core.types import Frame, Sample
from ..telemetry import calibration_info, frame_state, scene_info

IMAGE_HZ = 12
IMAGE_WIDTH = 640
IMAGE_PREFIX = b"JPG0"

Message = str | bytes


def jpeg(image: np.ndarray) -> bytes:
    """RGB image → JPEG, at most IMAGE_WIDTH wide."""
    if image.shape[1] > IMAGE_WIDTH:
        h = round(image.shape[0] * IMAGE_WIDTH / image.shape[1])
        image = cv2.resize(image, (IMAGE_WIDTH, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return IMAGE_PREFIX + buf.tobytes() if ok else b""


class Feed(Protocol):
    def hello(self) -> list[Message]: ...
    def poll(self) -> list[Message]: ...
    def command(self, name: str, value) -> bool: ...


class LiveFeed:
    """The running tracker. States are built here, on the server thread, from the newest sample, so a
    slow client never slows tracking down."""

    def __init__(self, tracker, context: Callable[[], dict] | None = None, lid: Callable[[], float | None] | None = None):
        self.tracker = tracker
        self.context = context or (lambda: {})
        self.lid = lid or (lambda: None)
        self._latest: tuple[Frame, Sample] | None = None
        self._last_t = 0
        self._last_image = 0.0
        self._last_calib = self._calib_key()
        tracker.add_frame_listener(self._on_frame)

    def _on_frame(self, frame: Frame, sample: Sample) -> None:
        self._latest = (frame, sample)  # a single reference swap; poll() reads the newest

    def _calib_key(self):
        profile = self.tracker.profile
        return id(profile), getattr(getattr(profile, "meta", None), "updated", None), \
            getattr(getattr(profile, "meta", None), "offset_pt", None)

    def hello(self) -> list[Message]:
        t = self.tracker
        return [json.dumps(scene_info(t.screen, t.plane, load_device(), self.lid())),
                json.dumps(calibration_info(t.profile))]

    def poll(self) -> list[Message]:
        latest = self._latest
        if latest is None:
            return []
        frame, sample = latest
        out: list[Message] = []
        key = self._calib_key()
        if key != self._last_calib:
            self._last_calib = key
            out.append(json.dumps(calibration_info(self.tracker.profile)))
        if sample.t_ns != self._last_t:
            self._last_t = sample.t_ns
            state = frame_state(sample, self.tracker.profile, self.tracker.plane, self.context(), self.tracker.fps)
            out.append(json.dumps(state))
        now = time.monotonic()
        if now - self._last_image >= 1 / IMAGE_HZ:
            self._last_image = now
            out.append(jpeg(frame.image))
        return out

    def command(self, name: str, value) -> bool:
        return False


@dataclass
class ReplayFrame:
    state: str  # JSON frame state
    image: bytes  # JPEG message


class ReplayFeed:
    """A recorded session, precomputed, played back at `fps` with play / pause / seek / speed commands."""

    def __init__(self, frames: list[ReplayFrame], scene: dict, calibration: dict, name: str, fps: float = 30.0):
        self.frames, self.name, self.fps = frames, name, fps
        self.scene = {**scene, "replay": {"session": name, "frames": len(frames),
                                          "markers": self._markers(frames)}}
        self.calibration = calibration
        self.index, self.playing, self.speed = 0, True, 1.0
        self._clock = time.monotonic()
        self._sent = -1

    @staticmethod
    def _markers(frames: list[ReplayFrame]) -> list[dict]:
        """Where each shown target starts, for the timeline."""
        out, last = [], None
        for i, f in enumerate(frames):
            s = json.loads(f.state)
            key = (s.get("phase"), tuple(s["target"]) if s.get("target") else None)
            if key != last:
                out.append({"index": i, "phase": s.get("phase")})
                last = key
        return out

    def hello(self) -> list[Message]:
        self._sent = -1  # resend the current frame to the new client
        return [json.dumps(self.scene), json.dumps(self.calibration)]

    def poll(self) -> list[Message]:
        now = time.monotonic()
        if self.playing:
            steps = int((now - self._clock) * self.fps * self.speed)
            if steps:
                self._clock = now
                self.index = (self.index + steps) % len(self.frames)
        else:
            self._clock = now
        if self.index == self._sent:
            return []
        self._sent = self.index
        f = self.frames[self.index]
        state = json.loads(f.state)
        state["replay"] = {"index": self.index, "playing": self.playing, "speed": self.speed}
        return [json.dumps(state), f.image]

    def command(self, name: str, value) -> bool:
        if name == "play":
            self.playing = True
        elif name == "pause":
            self.playing = False
        elif name == "seek" and value is not None:
            self.index = int(np.clip(int(value), 0, len(self.frames) - 1))
        elif name == "step" and value is not None:
            self.playing = False
            self.index = int(np.clip(self.index + int(value), 0, len(self.frames) - 1))
        elif name == "speed" and value:
            self.speed = float(value)
        else:
            return False
        self._sent = -1
        return True
