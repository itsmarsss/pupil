"""Streams gaze points over local UDP so another process (the voice controller in
`gaze.control`) can use them, and takes commands back (e.g. "recalibrate").

Points are JSON: {"t": wall-clock seconds, "x", "y": screen points (top-left origin,
smoothed), "state", "fixation", "confidence", "busy": calibration on screen, "mouth": how open
the mouth is, when a face is visible, for the controller's lip-movement gate}.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Callable

import numpy as np

from ..core.types import FaceObservation, Sample

log = logging.getLogger(__name__)

GAZE_PORT = 47800  # points out
CMD_PORT = 47801  # commands in
# MediaPipe face mesh: inner lips (upper, lower) and mouth corners
LIP_TOP, LIP_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 78, 308


def mouth_open(face: FaceObservation) -> float | None:
    """Inner-lip gap over mouth width: ~0 closed, ~0.3+ wide open; independent of distance to the camera."""
    lm = face.landmarks
    width = float(np.linalg.norm(lm[MOUTH_LEFT, :2] - lm[MOUTH_RIGHT, :2]))
    return float(np.linalg.norm(lm[LIP_TOP, :2] - lm[LIP_BOTTOM, :2])) / width if width > 0 else None


class GazePublisher:
    """Attach `on_sample` as a tracker listener; `busy()` says whether a calibration or
    check is on screen (the controller pauses while it is)."""

    def __init__(self, busy: Callable[[], bool] = lambda: False, port: int = GAZE_PORT):
        self._busy = busy
        self._addr = ("127.0.0.1", port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # monotonic → wall clock: the controller timestamps speech with time.time()
        self._offset = time.time() - time.monotonic_ns() / 1e9

    def message(self, s: Sample) -> dict:
        p = s.point
        msg = {"t": round(self._offset + s.t_ns / 1e9, 4), "state": p.state.value, "busy": bool(self._busy()),
               "confidence": round(float(p.confidence), 3), "fixation": p.fixation_id}
        if p.valid:
            msg["x"], msg["y"] = round(float(p.x), 1), round(float(p.y), 1)
        if s.face is not None and (m := mouth_open(s.face)) is not None:
            msg["mouth"] = round(m, 4)
        return msg

    def on_sample(self, s: Sample) -> None:
        try:
            self._sock.sendto(json.dumps(self.message(s)).encode(), self._addr)
        except OSError:
            pass  # nobody listening is fine

    def listen(self, handlers: dict[str, Callable[[], None]], port: int = CMD_PORT) -> None:
        """Run `handlers[cmd]` for each {"cmd": ...} received, on a background thread."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port))

        def loop():
            while True:
                try:
                    cmd = json.loads(sock.recv(4096)).get("cmd")
                except (OSError, ValueError, AttributeError):
                    continue
                if cmd in handlers:
                    log.info("remote command: %s", cmd)
                    handlers[cmd]()

        threading.Thread(target=loop, name="commands", daemon=True).start()
