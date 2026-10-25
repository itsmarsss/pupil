"""Receives gaze points from the tracker (`gaze --publish`) and keeps a short history.

The tracker already calibrates and smooths (fixation hold), so points are used as-is.
"""

import json
import socket
import statistics
import threading
import time
from collections import deque

from . import config


class GazeClient(threading.Thread):
    calibratable = True

    def __init__(self):
        super().__init__(daemon=True)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", config.GAZE_PORT))
        self._sock.settimeout(0.5)
        self._cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()
        self._history = deque(maxlen=900)  # (t, x, y) in screen points, ~30 s at 30 Hz
        self._last_seen = 0.0  # any message: the tracker is running
        self._last_point = 0.0  # a valid point: the tracker can see you
        self.busy = False  # the tracker has a calibration or accuracy check on screen
        self.lips = None  # a lips.LipGate fed with every tracker message

    def run(self):
        while True:
            try:
                msg = json.loads(self._sock.recv(4096))
            except (socket.timeout, ValueError):
                continue
            now = time.time()
            self._last_seen = now
            self.busy = msg.get("busy", False)
            if self.lips is not None:
                self.lips.update(msg, now)
            if "x" not in msg:
                continue  # no face, blink or uncalibrated: keep the last point
            with self._lock:
                self._history.append((msg["t"], msg["x"], msg["y"]))
                self._last_point = now

    @property
    def connected(self):
        return time.time() - self._last_seen < config.GAZE_STALE_S

    def point(self):
        """Latest gaze point in screen points, or None if the tracker has lost you."""
        with self._lock:
            if not self._history or self.busy or time.time() - self._last_point > config.GAZE_STALE_S:
                return None
            _, x, y = self._history[-1]
            return x, y

    def point_during(self, t0, t1):
        """Median gaze while the user was speaking: what "this" / "that" refers to."""
        with self._lock:
            window = [(x, y) for t, x, y in self._history if t0 <= t <= t1]
        if not window:
            return self.point()
        return statistics.median(p[0] for p in window), statistics.median(p[1] for p in window)

    def send(self, cmd: dict):
        self._cmd.sendto(json.dumps(cmd).encode(), ("127.0.0.1", config.CMD_PORT))


class MouseGaze(threading.Thread):
    """Stand-in for eye tracking: the mouse cursor is where the user is "looking".

    Same interface as GazeClient, so the rest of the app can't tell the difference.
    """

    calibratable = False
    busy = False

    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._history = deque(maxlen=900)  # (t, x, y) in screen points, ~30 s at 30 Hz

    def run(self):
        import Quartz  # local: only this stand-in needs it

        while True:
            loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))  # global points, origin top-left
            with self._lock:
                self._history.append((time.time(), loc.x, loc.y))
            time.sleep(1 / 30)

    connected = True

    def point(self):
        with self._lock:
            return self._history[-1][1:] if self._history else None

    def point_during(self, t0, t1):
        with self._lock:
            window = [(x, y) for t, x, y in self._history if t0 <= t <= t1]
        if not window:
            return self.point()
        return statistics.median(p[0] for p in window), statistics.median(p[1] for p in window)

    def send(self, cmd: dict):
        pass  # nothing to calibrate
