"""Where voice control gets gaze from: the tracker in the same process, or the mouse (for development).

Both offer the same interface: the latest point, the median point and confidence while the user spoke,
whether calibration screens are up, tracker commands, and reporting to the dashboard.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from typing import Callable

from ..core.types import Sample
from . import config


class TrackerGaze:
    """Reads gaze straight from the tracker (no network): attach `on_sample` as a tracker listener."""

    calibratable = True

    def __init__(self, busy: Callable[[], bool], commands: dict[str, Callable[[], None]],
                 report: Callable[[dict], None] = lambda msg: None):
        self._busy, self._commands, self._report = busy, commands, report
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=900)  # (wall t, x, y, confidence), ~30 s at 30 Hz
        self.lips = None  # a lips.LipGate fed with every frame
        self._last_seen = self._last_point = 0.0
        self._offset = time.time() - time.monotonic_ns() / 1e9  # speech is timestamped with time.time()

    def on_sample(self, s: Sample) -> None:
        now = time.time()
        self._last_seen = now
        p = s.point
        if self.lips is not None:  # same fields the tracker used to publish to the controller
            msg = {"state": p.state.value, "confidence": float(p.confidence), "busy": self.busy}
            if s.face is not None:
                msg["mouth"] = s.face.mouth_open
            self.lips.update(msg, now)
        if p.valid:
            with self._lock:
                self._history.append((self._offset + s.t_ns / 1e9, p.x, p.y, float(p.confidence)))
            self._last_point = now

    @property
    def busy(self) -> bool:
        return self._busy()

    @property
    def connected(self) -> bool:
        return time.time() - self._last_seen < config.GAZE_STALE_S

    def point(self):
        """Latest gaze point in screen points, or None if the tracker has lost you."""
        with self._lock:
            if not self._history or self.busy or time.time() - self._last_point > config.GAZE_STALE_S:
                return None
            return self._history[-1][1:3]

    def point_during(self, t0, t1):
        """Median gaze while the user was speaking: what "this" / "that" refers to."""
        with self._lock:
            window = [(x, y) for t, x, y, _ in self._history if t0 <= t <= t1]
        if not window:
            return self.point()
        return statistics.median(p[0] for p in window), statistics.median(p[1] for p in window)

    def confidence_during(self, t0, t1) -> float:
        """Median tracker confidence while speaking (lower = head turned past what calibration covered)."""
        with self._lock:
            window = [c for t, _, _, c in self._history if t0 <= t <= t1] or [c for *_, c in list(self._history)[-5:]]
        return statistics.median(window) if window else 0.0

    def send(self, cmd: dict) -> None:
        fn = self._commands.get(cmd.get("cmd"))
        if fn is not None:
            fn()

    def report(self, event: str, text: str = "", **extra) -> None:
        self._report({"event": event, "text": text, **extra})


class MouseGaze(threading.Thread):
    """Stand-in for eye tracking: the mouse cursor is where the user is "looking"."""

    calibratable = False
    busy = False
    connected = True

    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=900)  # (t, x, y)

    def run(self):
        import Quartz  # local: only this stand-in needs it

        while True:
            loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))  # global points, origin top-left
            with self._lock:
                self._history.append((time.time(), loc.x, loc.y))
            time.sleep(1 / 30)

    def point(self):
        with self._lock:
            return self._history[-1][1:] if self._history else None

    def point_during(self, t0, t1):
        with self._lock:
            window = [(x, y) for t, x, y in self._history if t0 <= t <= t1]
        if not window:
            return self.point()
        return statistics.median(p[0] for p in window), statistics.median(p[1] for p in window)

    def confidence_during(self, t0, t1) -> float:
        return 1.0

    def send(self, cmd: dict) -> None:
        pass  # nothing to calibrate

    def report(self, event: str, text: str = "", **extra) -> None:
        pass
