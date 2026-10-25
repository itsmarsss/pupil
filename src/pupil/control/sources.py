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
from .heat import FINAL_FIXATION_BOOST, GazeCloud
from .lips import LipGate

MIN_FINAL_FIXATION_SAMPLES = 3


class TrackerGaze:
    """Reads gaze straight from the tracker (no network): attach `on_sample` as a tracker listener."""

    calibratable = True

    def __init__(self, busy: Callable[[], bool], commands: dict[str, Callable[[], None]],
                 report: Callable[[dict], None] = lambda msg: None, sigma: Callable[[], float] = lambda: 60.0):
        """sigma(): the tracker's current accuracy as a Gaussian width in screen points (heat.GazeCloud)."""
        self._busy, self._commands, self._report, self.sigma = busy, commands, report, sigma
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=900)  # (wall t, x, y, confidence, fixation id), ~30 s at 30 Hz
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
                self._history.append((self._offset + s.t_ns / 1e9, p.x, p.y, float(p.confidence), p.fixation_id))
            self._last_point = now

    def frame_time(self, t: float) -> float:
        """A time.time() timestamp on the tracker's clock (monotonic seconds, as the dashboard plots use)."""
        return t - self._offset

    @property
    def busy(self) -> bool:
        return self._busy()

    @property
    def connected(self) -> bool:
        return time.time() - self._last_seen < config.TRACKER_STALE_S

    def point(self):
        """Latest gaze point in screen points, or None if the tracker has lost you."""
        with self._lock:
            if not self._history or self.busy or time.time() - self._last_point > config.TRACKER_STALE_S:
                return None
            return self._history[-1][1:3]

    def point_during(self, t0, t1):
        """Median gaze while the user was speaking: what "this" / "that" refers to."""
        with self._lock:
            window = [(x, y) for t, x, y, _, _ in self._history if t0 <= t <= t1]
        if not window:
            return self.point()
        return statistics.median(p[0] for p in window), statistics.median(p[1] for p in window)

    def cloud_during(self, t0, t1) -> GazeCloud:
        """Every gaze sample while the user spoke, as a heat map (heat.GazeCloud): what "this" could be."""
        with self._lock:
            samples = [(x, y, c, fixation) for t, x, y, c, fixation in self._history if t0 <= t <= t1] or \
                [(x, y, c, fixation) for _, x, y, c, fixation in list(self._history)[-5:]]
        points = [(x, y, c) for x, y, c, _ in samples]
        weights = [1.0] * len(samples)
        final_fixation = next((fixation for *_, fixation in reversed(samples) if fixation is not None), None)
        if final_fixation is not None:
            indices = [i for i, (*_, fixation) in enumerate(samples) if fixation == final_fixation]
            if len(indices) >= MIN_FINAL_FIXATION_SAMPLES:
                for i in indices:
                    weights[i] = FINAL_FIXATION_BOOST
        return GazeCloud(points, self.sigma(), weights)

    def confidence_during(self, t0, t1) -> float:
        """Median tracker confidence while speaking (lower = head turned past what calibration covered)."""
        with self._lock:
            window = [c for t, _, _, c, _ in self._history if t0 <= t <= t1] or \
                [c for _, _, _, c, _ in list(self._history)[-5:]]
        return statistics.median(window) if window else 0.0

    def send(self, cmd: dict) -> None:
        fn = self._commands.get(cmd.get("cmd"))
        if fn is not None:
            fn()

    def report(self, event: str, text: str = "", **extra) -> None:
        self._report({"event": event, "text": text, **extra})


def tracker_lips_listener(lips: LipGate | None, busy: Callable[[], bool]) -> Callable[[Sample], None]:
    """Feed the lip gate from tracker samples (mouse hybrid: camera for voice, cursor for gaze)."""

    def on_sample(sample: Sample) -> None:
        if lips is None:
            return
        p = sample.point
        msg = {"state": p.state.value, "confidence": float(p.confidence), "busy": busy()}
        if sample.face is not None:
            msg["mouth"] = sample.face.mouth_open
        lips.update(msg, time.time())

    return on_sample


class MouseGaze(threading.Thread):
    """Stand-in for eye tracking: the mouse cursor is where the user is "looking"."""

    calibratable = False
    busy = False
    connected = True

    def __init__(self, report: Callable[[dict], None] | None = None, *, lip_gate: bool = True):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=900)  # (t, x, y)
        self._report = report or (lambda _msg: None)
        self._running = True
        self.lip_gate = lip_gate  # webcam feeds LipGate; cursor is still gaze
        self.lips = None

    def stop(self) -> None:
        self._running = False

    def run(self):
        import Quartz  # local: only this stand-in needs it

        while self._running:
            loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))  # global points, origin top-left
            with self._lock:
                self._history.append((time.time(), loc.x, loc.y))
            time.sleep(1 / 30)

    def frame_time(self, t: float) -> float:
        return t  # voice timestamps are already wall clock

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

    def cloud_during(self, t0, t1) -> GazeCloud:
        """The pointer is exact: a tight cloud."""
        with self._lock:
            points = [(x, y, 1.0) for t, x, y in self._history if t0 <= t <= t1] or \
                [(x, y, 1.0) for _, x, y in list(self._history)[-1:]]
        return GazeCloud(points, 4.0)

    def send(self, cmd: dict) -> None:
        pass  # nothing to calibrate

    def report(self, event: str, text: str = "", **extra) -> None:
        self._report({"event": event, "text": text, **extra})
