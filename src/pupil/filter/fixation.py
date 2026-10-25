"""Velocity-threshold (I-VT) fixation detection on screen points."""
from __future__ import annotations

import math


class FixationDetector:
    """Assigns a fixation id while gaze velocity stays below `threshold_deg_s`.

    `pt_per_deg` converts screen points to degrees; update it as the viewing
    distance changes.
    """

    def __init__(self, threshold_deg_s: float = 60.0, min_duration_ms: float = 80.0):
        self.threshold_deg_s = threshold_deg_s
        self.min_duration_ms = min_duration_ms
        self._next_id = 0
        self._current: int | None = None
        self._start_ns: int | None = None
        self._last: tuple[float, float, int] | None = None

    def reset(self) -> None:
        self._current = None
        self._start_ns = None
        self._last = None

    def update(self, x: float, y: float, t_ns: int, pt_per_deg: float) -> int | None:
        """Returns the current fixation id, or None during saccades and before
        a fixation has lasted `min_duration_ms`."""
        last, self._last = self._last, (x, y, t_ns)
        if last is None or t_ns <= last[2]:
            self._start_ns = t_ns
            return None
        dt = (t_ns - last[2]) / 1e9
        velocity = math.hypot(x - last[0], y - last[1]) / pt_per_deg / dt
        if velocity > self.threshold_deg_s or dt > 0.25:
            self._current = None
            self._start_ns = t_ns
            return None
        if self._current is None:
            if self._start_ns is None:
                self._start_ns = last[2]
            if (t_ns - self._start_ns) / 1e6 < self.min_duration_ms:
                return None
            self._current = self._next_id
            self._next_id += 1
        return self._current
