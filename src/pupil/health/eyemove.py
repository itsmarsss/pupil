"""Alertness from eye movements: how fast the eyes jump, whether the jumps drag, gaze drift, how often they move.

With fatigue, saccades (the jumps between things you look at) slow down for their size and drag out (take longer
per degree), happen less often, and gaze holds less steadily on what you're looking at. A webcam tracker samples at ~30 Hz with a degree or two of
jitter, so absolute numbers mean little: each measure is compared with the user's own first minutes this session,
and only movements well above the jitter count (saccades over MIN_SACCADE_DEG at over SACCADE_DEG_S).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

SACCADE_DEG_S = 100.0  # frame-to-frame speed that's a jump, not jitter (~1-2 deg of noise at 30 Hz is ~30-60 deg/s)
MIN_SACCADE_DEG = 4.0
MIN_FIXATION_S = 0.4  # fixations at least this long measure drift
WINDOW_S = 120.0  # "now" = the last two minutes
BASELINE_S = 300.0  # the session's first five minutes are its reference
MIN_SACCADES = 8
MIN_FIXATIONS = 8


class EyeMovement:
    def __init__(self):
        self._prev = None  # (t, x_deg, y_deg)
        self._saccade = None  # in progress: (start t, start xy, peak deg/s)
        self.saccades: deque = deque()  # (t, amplitude deg, peak deg/s, duration s)
        self._fixation = None  # (id, start t, [xy...])
        self.fixations: deque = deque()  # (t, rms dispersion deg)
        self._baseline = {"saccades": [], "fixations": [], "t0": None, "face_s": 0.0}
        self._face_s = 0.0

    def update(self, t: float, x_pt: float, y_pt: float, valid: bool, fixation_id, pt_per_deg: float) -> None:
        if not valid or not math.isfinite(x_pt) or pt_per_deg <= 0:
            self._prev, self._saccade = None, None
            return
        x, y = x_pt / pt_per_deg, y_pt / pt_per_deg
        b = self._baseline
        if b["t0"] is None:
            b["t0"] = t
        in_baseline = t - b["t0"] < BASELINE_S
        if self._prev is not None:
            dt = t - self._prev[0]
            if 0 < dt < 0.2:
                self._face_s += dt
                if in_baseline:
                    b["face_s"] += dt
                speed = math.hypot(x - self._prev[1], y - self._prev[2]) / dt
                if speed >= SACCADE_DEG_S:
                    if self._saccade is None:
                        self._saccade = (self._prev[0], (self._prev[1], self._prev[2]), speed)
                    else:
                        self._saccade = (*self._saccade[:2], max(self._saccade[2], speed))
                elif self._saccade is not None:
                    start_t, (sx, sy), peak = self._saccade
                    amplitude = math.hypot(x - sx, y - sy)
                    if amplitude >= MIN_SACCADE_DEG:
                        record = (start_t, amplitude, peak, self._prev[0] - start_t)  # until the last fast frame
                        self.saccades.append(record)
                        if in_baseline:
                            b["saccades"].append(record)
                    self._saccade = None
        self._prev = (t, x, y)
        # fixation drift: how much gaze wanders around its centre while it holds on one thing
        if fixation_id is not None and self._fixation is not None and self._fixation[0] == fixation_id:
            self._fixation[2].append((x, y))
        else:
            if self._fixation is not None and len(self._fixation[2]) > 2 and t - self._fixation[1] >= MIN_FIXATION_S:
                pts = np.array(self._fixation[2])
                rms = float(np.sqrt(np.mean(np.sum((pts - pts.mean(0)) ** 2, axis=1))))
                self.fixations.append((self._fixation[1], rms))
                if in_baseline:
                    b["fixations"].append((self._fixation[1], rms))
            self._fixation = (fixation_id, t, [(x, y)]) if fixation_id is not None else None
        while self.saccades and self.saccades[0][0] < t - WINDOW_S:
            self.saccades.popleft()
        while self.fixations and self.fixations[0][0] < t - WINDOW_S:
            self.fixations.popleft()

    @staticmethod
    def _speed(saccades) -> float | None:
        """Peak speed per degree of jump (the main-sequence slope): lower = slower eyes."""
        return float(np.median([p / a for _, a, p, _ in saccades])) if len(saccades) >= MIN_SACCADES else None

    @staticmethod
    def _drag(saccades) -> float | None:
        """Seconds per degree of jump: higher = the eyes drag."""
        return float(np.median([d / a for _, a, _, d in saccades])) if len(saccades) >= MIN_SACCADES else None

    def snapshot(self, now: float) -> dict:
        """Each measure now / its baseline (1 = same as the start of the session), None until both exist."""
        b = self._baseline
        recent_face = min(self._face_s, WINDOW_S)

        def ratio(current, base):
            return None if current is None or base is None or base == 0 else round(current / base, 2)

        drift_now = float(np.median([r for _, r in self.fixations])) if len(self.fixations) >= MIN_FIXATIONS else None
        drift_base = float(np.median([r for _, r in b["fixations"]])) if len(b["fixations"]) >= MIN_FIXATIONS else None
        rate_now = len(self.saccades) / recent_face * 60 if recent_face >= 30 else None
        rate_base = len(b["saccades"]) / b["face_s"] * 60 if b["face_s"] >= 30 and b["saccades"] else None
        baseline_done = b["t0"] is not None and now - b["t0"] >= BASELINE_S
        return {
            "gaze_drift": ratio(drift_now, drift_base) if baseline_done else None,
            "saccade_speed": ratio(self._speed(self.saccades), self._speed(b["saccades"])) if baseline_done else None,
            "saccade_drag": ratio(self._drag(self.saccades), self._drag(b["saccades"])) if baseline_done else None,
            "saccade_rate": ratio(rate_now, rate_base) if baseline_done else None,
            "saccades_per_min": None if rate_now is None else round(rate_now, 1),
            "drift_deg": None if drift_now is None else round(drift_now, 2),
        }
