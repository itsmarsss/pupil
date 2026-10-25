"""Eye health while you work: blink rate, signs of eye fatigue, screen time and breaks, with gentle reminders.

Fed every tracker sample (cheap: a few comparisons per frame). The measures:
- blink rate per minute of face time (relaxed ~15-20; while staring at screens it often drops to ~5-7, drying the eyes)
- blink duration (blinks get slower with tiredness)
- PERCLOS: share of the last minute with the eyes mostly (>= 80%) closed, the standard drowsiness measure
- lid droop: how open the eyes are now against the start of the session
- screen time since the last break (face out of view or looking off-screen for BREAK_S: the 20-20-20 rule)
The fatigue score combines them. These are heuristics for nudges, not a medical measurement.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import date

import numpy as np

from ..core.types import Sample, TrackingState
from .blinks import MOSTLY_CLOSED, BlinkDetector

BREAK_S = 20.0  # away this long counts as a break (20-20-20)
RATE_WINDOW_S = 60.0
PERCLOS_WINDOW_S = 60.0
MIN_FACE_S = 30.0  # face time needed before blink rate means anything
SESSION_BASELINE_S = 180.0  # the first minutes of openness are the session's reference for droop
LOG_EVERY_S = 60.0
ON_SCREEN = {TrackingState.TRACKING, TrackingState.BLINK, TrackingState.UNCALIBRATED}

# reminders: (kind, condition text, cooldown)
TWENTY_MIN = 20 * 60
BLINK_LOW = 8.0  # per minute, sustained
BLINK_REMIND_COOLDOWN_S = 10 * 60
FATIGUE_REMIND_COOLDOWN_S = 30 * 60


def _clamp(x):
    return float(min(max(x, 0.0), 1.0))


class HealthTracker:
    def __init__(self, log_dir=None):
        self._lock = threading.Lock()
        self.blinks = BlinkDetector()
        self._blinks: deque = deque()  # Blink, last few minutes
        self._face: deque = deque()  # (t, frame duration) with the face visible, for face-time normalisation
        self._closed: deque = deque()  # (t, mostly closed?) for PERCLOS
        self._openness: deque = deque()  # (t, ratio to baseline), last minute
        self._session_open: list[float] = []  # ratios from the first minutes
        self._last_t: float | None = None
        self._screen_since: float | None = None  # start of the current stretch on screen
        self._away_since: float | None = None
        self.breaks = 0
        self.started = time.time()
        self._reminded: dict[str, float] = {}
        self.reminders_on = True
        self._log_dir = log_dir
        self._last_log = time.time()

    # ---- per frame (tracker thread) -------------------------------------------------------------

    def on_sample(self, s: Sample) -> None:
        t = s.t_ns / 1e9
        with self._lock:
            dt = min(t - self._last_t, 0.2) if self._last_t is not None else 0.0
            self._last_t = t
            face = s.face
            on_screen = face is not None and s.point.state in ON_SCREEN
            if on_screen:
                if self._away_since is not None and t - self._away_since >= BREAK_S:
                    self.breaks += 1
                    self._screen_since = t
                if self._screen_since is None:
                    self._screen_since = t
                self._away_since = None
            elif self._away_since is None:
                self._away_since = t
            if face is None:
                self.blinks.lost()
                return
            openness = float(np.mean(face.eye_openness))
            blink = self.blinks.update(t, openness)
            if blink is not None:
                self._blinks.append(blink)
            ratio = self.blinks.ratio(openness)
            self._face.append((t, dt))
            if ratio is not None:
                self._closed.append((t, ratio < MOSTLY_CLOSED))
                self._openness.append((t, ratio))
                if t - self._face[0][0] < SESSION_BASELINE_S and len(self._session_open) < 6000:
                    self._session_open.append(ratio)
            horizon = t - max(RATE_WINDOW_S * 5, PERCLOS_WINDOW_S)
            for q in (self._blinks,):
                while q and q[0].t < horizon:
                    q.popleft()
            for q in (self._face, self._closed, self._openness):
                while q and q[0][0] < horizon:
                    q.popleft()

    # ---- summaries (any thread) -------------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            now = self._last_t or 0.0
            face_s = sum(d for t, d in self._face if t >= now - RATE_WINDOW_S)
            recent = [b for b in self._blinks if b.t >= now - RATE_WINDOW_S]
            rate = len(recent) / face_s * 60 if face_s >= MIN_FACE_S else None
            longer = [b.duration for b in self._blinks if b.t >= now - 2 * RATE_WINDOW_S]
            duration = float(np.mean(longer)) if len(longer) >= 3 else None
            closed = [c for t, c in self._closed if t >= now - PERCLOS_WINDOW_S]
            perclos = sum(closed) / len(closed) if len(closed) >= 30 * 10 else None
            open_now = [r for t, r in self._openness if t >= now - RATE_WINDOW_S]
            droop = None
            if len(self._session_open) > 300 and len(open_now) > 100:
                droop = float(np.median(open_now) / np.median(self._session_open))
            on_screen_min = (now - self._screen_since) / 60 if self._screen_since is not None and self._away_since is None else 0.0
            breaks = self.breaks
        parts = {  # each 0 (fine) .. 1 (strong sign of tiredness), and its weight
            "blink_rate": (_clamp((12 - rate) / 8) if rate is not None else None, 0.25),
            "blink_duration": (_clamp((duration - 0.25) / 0.25) if duration is not None else None, 0.15),
            "perclos": (_clamp((perclos - 0.05) / 0.15) if perclos is not None else None, 0.25),
            "droop": (_clamp((0.95 - droop) / 0.2) if droop is not None else None, 0.15),
            "screen_time": (_clamp((on_screen_min - 20) / 40), 0.2),
        }
        known = [(v, w) for v, w in parts.values() if v is not None]
        score = round(100 * sum(v * w for v, w in known) / sum(w for _, w in known)) if known else None
        level = None if score is None else "low" if score < 35 else "moderate" if score < 65 else "high"
        return {
            "blink_rate": None if rate is None else round(rate, 1),
            "blink_duration_ms": None if duration is None else round(duration * 1000),
            "perclos": None if perclos is None else round(perclos, 3),
            "openness_vs_start": None if droop is None else round(droop, 2),
            "minutes_on_screen": round(on_screen_min, 1),
            "breaks": breaks,
            "fatigue": score,
            "level": level,
        }

    def status_line(self) -> str:
        h = self.snapshot()
        parts = []
        if h["blink_rate"] is not None:
            parts.append(f"{h['blink_rate']:.0f} blinks/min")
        if h["level"]:
            parts.append(f"fatigue {h['level']}")
        parts.append(f"{h['minutes_on_screen']:.0f} min since a break")
        return " · ".join(parts)

    # ---- reminders (main thread, ~1 Hz) --------------------------------------------------------------

    def due_reminders(self) -> list[str]:
        """Gentle nudges that are due now, each rate-limited."""
        if not self.reminders_on:
            return []
        h, now, out = self.snapshot(), time.time(), []

        def due(kind, cooldown):
            if now - self._reminded.get(kind, 0) >= cooldown:
                self._reminded[kind] = now
                return True
            return False

        if h["minutes_on_screen"] * 60 >= TWENTY_MIN and due("20-20-20", TWENTY_MIN):
            out.append("20 min on screen: look ~6 m away for 20 s.")
        if h["blink_rate"] is not None and h["blink_rate"] < BLINK_LOW and due("blink", BLINK_REMIND_COOLDOWN_S):
            out.append(f"Only {h['blink_rate']:.0f} blinks a minute: blink slowly a few times.")
        if h["level"] == "high" and due("fatigue", FATIGUE_REMIND_COOLDOWN_S):
            out.append("Your eyes look tired: a 5-minute break would help.")
        self._log(h)
        return out

    def _log(self, h: dict) -> None:
        """A line a minute in the day's health log (local only)."""
        if self._log_dir is None or time.time() - self._last_log < LOG_EVERY_S:
            return
        self._last_log = time.time()
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            with open(self._log_dir / f"{date.today().isoformat()}.jsonl", "a") as f:
                f.write(json.dumps({"t": round(time.time()), **h}) + "\n")
        except OSError:
            pass
