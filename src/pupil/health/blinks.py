"""Blinks, from how open the eyes are on every frame (FaceObservation.eye_openness, ~30 fps).

Openness (lid gap / eye width) depends on the person, the camera angle and the light, so a blink is measured against
the user's own open-eye baseline, which adapts: the eyes "close" below CLOSING of it and a blink is a closing that
reopens within MIN_BLINK_S..MAX_BLINK_S. Longer closures aren't blinks (resting the eyes, looking down).
"""
from __future__ import annotations

from dataclasses import dataclass

CLOSING = 0.55  # below this share of the open baseline the eyes count as closing (a blink in progress)
MOSTLY_CLOSED = 0.2  # below this share they're "mostly closed" (>= 80% closed: the PERCLOS P80 criterion)
MIN_BLINK_S = 0.05
MAX_BLINK_S = 0.5
BASELINE_RATE = 0.01  # per open frame (~3 s to adapt at 30 fps)
WARMUP_FRAMES = 15  # frames before the baseline is trusted


@dataclass(frozen=True)
class Blink:
    t: float  # when the eyes closed (seconds, tracker clock)
    duration: float  # seconds closed
    depth: float  # lowest openness as a share of the baseline (0: fully closed)


class BlinkDetector:
    def __init__(self):
        self.baseline: float | None = None  # open-eye openness, adapting
        self._frames = 0
        self._closed_at: float | None = None
        self._min_ratio = 1.0

    def ratio(self, openness: float) -> float | None:
        """Openness as a share of this person's open-eye baseline (None until warmed up)."""
        if self.baseline is None or self._frames < WARMUP_FRAMES:
            return None
        return openness / self.baseline

    def update(self, t: float, openness: float) -> Blink | None:
        """One frame with the eyes visible. Returns a Blink when one has just ended."""
        self._frames += 1
        if self.baseline is None:
            self.baseline = openness
        ratio = openness / self.baseline
        if self._closed_at is None:
            if ratio < CLOSING and self._frames >= WARMUP_FRAMES:
                self._closed_at, self._min_ratio = t, ratio
            else:  # open: let the baseline follow slow changes (light, posture), never a blink
                self.baseline += BASELINE_RATE * (openness - self.baseline)
            return None
        self._min_ratio = min(self._min_ratio, ratio)
        if ratio < CLOSING:
            return None
        duration, closed_at = t - self._closed_at, self._closed_at
        self._closed_at = None
        if MIN_BLINK_S <= duration <= MAX_BLINK_S:
            return Blink(closed_at, duration, max(self._min_ratio, 0.0))
        return None

    def lost(self) -> None:
        """The face went out of view: a closing in progress can't be judged."""
        self._closed_at = None
