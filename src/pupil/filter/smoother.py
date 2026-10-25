"""Gaze smoothing for display: a One Euro filter plus an optional fixation hold.

Works in degrees of visual angle so the settings mean the same thing at any viewing
distance. The fixation hold keeps the output at the running mean of the current
fixation and only moves once several consecutive samples land clearly outside it,
which removes most visible wobble without slowing real eye movements much.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .one_euro import OneEuroFilter


@dataclass(frozen=True)
class SmoothingSettings:
    min_cutoff: float = 1.0  # Hz; lower = smoother at rest
    beta: float = 0.3  # per °/s; higher = less lag when moving
    hold_radius_deg: float | None = 2.5  # None disables the fixation hold
    hold_exit_frames: int = 2  # consecutive samples outside the radius to break the hold


# Tuned by replaying the validation streams of four live sessions:
# medium cut ring jitter 0.55° → 0.12° vs the previous filter, and the slow 10 % of
# arrivals got faster (983 → 748 ms); the median arrival stayed ~0.35–0.4 s.
PRESETS = {
    "off": None,
    "light": SmoothingSettings(hold_radius_deg=None),
    "medium": SmoothingSettings(),
    "strong": SmoothingSettings(min_cutoff=0.3, hold_radius_deg=3.0),
}


class GazeSmoother:
    def __init__(self, settings: SmoothingSettings | None):
        self.settings = settings
        self._euro = OneEuroFilter(settings.min_cutoff, settings.beta) if settings else None
        self.reset()

    def reset(self) -> None:
        if self._euro is not None:
            self._euro.reset()
        self._hold: np.ndarray | None = None  # fixation centre, degrees
        self._hold_n = 0
        self._outside: list[np.ndarray] = []

    def __call__(self, x_pt: float, y_pt: float, t_ns: int, pt_per_deg: float) -> tuple[float, float]:
        if self.settings is None:
            return x_pt, y_pt
        p = np.array([x_pt, y_pt]) / pt_per_deg
        r = self.settings.hold_radius_deg
        if r is not None:
            # Hold decisions use raw samples, so one stray frame can't break a fixation;
            # the One Euro filter then smooths the held point (and the jump when it moves).
            p = self._held(p, r)
        return tuple(self._euro(p, t_ns) * pt_per_deg)

    def _held(self, p: np.ndarray, r: float) -> np.ndarray:
        if self._hold is None:
            self._hold, self._hold_n, self._outside = p.copy(), 1, []
        elif np.linalg.norm(p - self._hold) <= r:
            self._outside = []
            self._hold_n += 1
            self._hold += (p - self._hold) / min(self._hold_n, 30)  # running mean, slowly adapting
        else:
            self._outside.append(p)
            if len(self._outside) >= self.settings.hold_exit_frames:
                # left the fixation: restart from where gaze went
                self._hold = np.mean(self._outside, axis=0)
                self._hold_n = len(self._outside)
                self._outside = []
        return self._hold.copy()
