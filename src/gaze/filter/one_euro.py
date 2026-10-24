"""One Euro filter (Casiez et al., CHI 2012): smooth at rest, responsive when moving."""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filters scalars or arrays; timestamps in nanoseconds."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t_ns: int | None = None

    def __call__(self, x, t_ns: int):
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t_ns is None or t_ns <= self._t_ns:
            self._x, self._dx, self._t_ns = x, np.zeros_like(x), t_ns
            return x.copy()
        dt = (t_ns - self._t_ns) / 1e9
        if dt > 1.0:  # long gap: start over rather than smear across it
            self._x, self._dx, self._t_ns = x, np.zeros_like(x), t_ns
            return x.copy()
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        tau = 1.0 / (2 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self._x = a * x + (1 - a) * self._x
        self._t_ns = t_ns
        return self._x.copy()
