"""Target layouts and the sample window logic for calibration and validation dots."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def grid(n: int, width: float, height: float, margin: float = 0.1) -> list[tuple[float, float]]:
    """5 (centre + corners), 9 (3x3), 13 (3x3 + inner diagonals) or 16 (4x4) targets in
    screen points. The first point is always the centre (or nearest to it), to orient."""
    xs = [margin, 0.5, 1 - margin]
    ys = [margin, 0.5, 1 - margin]
    if n == 5:
        pts = [(0.5, 0.5), (margin, margin), (1 - margin, margin), (margin, 1 - margin), (1 - margin, 1 - margin)]
    elif n == 9:
        pts = [(x, y) for y in ys for x in xs]
    elif n == 13:
        q = [0.3, 0.7]
        pts = [(x, y) for y in ys for x in xs] + [(x, y) for y in q for x in q]
    elif n == 16:
        g = np.linspace(margin, 1 - margin, 4)
        pts = [(float(x), float(y)) for y in g for x in g]
    elif n == 3:  # quick launch check
        pts = [(0.5, 0.5), (0.2, 0.25), (0.8, 0.75)]
    else:
        raise ValueError(f"unsupported target count {n}")
    first = min(range(len(pts)), key=lambda i: (pts[i][0] - 0.5) ** 2 + (pts[i][1] - 0.5) ** 2)
    pts = [pts[first]] + pts[:first] + pts[first + 1 :]
    return [(x * width, y * height) for x, y in pts]


def shuffled(points, seed: int | None = None):
    """Centre first (to orient), the rest in random order to avoid anticipatory looks."""
    rng = np.random.default_rng(seed)
    rest = list(points[1:])
    rng.shuffle(rest)
    return [points[0], *rest]


@dataclass(frozen=True)
class TargetTiming:
    dwell_s: float = 1.3  # how long each dot is shown
    settle_s: float = 0.45  # ignore samples right after onset (saccade + latency)
    tail_s: float = 0.05  # ignore the last moment as the dot leaves

    def accepts(self, onset_ns: int, t_ns: int) -> bool:
        dt = (t_ns - onset_ns) / 1e9
        return self.settle_s <= dt <= self.dwell_s - self.tail_s
