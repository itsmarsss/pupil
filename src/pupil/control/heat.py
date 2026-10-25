"""Where the user is looking, as a heat map rather than a point.

Webcam gaze is only good to a few centimetres (a few degrees), about the width of a browser tab, so "that tab" can't
be read off a single point. A gaze *cloud* is every gaze sample while the user spoke, each a 2D Gaussian whose
width is the tracker's currently measured accuracy (wider when a sample's confidence is low). The heat an element
gets is how much of the cloud falls inside its rectangle, computed exactly (the Gaussian integral over a box
separates into erf terms). The target is the element with the most heat x the best fit with what was said; when
the top two are close, the controller asks instead of guessing. The same cloud is what the ring and dashboard draw.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

MIN_SIGMA_PT = 20.0
MAX_SIGMA_PT = 400.0
DEFAULT_SIGMA_DEG = 3.0  # until a validation has measured it
AMBIGUOUS_TOP = 0.65  # ask when the best candidate has less than this share...
AMBIGUOUS_RATIO = 0.5  # ...and the runner-up has at least this fraction of its share
FINAL_FIXATION_BOOST = 2.0


def sigma_from_accuracy(accuracy_pt: float) -> float:
    """Per-axis Gaussian sigma from a mean radial error (Rayleigh: mean = sigma * sqrt(pi / 2))."""
    return min(max(accuracy_pt / math.sqrt(math.pi / 2), MIN_SIGMA_PT), MAX_SIGMA_PT)


@dataclass
class GazeCloud:
    points: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, confidence), screen points
    sigma: float = 60.0  # points, for a fully confident sample
    weights: list[float] = field(default_factory=list)  # optional per-sample attention weight

    def __bool__(self):
        return bool(self.points)

    def _sigma(self, confidence: float) -> float:
        return self.sigma / math.sqrt(max(confidence, 0.25))  # less sure: wider

    def _weight(self, index: int, confidence: float) -> float:
        attention = self.weights[index] if index < len(self.weights) else 1.0
        return max(confidence, 0.25) * max(attention, 0.0)

    @property
    def centre(self) -> tuple[float, float] | None:
        if not self.points:
            return None
        w = [self._weight(i, c) for i, (*_, c) in enumerate(self.points)]
        return (sum(x * k for (x, _, _), k in zip(self.points, w)) / sum(w),
                sum(y * k for (_, y, _), k in zip(self.points, w)) / sum(w))

    def mass(self, x: float, y: float, w: float, h: float) -> float:
        """Share of the cloud inside the rectangle (0..1)."""
        if not self.points:
            return 0.0
        total = weight = 0.0
        for i, (px, py, c) in enumerate(self.points):
            s = self._sigma(c) * math.sqrt(2)
            k = self._weight(i, c)
            fx = 0.5 * (math.erf((x + w - px) / s) - math.erf((x - px) / s))
            fy = 0.5 * (math.erf((y + h - py) / s) - math.erf((y - py) / s))
            total += k * fx * fy
            weight += k
        return total / weight

    def density(self, x: float, y: float) -> float:
        """Heat at a point (unnormalised), for drawing."""
        heat = 0.0
        for i, (px, py, c) in enumerate(self.points):
            s = self._sigma(c)
            heat += self._weight(i, c) * math.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * s * s)) / (s * s)
        return heat

    def summary(self) -> dict:
        start = max(0, len(self.points) - 40)
        out = {"sigma": round(self.sigma, 1),
               "points": [(round(x), round(y), round(c, 2)) for x, y, c in self.points[start:]]}
        if self.weights and any(abs(w - 1.0) > 1e-9 for w in self.weights):
            out["weights"] = [round(w, 2) for w in self.weights[start:]]
        return out


def scores(elements, cloud: GazeCloud, fit=None) -> list[float]:
    """Each element's share of the belief that it's the target: its heat x how well it fits what was said
    (`fit`, e.g. Jev's per-element probabilities; equal if not given), normalised to sum to 1."""
    if not elements:
        return []
    fit = fit or [1.0] * len(elements)
    raw = [(cloud.mass(e.x, e.y, e.w, e.h) + 1e-4) * max(f, 1e-3) for e, f in zip(elements, fit)]
    total = sum(raw)
    return [r / total for r in raw]


def ambiguous(shares: list[float]) -> bool:
    """Too close to call: the best has under AMBIGUOUS_TOP and the runner-up is within AMBIGUOUS_RATIO of it."""
    if len(shares) < 2:
        return False
    top, second = sorted(shares, reverse=True)[:2]
    return top < AMBIGUOUS_TOP and second >= AMBIGUOUS_RATIO * top


def neighbour(target, candidates, direction: str):
    """The element next to `target` in `direction` (left / right / up / down) among its siblings: same kind of
    control in the same row (left/right) or column (up/down), nearest first. None if there isn't one."""
    tx, ty = target.center
    same = [c for c in candidates if c is not target and c.role == target.role
            and (c.parent_role == target.parent_role)]
    if direction in ("left", "right"):
        row = [c for c in same if abs(c.center[1] - ty) < max(target.h, c.h) / 2]
        side = [c for c in row if (c.center[0] < tx if direction == "left" else c.center[0] > tx)]
        return min(side, key=lambda c: abs(c.center[0] - tx), default=None)
    column = [c for c in same if abs(c.center[0] - tx) < max(target.w, c.w) / 2]
    side = [c for c in column if (c.center[1] < ty if direction == "up" else c.center[1] > ty)]
    return min(side, key=lambda c: abs(c.center[1] - ty), default=None)
