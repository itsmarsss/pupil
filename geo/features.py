"""Landmark features used to make the geometric gaze more robust: eyelid-proof iris centres, eye openness for
blink rejection, rigid-landmark head pose, and a One Euro filter for display smoothing."""
from __future__ import annotations

import math

import numpy as np

# Iris ring points (4 per iris, around the edge) for the subject's right / left eye
R_IRIS_RING = (469, 470, 471, 472)
L_IRIS_RING = (474, 475, 476, 477)

# Eye-aspect-ratio landmarks: corner, top, top, corner, bottom, bottom (Soukupová & Čech style)
R_EAR = (33, 160, 158, 133, 153, 144)
L_EAR = (362, 385, 387, 263, 373, 380)

# Landmarks that don't move with gaze or expression: nose bridge/tip, glabella/forehead, cheekbones, face sides.
# No eyelids, brows, mouth or jaw.
RIGID = (1, 4, 5, 6, 8, 9, 10, 151, 168, 195, 197, 45, 275, 116, 345, 123, 352, 234, 454, 93, 323, 127, 356)


def iris_center_equator(lms: np.ndarray, ring: tuple[int, ...]) -> np.ndarray:
    """Iris centre from the two ring points on its horizontal midline. The eyelids cover the top/bottom of the
    iris and drag MediaPipe's centre point with them; the left/right edge points are rarely occluded."""
    p = lms[list(ring), :2]
    # the horizontal pair is whichever opposite pair is further apart in x
    pairs = ((0, 2), (1, 3))
    a, b = max(pairs, key=lambda ij: abs(p[ij[0], 0] - p[ij[1], 0]))
    return (p[a] + p[b]) / 2


def eye_aspect_ratio(lms: np.ndarray, idx: tuple[int, ...], aspect: float) -> float:
    """Eye openness. `aspect` = image width / height, since landmarks are normalised per axis."""
    p = lms[list(idx), :2] * [aspect, 1.0]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float(v / (2 * h)) if h > 0 else 0.0


def ear(lms: np.ndarray, aspect: float) -> float:
    return (eye_aspect_ratio(lms, R_EAR, aspect) + eye_aspect_ratio(lms, L_EAR, aspect)) / 2


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising ||R @ src_i + t - dst_i||."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cd - R @ cs


class OneEuro:
    """One Euro filter (Casiez et al., CHI 2012): smooths jitter when still, follows quickly when moving."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x
