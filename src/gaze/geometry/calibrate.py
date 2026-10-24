"""Per-person calibration of the geometric model (ported from geo/calib.py, default config).

  1. Physical fit: angle = gain * measured + kappa, for yaw and pitch, shared by both eyes. Fitted on per-dot
     mean measurements (single noisy frames bias the gains low) with Huber-style reweighting of outlier dots.
  2. Quadratic screen-space residual correction (what helped most on real runs).
  3. Head compensation: linear correction for head movement away from the calibration pose, learned from the
     head-movement step (dot calibration alone can't learn it).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eyeball import EyeMeasure, ScreenPlane, hits

MISS_CM = 50.0
RESIDUAL_RIDGE = 0.02  # x number of dots
RESIDUAL_MIN_DOTS = 8
HEAD_COMP_RIDGE = 0.5  # x number of frames, standardised features
HEAD_SCALE_FLOOR = np.array([0.5, 0.5, 0.2, 0.2])  # degrees, degrees, cm, cm


def _head_features(m: EyeMeasure) -> np.ndarray:
    return np.array([m.head_ypr[0], m.head_ypr[1], m.nose[0], m.nose[1]])


def _quad(xy: np.ndarray, centre, half) -> np.ndarray:
    x = (xy[..., 0] - centre[0]) / half[0]
    y = (xy[..., 1] - centre[1]) / half[1]
    return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], -1)


def _mean_measure(ms: list[EyeMeasure]) -> EyeMeasure:
    B = np.mean([m.basis for m in ms], axis=0)
    U, _, Vt = np.linalg.svd(B)
    return EyeMeasure(np.mean([m.centres for m in ms], 0), np.mean([m.angles for m in ms], 0), U @ Vt,
                      np.mean([m.head_ypr for m in ms], 0), np.mean([m.nose for m in ms], 0))


def _lm(res_fn, p0: np.ndarray, iters: int = 60, weights=None) -> np.ndarray:
    """Levenberg–Marquardt with a forward-difference Jacobian."""
    w = 1.0 if weights is None else np.sqrt(weights)
    f = lambda p: res_fn(p) * w  # noqa: E731
    p, r = p0.copy(), f(p0)
    cost, lam = r @ r, 1e-2
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for j in range(p.size):
            dp = np.zeros(p.size)
            dp[j] = 1e-3
            J[:, j] = (f(p + dp) - r) / 1e-3
        A, b = J.T @ J, J.T @ r
        improved = False
        while lam <= 1e8:
            delta = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), b)
            r_new = f(p + delta)
            if r_new @ r_new < cost:
                p, r, cost, lam, improved = p + delta, r_new, r_new @ r_new, lam / 3, True
                break
            lam *= 4
        if not improved or np.abs(delta).max() < 1e-7:
            break
    return p


@dataclass
class GeoCalibration:
    plane: ScreenPlane
    gain: np.ndarray  # (2,) yaw, pitch
    kappa: np.ndarray  # (2,) degrees
    residual: tuple | None = None  # (centre, half, coef 6x2)
    head_comp: tuple | None = None  # (ref, scale, coef 4x2)

    def hit_cm(self, m: EyeMeasure, head_comp: bool = True) -> np.ndarray | None:
        h = hits(m, self.gain, self.kappa)
        if h is None:
            return None
        if self.residual is not None:
            centre, half, coef = self.residual
            h = h + _quad(h, centre, half) @ coef
        if head_comp and self.head_comp is not None:
            ref, scale, coef = self.head_comp
            h = h + ((_head_features(m) - ref) / scale) @ coef
        return h

    def predict(self, m: EyeMeasure, head_comp: bool = True) -> tuple[float, float] | None:
        h = self.hit_cm(m, head_comp)
        if h is None:
            return None
        x, y = self.plane.to_pt(h)
        return float(x), float(y)


def fit(dots: list[tuple[list[EyeMeasure], tuple[float, float]]], plane: ScreenPlane,
        sweep: list[tuple[list[EyeMeasure], tuple[float, float]]] | None = None) -> GeoCalibration:
    """dots: per calibration dot, its frames' measurements and the dot (screen pt). sweep: same for the
    head-movement step (eyes on one dot, head moving)."""
    groups = [(_mean_measure(ms), plane.to_cm(*target)) for ms, target in dots if ms]

    def residuals(p):
        out = []
        for m, tgt in groups:
            h = hits(m, np.exp(p[2:]), p[:2])
            out.extend((h - tgt) if h is not None else (MISS_CM, MISS_CM))
        return np.array(out)

    p = _lm(residuals, np.zeros(4))
    for _ in range(2):  # down-weight dots with large residuals
        r = residuals(p)
        k = 1.5 * 1.4826 * np.median(np.abs(r)) + 1e-6
        p = _lm(residuals, p, iters=30, weights=np.minimum(1.0, k / (np.abs(r) + 1e-9)))
    cal = GeoCalibration(plane, gain=np.exp(p[2:]), kappa=p[:2])

    ok = [(cal.hit_cm(m, head_comp=False), tgt) for m, tgt in groups]
    ok = [(h, t) for h, t in ok if h is not None]
    if len(ok) >= RESIDUAL_MIN_DOTS:
        H, Y = np.array([h for h, _ in ok]), np.array([t for _, t in ok])
        w_cm = plane.screen.width_pt * plane.cm_per_pt
        h_cm = plane.screen.height_pt * plane.cm_per_pt
        centre = np.array([plane.camera_x_offset_cm, plane.camera_below_top_cm - h_cm / 2])
        half = np.array([w_cm / 2, h_cm / 2])
        X = _quad(H, centre, half)
        reg = RESIDUAL_RIDGE * len(ok) * np.eye(6)
        reg[0, 0] = 1e-6  # don't shrink the constant offset
        cal.residual = (centre, half, np.linalg.solve(X.T @ X + reg, X.T @ (Y - H)))

    if sweep:
        feats, resid, dot_feats = [], [], []
        for frames, target, is_dot in [(ms, t, True) for ms, t in dots] + [(ms, t, False) for ms, t in sweep]:
            tgt = plane.to_cm(*target)
            for m in frames:
                h = cal.hit_cm(m, head_comp=False)
                if h is None:
                    continue
                feats.append(_head_features(m))
                resid.append(tgt - h)
                if is_dot:
                    dot_feats.append(feats[-1])
        if len(feats) >= 30 and dot_feats:
            X, Y = np.array(feats), np.array(resid)
            ref = np.mean(dot_feats, axis=0)
            scale = np.maximum(X.std(axis=0), HEAD_SCALE_FLOOR)
            Z = (X - ref) / scale
            coef = np.linalg.solve(Z.T @ Z + HEAD_COMP_RIDGE * len(Z) * np.eye(4), Z.T @ Y)
            cal.head_comp = (ref, scale, coef)
    return cal
