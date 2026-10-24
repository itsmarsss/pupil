"""Per-user gaze calibration and prediction (no tape measure).

Physical model, in the head frame: gaze angle = gain * measured angle + kappa, for yaw and pitch, shared by both
eyes or per eye. Optional fixes, each switchable via GazeConfig so a test can compare them on the same data:

  iris_mode="equator"  vertical iris centre from the iris's left/right edge points (eyelids drag the centre point)
  per_eye              separate kappa/gains per eye, then average the two screen hits
  rigid_pose           head rotation from rigid landmarks (nose, forehead, cheekbones) via a per-user reference
                       shape, so eyelid motion can't leak into head pitch
  residual_affine      small ridge-regularised affine correction on screen after the physical fit
  blink_reject         drop frames whose eye openness is well below the user's normal

The camera FOV is NOT fitted: with free gains it can't be separated from them (see test_calib.py); measure it once
with calibrate_fov.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

import features as F
import geometry as g

USER_JSON = Path(__file__).with_name("user.json")
MISS_CM = 50.0      # residual for a frame whose gaze ray doesn't reach the screen
BLINK_RATIO = 0.6   # eye openness below this fraction of the user's median = blink


@dataclass(frozen=True)
class GazeConfig:
    iris_mode: str = "equator"   # "center" (MediaPipe's centre landmark) or "equator"
    per_eye: bool = True
    rigid_pose: bool = True
    residual_affine: bool = True
    blink_reject: bool = True

    def label(self) -> str:
        on = [n for n, v in (("equator", self.iris_mode == "equator"), ("per-eye", self.per_eye), ("rigid", self.rigid_pose),
                             ("affine", self.residual_affine), ("blink", self.blink_reject)) if v]
        return "+".join(on) or "baseline"


# Cumulative ablation: each row adds one fix to the previous one
ABLATION = [
    GazeConfig("center", False, False, False, False),
    GazeConfig("center", False, False, False, True),
    GazeConfig("equator", False, False, False, True),
    GazeConfig("equator", True, False, False, True),
    GazeConfig("equator", True, True, False, True),
    GazeConfig("equator", True, True, True, True),
]


@dataclass
class Fixation:
    lms: np.ndarray     # (478, 3)
    pose: np.ndarray    # (4, 4)
    target: np.ndarray  # (3,) target on the screen plane, camera space cm
    group: int = 0      # which calibration dot it belongs to


def _eye(d: dict, base: g.EyeParams) -> g.EyeParams:
    return replace(base, **{k: d[k] for k in ("kappa_yaw_deg", "kappa_pitch_deg", "gain_yaw", "gain_pitch")})


@dataclass
class UserCalib:
    config: dict = field(default_factory=lambda: asdict(GazeConfig()))
    eyes: dict = field(default_factory=lambda: {e: {"kappa_yaw_deg": 0.0, "kappa_pitch_deg": 0.0, "gain_yaw": 1.0, "gain_pitch": 1.0}
                                                for e in ("right", "left")})
    affine: list | None = None       # 2x3, screen-plane cm: corrected = A @ [x, y, 1]
    rigid_ref: list | None = None    # (N, 3) rigid landmarks in the head frame
    ear_ref: float | None = None     # median eye openness during calibration
    rms_cm: float = float("nan")

    @property
    def cfg(self) -> GazeConfig:
        return GazeConfig(**self.config)

    def eye(self, base: g.EyeParams = g.EyeParams()) -> dict[str, g.EyeParams]:
        return {e: _eye(self.eyes[e], base) for e in ("right", "left")}

    # shorthands for status lines (both eyes averaged)
    def mean(self, k: str) -> float:
        return (self.eyes["right"][k] + self.eyes["left"][k]) / 2

    def save(self, path: Path = USER_JSON) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = USER_JSON) -> "UserCalib | None":
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        if "eyes" not in d:  # older single-model file
            e = {k: d.get(k, v) for k, v in (("kappa_yaw_deg", 0.0), ("kappa_pitch_deg", 0.0), ("gain_yaw", 1.0), ("gain_pitch", 1.0))}
            return cls(config=asdict(GazeConfig("center", False, False, False, False)), eyes={"right": dict(e), "left": dict(e)},
                       rms_cm=d.get("rms_cm", float("nan")))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------- per-frame inputs ----------

def iris_points(lms: np.ndarray, mode: str) -> dict[str, np.ndarray] | None:
    if mode == "center":
        return None
    return {"right": F.iris_center_equator(lms, F.R_IRIS_RING), "left": F.iris_center_equator(lms, F.L_IRIS_RING)}


def rigid_head_R(lms: np.ndarray, pose: np.ndarray, intr: g.Intrinsics, ref: np.ndarray, depth_scale: float | None) -> np.ndarray:
    _, t = g.split_pose(pose)
    k = g.mediapipe_depth_scale(intr) if depth_scale is None else depth_scale
    pts = g.backproject(lms[list(F.RIGID)], abs(t[2] * k), intr)
    R, _ = F.kabsch(ref, pts)
    return R


def rigid_reference(fix: list[Fixation], intr: g.Intrinsics) -> np.ndarray:
    """Average rigid-landmark shape in MediaPipe's head frame over the calibration frames."""
    k = g.mediapipe_depth_scale(intr)
    shapes = []
    for f in fix:
        R, t = g.split_pose(f.pose)
        t = np.array([t[0], t[1], t[2] * k])
        pts = g.backproject(f.lms[list(F.RIGID)], abs(t[2]), intr)
        shapes.append((pts - t) @ R)  # R.T @ (p - t) for each row
    return np.mean(shapes, axis=0)


def predict(lms: np.ndarray, pose: np.ndarray, intr: g.Intrinsics, screen: g.Screen, uc: UserCalib | None,
            base: g.EyeParams = g.EyeParams(), depth_scale: float | None = None, aspect: float | None = None) -> g.GazeFrame | None:
    """Gaze for one frame with the user's calibration and its config. None = blink (when blink rejection is on)."""
    if uc is None:
        return g.compute(lms, pose, intr, screen, base, depth_scale)
    cfg = uc.cfg
    if cfg.blink_reject and uc.ear_ref and F.ear(lms, aspect or intr.width / intr.height) < BLINK_RATIO * uc.ear_ref:
        return None
    head_R = rigid_head_R(lms, pose, intr, np.array(uc.rigid_ref), depth_scale) if cfg.rigid_pose and uc.rigid_ref else None
    gf = g.compute(lms, pose, intr, screen, uc.eye(base), depth_scale, iris_points(lms, cfg.iris_mode), head_R)
    if cfg.residual_affine and uc.affine is not None and gf.hit is not None:
        A = np.array(uc.affine)
        gf.hit = np.array([*(A @ [gf.hit[0], gf.hit[1], 1.0]), 0.0])
        gf.hit_px = screen.to_px(gf.hit)
        gf.direction = g.normalize(gf.hit - gf.origin)
    return gf


# ---------- fitting ----------

def _lm(res_fn, p0: np.ndarray, steps: np.ndarray, iters: int = 60, weights: np.ndarray | None = None):
    """Levenberg-Marquardt with forward-difference Jacobian; optional per-residual weights."""
    w = np.ones(1) if weights is None else np.sqrt(weights)
    f = lambda p: res_fn(p) * w
    p, r = p0.copy(), f(p0)
    cost, lam, n = r @ r, 1e-2, p0.size
    for _ in range(iters):
        J = np.empty((r.size, n))
        for j in range(n):
            dp = np.zeros(n)
            dp[j] = steps[j]
            J[:, j] = (f(p + dp) - r) / steps[j]
        A, b = J.T @ J, J.T @ r
        improved = False
        while lam <= 1e8:
            delta = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), b)
            r_new = f(p + delta)
            if r_new @ r_new < cost:
                p, r, cost, lam, improved = p + delta, r_new, r_new @ r_new, lam / 3, True
                break
            lam *= 4
        if not improved or np.abs(delta).max() < 1e-6:
            break
    return p


def _params_to_eyes(p: np.ndarray, per_eye: bool) -> dict:
    def one(q):
        return {"kappa_yaw_deg": float(q[0]), "kappa_pitch_deg": float(q[1]), "gain_yaw": float(np.exp(q[2])), "gain_pitch": float(np.exp(q[3]))}
    return {"right": one(p[:4]), "left": one(p[4:8] if per_eye else p[:4])}


def _hits(fix: list[Fixation], uc: UserCalib, intr: g.Intrinsics, screen: g.Screen, base: g.EyeParams, per_eye_hits: bool):
    """Residual-ready hits (cm) for each fixation: per-eye hits stacked, or the combined hit."""
    cfg = uc.cfg
    eyes = uc.eye(base)
    ref = np.array(uc.rigid_ref) if cfg.rigid_pose and uc.rigid_ref else None
    out = []
    for f in fix:
        head_R = rigid_head_R(f.lms, f.pose, intr, ref, None) if ref is not None else None
        gf = g.compute(f.lms, f.pose, intr, screen, eyes, None, iris_points(f.lms, cfg.iris_mode), head_R)
        if per_eye_hits:
            out.append([gf.eye_hits.get(e) for e in ("right", "left")])
        else:
            out.append([gf.hit])
    return out


def _residuals(hits, fix) -> np.ndarray:
    r = []
    for hs, f in zip(hits, fix):
        for h in hs:
            r.extend((h[:2] - f.target[:2]) if h is not None else (MISS_CM, MISS_CM))
    return np.array(r)


def _group_means(fix: list[Fixation]) -> list[Fixation]:
    """One averaged fixation per group (a calibration dot, or a coverage tile for mouse-follow calibration).
    Fitting single noisy frames biases the gains low (noise in the measured angle = regression dilution);
    averaging ~10 frames first cuts that noise ~3x."""
    out = []
    for grp in sorted({f.group for f in fix}):
        fs = [f for f in fix if f.group == grp]
        pose = np.mean([f.pose for f in fs], axis=0)
        U, _, Vt = np.linalg.svd(pose[:3, :3])  # re-orthonormalise the averaged rotation
        pose[:3, :3] = U @ Vt
        out.append(Fixation(np.mean([f.lms for f in fs], axis=0), pose, np.mean([f.target for f in fs], axis=0), grp))
    return out


def fit(fix: list[Fixation], intr: g.Intrinsics, screen: g.Screen, cfg: GazeConfig = GazeConfig(),
        base: g.EyeParams = g.EyeParams()) -> UserCalib:
    aspect = intr.width / intr.height
    uc = UserCalib(config=asdict(cfg))
    ears = np.array([F.ear(f.lms, aspect) for f in fix])
    uc.ear_ref = float(np.median(ears))
    if cfg.blink_reject:
        fix = [f for f, e in zip(fix, ears) if e >= BLINK_RATIO * uc.ear_ref]
    if cfg.rigid_pose:
        uc.rigid_ref = rigid_reference(fix, intr).tolist()

    # Drop glances/saccades: per dot, frames whose uncalibrated hit is far from that dot's median
    raw = _hits(fix, uc, intr, screen, base, per_eye_hits=False)
    keep = []
    for grp in sorted({f.group for f in fix}):
        idx = [i for i, f in enumerate(fix) if f.group == grp and raw[i][0] is not None]
        if not idx:
            continue
        pts = np.array([raw[i][0][:2] for i in idx])
        d = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
        lim = max(3 * 1.4826 * np.median(d), 1.0)
        keep += [i for i, di in zip(idx, d) if di <= lim]
    fix = _group_means([fix[i] for i in sorted(keep)])

    n = 8 if cfg.per_eye else 4
    steps = np.tile([1e-2, 1e-2, 1e-3, 1e-3], n // 4)

    def res(p, weights_fix=fix):
        trial = replace(uc, eyes=_params_to_eyes(p, cfg.per_eye))
        return _residuals(_hits(weights_fix, trial, intr, screen, base, per_eye_hits=cfg.per_eye), weights_fix)

    p = _lm(res, np.zeros(n), steps)
    # Huber-style reweighting: down-weight dots with large residuals, refit
    for _ in range(2):
        r = res(p)
        k = 1.5 * 1.4826 * np.median(np.abs(r)) + 1e-6
        w = np.minimum(1.0, k / (np.abs(r) + 1e-9))
        p = _lm(res, p, steps, iters=30, weights=w)
    uc.eyes = _params_to_eyes(p, cfg.per_eye)

    combined = _hits(fix, uc, intr, screen, base, per_eye_hits=False)
    ok = [(h[0], f) for h, f in zip(combined, fix) if h[0] is not None]
    if cfg.residual_affine and len({f.group for _, f in ok}) >= 4:
        # ridge regression toward identity: corrected = A @ [x, y, 1]
        X = np.array([[h[0], h[1], 1.0] for h, _ in ok])
        Y = np.array([f.target[:2] for _, f in ok])
        lam = 1.0 * len(ok)  # keeps the slopes within a few % of identity unless the data insist
        A0 = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        reg = lam * np.diag([1.0, 1.0, 0.01])
        uc.affine = np.linalg.solve(X.T @ X + reg, X.T @ Y + reg @ A0.T).T.tolist()
    final = []
    for h, f in ok:
        hit = h
        if uc.affine is not None:
            hit = np.array([*(np.array(uc.affine) @ [h[0], h[1], 1.0]), 0.0])
        final.append(np.linalg.norm(hit[:2] - f.target[:2]))
    uc.rms_cm = float(np.sqrt(np.mean(np.square(final)))) if final else float("nan")
    return uc


def drift_correct(uc: UserCalib, fix: list[Fixation], intr: g.Intrinsics, screen: g.Screen,
                  base: g.EyeParams = g.EyeParams()) -> UserCalib:
    """Re-centre after the head has moved: refit only a common offset added to both eyes' kappa (1 dot)."""
    aspect = intr.width / intr.height
    if uc.cfg.blink_reject and uc.ear_ref:
        fix = [f for f in fix if F.ear(f.lms, aspect) >= BLINK_RATIO * uc.ear_ref]
    if not fix:
        return uc

    def with_offset(p):
        eyes = {e: dict(v, kappa_yaw_deg=v["kappa_yaw_deg"] + p[0], kappa_pitch_deg=v["kappa_pitch_deg"] + p[1])
                for e, v in uc.eyes.items()}
        return replace(uc, eyes=eyes)

    def res(p):
        trial = with_offset(p)
        r = []
        for f in fix:
            gf = predict(f.lms, f.pose, intr, screen, replace(trial, config=dict(trial.config, blink_reject=False)), base)
            r.extend((gf.hit[:2] - f.target[:2]) if gf is not None and gf.hit is not None else (MISS_CM, MISS_CM))
        return np.array(r)

    p = _lm(res, np.zeros(2), np.array([1e-2, 1e-2]), iters=30)
    return with_offset(p)
