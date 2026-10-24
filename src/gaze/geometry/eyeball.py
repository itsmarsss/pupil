"""Model-based gaze geometry (ported from geo/geometry.py).

Camera space in cm, OpenGL-style: +x = image right (the user's left; frames aren't mirrored), +y up, +z out of
the lens toward the user's back, so the face is at negative z. The camera is the origin and the screen is the
plane z = 0 (camera and screen are both in the lid).

Per frame:
  1. Back-project MediaPipe's landmarks into metric 3D using the real focal length and the head depth from
     MediaPipe's facial transformation matrix (rescaled from the 63° vertical FOV MediaPipe assumes).
  2. Eyeball centre = eye-corner midpoint pushed back along the head's forward axis.
  3. Iris 3D = camera ray through the iris pixel ∩ eyeball sphere.
  4. Optical axis = centre → iris, expressed as yaw/pitch in the head frame.

`measure()` stops there and returns the per-frame quantities that don't depend on the user's calibration, so
fitting gain/kappa only re-runs cheap angle arithmetic (`hits()`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.screen import Screen
from ..core.types import FaceObservation

R_EYE_CORNERS = (33, 133)  # subject's right eye: outer, inner
L_EYE_CORNERS = (362, 263)  # subject's left eye: inner, outer
NOSE_TIP = 1
IRIS = {"right": 468, "left": 473}
CORNERS = {"right": R_EYE_CORNERS, "left": L_EYE_CORNERS}
MEDIAPIPE_VFOV_DEG = 63.0
EYEBALL_RADIUS_CM = 1.2
CENTRE_DEPTH_CM = 0.6  # eyeball centre behind the eye-corner midpoint


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 0, v / np.where(n > 0, n, 1), v)


@dataclass(frozen=True)
class ScreenPlane:
    """Maps our screen points (top-left origin) to camera-space cm on the lid plane."""

    screen: Screen
    camera_below_top_cm: float = 0.32
    camera_x_offset_cm: float = 0.0  # camera centre offset from screen centre (+ = camera +x = user's left)

    @property
    def cm_per_pt(self) -> float:
        return self.screen.mm_per_pt / 10

    def to_cm(self, x_pt, y_pt) -> np.ndarray:
        x = self.camera_x_offset_cm + self.screen.width_pt * self.cm_per_pt / 2 - np.asarray(x_pt) * self.cm_per_pt
        y = self.camera_below_top_cm - np.asarray(y_pt) * self.cm_per_pt
        return np.stack([x, y], axis=-1)

    def to_pt(self, xy_cm: np.ndarray) -> np.ndarray:
        xy_cm = np.asarray(xy_cm)
        x = (self.camera_x_offset_cm + self.screen.width_pt * self.cm_per_pt / 2 - xy_cm[..., 0]) / self.cm_per_pt
        y = (self.camera_below_top_cm - xy_cm[..., 1]) / self.cm_per_pt
        return np.stack([x, y], axis=-1)


@dataclass(frozen=True)
class EyeMeasure:
    """Calibration-independent quantities for one frame."""

    centres: np.ndarray  # (2, 3) eyeball centres, right then left (cm)
    angles: np.ndarray  # (2, 2) measured optical-axis yaw, pitch in the head frame (degrees)
    basis: np.ndarray  # (3, 3) head axes as columns: x, y, forward (toward the camera)
    head_ypr: np.ndarray  # (3,) head yaw, pitch, roll (degrees)
    nose: np.ndarray  # (3,) nose tip (cm)

    def features(self) -> np.ndarray:
        """Compact descriptors for other models: eye angles (4), head yaw/pitch (2), nose position (3)."""
        return np.concatenate([self.angles.ravel(), self.head_ypr[:2], self.nose])


def _split_pose(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = np.asarray(M, float).reshape(4, 4)
    if abs(M[3, 0]) + abs(M[3, 1]) + abs(M[3, 2]) > abs(M[0, 3]) + abs(M[1, 3]) + abs(M[2, 3]):
        M = M.T  # translation was in the bottom row
    return M[:3, :3].copy(), M[:3, 3].copy()


def _head_ypr(R: np.ndarray) -> np.ndarray:
    f, up = R[:, 2], R[:, 1]
    return np.array([np.degrees(np.arctan2(f[0], f[2])), np.degrees(np.arctan2(f[1], np.hypot(f[0], f[2]))),
                     np.degrees(np.arctan2(-up[0], up[1]))])


def to_angles(B: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Directions (..., 3) → (..., 2) yaw, pitch in degrees relative to the head frame B."""
    h = d @ B
    return np.degrees(np.stack([np.arctan2(h[..., 0], h[..., 2]), np.arctan2(h[..., 1], np.hypot(h[..., 0], h[..., 2]))], -1))


def from_angles(B: np.ndarray, ang: np.ndarray) -> np.ndarray:
    a, b = np.radians(ang[..., 0]), np.radians(ang[..., 1])
    local = np.stack([np.cos(b) * np.sin(a), np.sin(b), np.cos(b) * np.cos(a)], -1)
    return local @ B.T


def _ray_sphere_near(ray: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    b = ray @ centre
    disc = b * b - (centre @ centre - radius * radius)
    if disc >= 0:
        return ray * (b - np.sqrt(disc))
    return centre + radius * normalize(ray * b - centre)  # missed: closest sphere point to the ray


def measure(face: FaceObservation) -> EyeMeasure | None:
    if face.transform is None or face.focal_px is None:
        return None
    w, h = face.frame_size
    tan_x, tan_y = w / (2 * face.focal_px), h / (2 * face.focal_px)
    R, t = _split_pose(face.transform)
    f_mp = 1 / np.tan(np.radians(MEDIAPIPE_VFOV_DEG) / 2)
    depth = abs(t[2]) * (1 / tan_y) / f_mp

    lm = face.landmarks
    u, v, z = lm[:, 0] / w, lm[:, 1] / h, lm[:, 2] / w  # MediaPipe's normalised coordinates
    d = depth * (1 + 2 * tan_x * z)
    pts = np.stack([(2 * u - 1) * tan_x * d, (1 - 2 * v) * tan_y * d, -d], axis=1)

    fwd = R[:, 2] if R[2, 2] > 0 else -R[:, 2]
    B = np.stack([R[:, 0], R[:, 1], fwd], axis=1)
    centres, angles = [], []
    for eye in ("right", "left"):
        c = pts[list(CORNERS[eye])].mean(axis=0) - fwd * CENTRE_DEPTH_CM
        i = IRIS[eye]
        ray = normalize(np.array([(2 * u[i] - 1) * tan_x, (1 - 2 * v[i]) * tan_y, -1.0]))
        iris = _ray_sphere_near(ray, c, EYEBALL_RADIUS_CM)
        centres.append(c)
        angles.append(to_angles(B, normalize(iris - c)))
    return EyeMeasure(np.array(centres), np.array(angles), B, _head_ypr(R), pts[NOSE_TIP])


def hits(m: EyeMeasure, gain: np.ndarray, kappa: np.ndarray) -> np.ndarray | None:
    """Screen-plane hit (cm, x/y) of the calibrated visual axes, averaged over both eyes.
    gain, kappa: (2,) for yaw, pitch; angle = gain * measured + kappa."""
    dirs = from_angles(m.basis, m.angles * gain + kappa)  # (2, 3)
    if np.any(dirs[:, 2] <= 1e-6):
        return None  # a ray points away from the screen
    s = -m.centres[:, 2] / dirs[:, 2]
    return (m.centres + dirs * s[:, None])[:, :2].mean(axis=0)


MEASURE_DIM = 25  # flattened EyeMeasure: centres 6, angles 4, basis 9, head_ypr 3, nose 3
GEO_DIM = 11  # geo_block: EyeMeasure.features() 9 + uncalibrated hit 2


def to_array(m: EyeMeasure) -> np.ndarray:
    return np.concatenate([m.centres.ravel(), m.angles.ravel(), m.basis.ravel(), m.head_ypr, m.nose]).astype(np.float32)


def from_array(a: np.ndarray) -> EyeMeasure:
    a = np.asarray(a, np.float64)
    return EyeMeasure(a[0:6].reshape(2, 3), a[6:10].reshape(2, 2), a[10:19].reshape(3, 3), a[19:22], a[22:25])


def geo_block(m: EyeMeasure, plane: ScreenPlane) -> np.ndarray | None:
    """Geometric features for the learned calibrator: eye and head angles, nose position, and where the
    uncalibrated eye model says the gaze lands (screen points)."""
    h = hits(m, np.ones(2), np.zeros(2))
    if h is None:
        return None
    return np.concatenate([m.features(), plane.to_pt(h)]).astype(np.float32)
