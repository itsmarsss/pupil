"""Model-based gaze geometry.

Frame: camera space in cm, OpenGL-style ("RUB"): +x = image right (the user's LEFT, since the
raw webcam image is not mirrored), +y = up, +z = out of the lens toward the user's back
(so the face sits at negative z). The camera is the origin.

Pipeline per frame:
  1. Back-project MediaPipe's 478 landmarks into metric 3D using the real camera intrinsics
     and the head depth from MediaPipe's facial transformation matrix (rescaled for the
     intrinsics MediaPipe assumed vs the real ones).
  2. Eyeball centre = midpoint of the eye corners, pushed back along the head's forward axis.
  3. Iris 3D = where the camera ray through the iris pixel meets the eyeball sphere.
  4. Gaze ray = eyeball centre -> iris; intersect with the screen plane (z = 0, the lid).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# MediaPipe face mesh indices
R_EYE_CORNERS = (33, 133)   # subject's right eye: outer, inner
L_EYE_CORNERS = (362, 263)  # subject's left eye: inner, outer
NOSE_TIP = 1
R_IRIS_CENTER = 468
L_IRIS_CENTER = 473

MEDIAPIPE_VFOV_DEG = 63.0  # perspective MediaPipe's face-geometry module assumes


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float  # px
    fy: float  # px

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        fx = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        return cls(width, height, fx, fx)  # square pixels

    @property
    def tan_x(self) -> float:
        return self.width / (2 * self.fx)

    @property
    def tan_y(self) -> float:
        return self.height / (2 * self.fy)


@dataclass
class Screen:
    """Usable display rectangle in the lid plane (z = 0), camera at the origin."""
    width_cm: float
    height_cm: float
    width_px: int
    height_px: int
    cam_to_top_cm: float = 0.5   # camera centre above the top edge of the usable area
    cam_x_offset_cm: float = 0.0  # camera centre offset from screen centre (+ = camera +x)

    def corners(self) -> np.ndarray:
        x0 = self.cam_x_offset_cm - self.width_cm / 2
        x1 = self.cam_x_offset_cm + self.width_cm / 2
        y0 = -self.cam_to_top_cm
        y1 = -self.cam_to_top_cm - self.height_cm
        return np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], float)

    def to_px(self, p: np.ndarray) -> np.ndarray:
        """Camera-space point on the screen plane -> screen pixels (origin top-left as the user sees it)."""
        # camera +x is the user's left, so screen u grows toward camera -x
        u = (self.cam_x_offset_cm + self.width_cm / 2 - p[0]) / self.width_cm
        v = (-self.cam_to_top_cm - p[1]) / self.height_cm
        return np.array([u * self.width_px, v * self.height_px])

    def from_px(self, px: np.ndarray) -> np.ndarray:
        u, v = px[0] / self.width_px, px[1] / self.height_px
        return np.array([self.cam_x_offset_cm + self.width_cm / 2 - u * self.width_cm,
                         -self.cam_to_top_cm - v * self.height_cm, 0.0])


@dataclass
class EyeParams:
    eyeball_radius_cm: float = 1.2
    center_depth_cm: float = 0.6  # eyeball centre behind the eye-corner midpoint, along the head axis
    # Per-person correction of the measured eye rotation, in the head frame so it follows head rotation:
    #   angle = gain * measured + kappa. Kappa is the visual-axis offset (~5°); the gains absorb iris-landmark
    #   bias, notably vertical, where the eyelid follows the iris and hides its true movement.
    kappa_yaw_deg: float = 0.0
    kappa_pitch_deg: float = 0.0
    gain_yaw: float = 1.0
    gain_pitch: float = 1.0


@dataclass
class Eye:
    corners: np.ndarray      # (2, 3)
    center: np.ndarray       # (3,) eyeball centre
    iris: np.ndarray         # (3,) iris centre on the sphere
    direction: np.ndarray    # (3,) unit optical axis
    iris_hit_sphere: bool    # False if the iris ray missed the sphere (clamped)


@dataclass
class GazeFrame:
    points: np.ndarray                 # (478, 3) metric landmarks
    head_R: np.ndarray                 # (3, 3) head rotation (columns = head axes in camera space)
    head_t: np.ndarray                 # (3,) head origin (MediaPipe canonical origin), rescaled
    head_quat_xyzw: np.ndarray         # (4,)
    head_ypr_deg: np.ndarray           # (3,) yaw, pitch, roll
    eyes: dict[str, Eye] = field(default_factory=dict)
    origin: np.ndarray | None = None   # cyclopean eye (mean of eyeball centres)
    direction: np.ndarray | None = None
    hit: np.ndarray | None = None      # (3,) on screen plane
    hit_px: np.ndarray | None = None   # (2,)
    eye_hits: dict[str, np.ndarray] = field(default_factory=dict)
    eye_hits_px: dict[str, np.ndarray] = field(default_factory=dict)
    # where the face points: ray from the nose tip along the head's forward axis
    head_origin: np.ndarray | None = None
    head_dir: np.ndarray | None = None
    head_hit: np.ndarray | None = None
    head_hit_px: np.ndarray | None = None


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def head_ypr_deg(R: np.ndarray) -> np.ndarray:
    """Yaw (about y), pitch (about x), roll (about z) of the head's forward axis, in degrees.
    0/0/0 = facing straight into the camera."""
    f = R[:, 2]
    yaw = np.degrees(np.arctan2(f[0], f[2]))
    pitch = np.degrees(np.arctan2(f[1], np.hypot(f[0], f[2])))
    up = R[:, 1]
    roll = np.degrees(np.arctan2(-up[0], up[1]))
    return np.array([yaw, pitch, roll])


def split_pose(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MediaPipe facial transformation matrix -> (R, t). Accepts row- or column-major layouts."""
    M = np.asarray(M, float).reshape(4, 4)
    if abs(M[3, 0]) + abs(M[3, 1]) + abs(M[3, 2]) > abs(M[0, 3]) + abs(M[1, 3]) + abs(M[2, 3]):
        M = M.T  # translation was in the bottom row
    return M[:3, :3].copy(), M[:3, 3].copy()


def mediapipe_depth_scale(intr: Intrinsics) -> float:
    """MediaPipe solves head pose assuming a 63° vertical FOV. With the real focal length,
    lateral positions stay the same and depths scale by f_real / f_mediapipe."""
    f_mp = 1 / np.tan(np.radians(MEDIAPIPE_VFOV_DEG) / 2)
    return (1 / intr.tan_y) / f_mp


def backproject(lms: np.ndarray, head_depth_cm: float, intr: Intrinsics) -> np.ndarray:
    """(N, 3) normalized MediaPipe landmarks (x, y in [0,1] of the image, z relative depth in
    units of image width, smaller = closer) -> (N, 3) camera-space cm."""
    # landmark z is relative to the head centre, in image-width units; 1 image width at depth D spans 2·tan_x·D cm
    d = head_depth_cm * (1 + 2 * intr.tan_x * lms[:, 2])
    xn = 2 * lms[:, 0] - 1
    yn = 1 - 2 * lms[:, 1]
    return np.stack([xn * intr.tan_x * d, yn * intr.tan_y * d, -d], axis=1)


def pixel_ray(u: float, v: float, intr: Intrinsics) -> np.ndarray:
    """Normalized image coords -> unit ray direction from the camera."""
    return normalize(np.array([(2 * u - 1) * intr.tan_x, (1 - 2 * v) * intr.tan_y, -1.0]))


def ray_sphere_near(ray: np.ndarray, center: np.ndarray, radius: float) -> tuple[np.ndarray, bool]:
    """Nearest intersection of a ray from the origin with a sphere; if it misses, the sphere point
    closest to the ray."""
    b = ray @ center
    disc = b * b - (center @ center - radius * radius)
    if disc >= 0:
        return ray * (b - np.sqrt(disc)), True
    closest = ray * b
    return center + radius * normalize(closest - center), False


def ray_plane_z0(origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    if direction[2] <= 1e-6:  # pointing away from the screen
        return None
    return origin + direction * (-origin[2] / direction[2])


def head_basis(R: np.ndarray) -> np.ndarray:
    """Columns: head x, head y, head forward (pointing toward the camera)."""
    fwd = R[:, 2] if R[2, 2] > 0 else -R[:, 2]
    return np.stack([R[:, 0], R[:, 1], fwd], axis=1)


def to_angles(B: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """Direction -> (yaw, pitch) in degrees relative to the head's forward axis."""
    h = B.T @ d
    return float(np.degrees(np.arctan2(h[0], h[2]))), float(np.degrees(np.arctan2(h[1], np.hypot(h[0], h[2]))))


def from_angles(B: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    a, b = np.radians(yaw), np.radians(pitch)
    return B @ np.array([np.cos(b) * np.sin(a), np.sin(b), np.cos(b) * np.cos(a)])


def apply_eye_model(B: np.ndarray, d: np.ndarray, eye: EyeParams) -> np.ndarray:
    yaw, pitch = to_angles(B, d)
    return from_angles(B, eye.gain_yaw * yaw + eye.kappa_yaw_deg, eye.gain_pitch * pitch + eye.kappa_pitch_deg)


def invert_eye_model(B: np.ndarray, d: np.ndarray, eye: EyeParams) -> np.ndarray:
    """Visual direction -> the optical direction the measurement would show (for synthetic data)."""
    yaw, pitch = to_angles(B, d)
    return from_angles(B, (yaw - eye.kappa_yaw_deg) / eye.gain_yaw, (pitch - eye.kappa_pitch_deg) / eye.gain_pitch)


def compute(lms: np.ndarray, pose: np.ndarray, intr: Intrinsics, screen: Screen,
            eye: "EyeParams | dict[str, EyeParams]" = EyeParams(), depth_scale: float | None = None,
            iris_px: dict[str, np.ndarray] | None = None, head_R: np.ndarray | None = None) -> GazeFrame:
    """lms: (478, 3) normalized landmarks; pose: MediaPipe 4x4 facial transformation matrix.
    eye: one EyeParams for both eyes, or {"right": ..., "left": ...}.
    iris_px: per-eye iris centre (normalized image coords) to use instead of MediaPipe's centre landmark.
    head_R: head rotation to use instead of MediaPipe's (e.g. from rigid landmarks)."""
    R, t = split_pose(pose)
    if head_R is not None:
        R = head_R
    k = mediapipe_depth_scale(intr) if depth_scale is None else depth_scale
    t = np.array([t[0], t[1], t[2] * k])
    head_depth = abs(t[2])

    pts = backproject(lms, head_depth, intr)

    fwd = R[:, 2] if R[2, 2] > 0 else -R[:, 2]  # head forward axis points toward the camera (+z)
    gf = GazeFrame(points=pts, head_R=R, head_t=t, head_quat_xyzw=rot_to_quat_xyzw(R), head_ypr_deg=head_ypr_deg(R))
    B = head_basis(R)
    eyes = eye if isinstance(eye, dict) else {"right": eye, "left": eye}
    gf.head_origin, gf.head_dir = pts[NOSE_TIP], B[:, 2]
    gf.head_hit = ray_plane_z0(gf.head_origin, gf.head_dir)
    if gf.head_hit is not None:
        gf.head_hit_px = screen.to_px(gf.head_hit)

    for name, corners_idx, iris_idx in (("right", R_EYE_CORNERS, R_IRIS_CENTER), ("left", L_EYE_CORNERS, L_IRIS_CENTER)):
        corners = pts[list(corners_idx)]
        ep = eyes[name]
        center = corners.mean(axis=0) - fwd * ep.center_depth_cm
        u, v = iris_px[name] if iris_px is not None else lms[iris_idx, :2]
        ray = pixel_ray(u, v, intr)
        iris, ok = ray_sphere_near(ray, center, ep.eyeball_radius_cm)
        d = normalize(iris - center)
        if not (ep.kappa_yaw_deg == 0 and ep.kappa_pitch_deg == 0 and ep.gain_yaw == 1 and ep.gain_pitch == 1):
            d = apply_eye_model(B, d, ep)
        gf.eyes[name] = Eye(corners, center, iris, d, ok)
        h = ray_plane_z0(center, d)
        if h is not None:
            gf.eye_hits[name] = h
            gf.eye_hits_px[name] = screen.to_px(h)

    # Combine the eyes by averaging their screen hits: the two rays converge on the target, so a single
    # averaged-direction ray from between the eyes would miss it slightly.
    gf.origin = (gf.eyes["right"].center + gf.eyes["left"].center) / 2
    if len(gf.eye_hits) == 2:
        gf.hit = (gf.eye_hits["right"] + gf.eye_hits["left"]) / 2
        gf.direction = normalize(gf.hit - gf.origin)
    if gf.hit is not None:
        gf.hit_px = screen.to_px(gf.hit)
    return gf
