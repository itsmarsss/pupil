"""MediaPipe face landmarks → face/eye crops, eye openness, head pose and distance."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..core.paths import ensure_face_landmarker
from ..core.types import FaceObservation, Frame, HeadPose, Rect
from ..filter.one_euro import OneEuroFilter

# MediaPipe face mesh indices. "Left"/"right" mean the image's left/right.
L_OUTER, L_INNER, L_TOP, L_BOTTOM = 33, 133, 159, 145
R_INNER, R_OUTER, R_TOP, R_BOTTOM = 362, 263, 386, 374
L_IRIS_H = (469, 471)
R_IRIS_H = (474, 476)
LIP = [61, 91, 14, 178, 402, 324, 95]

IRIS_DIAMETER_MM = 11.7

# Generic face model for solvePnP, in mm with y up and z toward the viewer; flipped
# below to OpenCV's camera convention (y down, z away from the camera).
_PNP_IDX = [1, 152, 33, 263, 61, 291]
_PNP_MODEL = np.array([1.0, -1.0, -1.0]) * np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip
        [0.0, -63.6, -12.5],  # chin
        [-43.3, 32.7, -26.0],  # eye outer corner (image left)
        [43.3, 32.7, -26.0],  # eye outer corner (image right)
        [-28.9, -28.9, -24.1],  # mouth corner
        [28.9, -28.9, -24.1],
    ]
)


def eye_rects(lm: np.ndarray) -> tuple[Rect, Rect]:
    """Eye boxes in the layout the gaze model was trained on: corners padded by 20% of
    the inner-corner distance, height = 0.75 x width, 60% above the corner line."""
    scale = abs(lm[R_INNER, 0] - lm[L_INNER, 0]) / 100
    pad = 20 * scale

    def box(a: int, b: int) -> Rect:
        x0, x1 = lm[a, 0] - pad, lm[b, 0] + pad
        h = abs(x1 - x0) * 0.75
        cy = (lm[a, 1] + lm[b, 1]) / 2
        y0, y1 = cy - 0.6 * h, cy + 0.4 * h
        return int(x0), int(y0), int(x1) - int(x0), int(y1) - int(y0)

    return box(L_OUTER, L_INNER), box(R_INNER, R_OUTER)


def face_rect(lm: np.ndarray, w: int, h: int) -> Rect:
    """Landmark bounding box, squared up by moving width/height difference to the other axis."""
    min_x, max_x = max(lm[:, 0].min(), 0), min(lm[:, 0].max(), w)
    min_y, max_y = max(lm[:, 1].min(), 0), min(lm[:, 1].max(), h)
    delta = ((max_x - min_x) - (max_y - min_y)) / 4
    x0 = max(min_x + delta, 0)
    y0 = max(min_y - delta, 0)
    x1 = min(max_x - delta, w - 1)
    y1 = min(max_y + delta, h - 1)
    return int(x0), int(y0), int(x1) - int(x0), int(y1) - int(y0)


def _inside(rect: Rect, w: int, h: int) -> bool:
    x, y, rw, rh = rect
    return x > 0 and y > 0 and x + rw < w and y + rh < h


def _clamp_rect(r: np.ndarray, w: int, h: int) -> Rect:
    x = max(0, min(int(round(r[0])), w - 1))
    y = max(0, min(int(round(r[1])), h - 1))
    rw = max(5, min(int(round(r[2])), w - x))
    rh = max(5, min(int(round(r[3])), h - y))
    return x, y, rw, rh


def crop(image: np.ndarray, rect: Rect) -> np.ndarray | None:
    x, y, w, h = rect
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x >= image.shape[1] or y >= image.shape[0]:
        return None
    return image[y : y + h, x : x + w]


class FaceTracker:
    """Detects one face per frame. Landmarks are found on a copy no wider than
    `detect_width` and scaled back, so crops come from the full-resolution frame."""

    def __init__(self, detect_width: int = 640, smooth: bool = True, hfov_deg: float = 54.0):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(ensure_face_landmarker()),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        self.detect_width = detect_width
        self.smooth = smooth
        self.hfov_deg = hfov_deg  # horizontal field of view of the (cropped) frame
        self._filters = {k: OneEuroFilter(min_cutoff=1.0, beta=0.01) for k in ("face", "left", "right")}

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()

    def _landmarks(self, frame: Frame) -> np.ndarray | None:
        image = frame.image
        h, w = image.shape[:2]
        if w > self.detect_width:
            small = cv2.resize(image, (self.detect_width, int(h * self.detect_width / w)), interpolation=cv2.INTER_AREA)
        else:
            small = image
        ts_ms = max(frame.t_ns // 1_000_000, self._last_ts_ms + 1)  # VIDEO mode needs increasing timestamps
        self._last_ts_ms = ts_ms
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None
        pts = result.face_landmarks[0]
        return np.array([[p.x * w, p.y * h, p.z * w] for p in pts], dtype=np.float64)

    def process(self, frame: Frame) -> FaceObservation | None:
        """Returns None when there's no usable face (missing, or eyes/mouth off-frame)."""
        h, w = frame.height, frame.width
        lm = self._landmarks(frame)
        if lm is None:
            self.reset()
            return None
        lm = np.round(lm)
        if np.mean(lm[LIP, 1]) >= h:
            self.reset()
            return None

        f_rect = face_rect(lm, w, h)
        l_rect, r_rect = eye_rects(lm)
        if not (_inside(l_rect, w, h) and _inside(r_rect, w, h)):
            self.reset()
            return None

        if self.smooth:
            # Filters were tuned for 640-wide frames; velocities scale with resolution.
            beta_scale = 640 / w
            rects = []
            for key, r in (("face", f_rect), ("left", l_rect), ("right", r_rect)):
                flt = self._filters[key]
                flt.beta = 0.01 * beta_scale
                rects.append(_clamp_rect(flt(np.array(r, dtype=np.float64), frame.t_ns), w, h))
            f_rect, l_rect, r_rect = rects

        openness = (self._openness(lm, L_OUTER, L_INNER, L_TOP, L_BOTTOM), self._openness(lm, R_INNER, R_OUTER, R_TOP, R_BOTTOM))
        iris_px = self._iris_diameter(lm)
        focal_px = (w / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        distance = focal_px * IRIS_DIAMETER_MM / iris_px if iris_px > 1 else None

        return FaceObservation(
            t_ns=frame.t_ns,
            frame_size=(w, h),
            landmarks=lm,
            face_rect=f_rect,
            left_eye_rect=l_rect,
            right_eye_rect=r_rect,
            eye_openness=openness,
            iris_diameter_px=iris_px,
            head_pose=self._head_pose(lm, focal_px, w, h),
            distance_mm=distance,
        )

    @staticmethod
    def _openness(lm: np.ndarray, a: int, b: int, top: int, bottom: int) -> float:
        width = math.dist(lm[a, :2], lm[b, :2])
        return math.dist(lm[top, :2], lm[bottom, :2]) / max(width, 1e-6)

    @staticmethod
    def _iris_diameter(lm: np.ndarray) -> float:
        if lm.shape[0] < 478:
            return 0.0
        return float(np.mean([math.dist(lm[i, :2], lm[j, :2]) for i, j in (L_IRIS_H, R_IRIS_H)]))

    @staticmethod
    def _head_pose(lm: np.ndarray, focal_px: float, w: int, h: int) -> HeadPose | None:
        cam = np.array([[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1]], dtype=np.float64)
        img_pts = lm[_PNP_IDX, :2].astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(_PNP_MODEL, img_pts, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
        return HeadPose(rvec.ravel(), tvec.ravel()) if ok else None

    def close(self) -> None:
        self._landmarker.close()
