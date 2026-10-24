"""Data passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Rects are (x, y, w, h) in frame pixels.
Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class Frame:
    t_ns: int  # monotonic capture time
    image: np.ndarray  # H x W x 3, RGB, uint8
    source_id: str = ""

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


@dataclass(frozen=True)
class HeadPose:
    rvec: np.ndarray  # Rodrigues rotation, camera frame
    tvec: np.ndarray  # translation in mm, camera frame

    @property
    def yaw_pitch_roll_deg(self) -> tuple[float, float, float]:
        import cv2

        r, _ = cv2.Rodrigues(self.rvec)
        sy = float(np.hypot(r[0, 0], r[1, 0]))
        pitch = float(np.degrees(np.arctan2(r[2, 1], r[2, 2])))
        yaw = float(np.degrees(np.arctan2(-r[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
        return yaw, pitch, roll


@dataclass(frozen=True)
class FaceObservation:
    t_ns: int
    frame_size: tuple[int, int]  # (w, h)
    landmarks: np.ndarray  # 478 x 3, frame pixels (z scaled like x)
    face_rect: Rect
    left_eye_rect: Rect  # eye on the image's left (subject's right eye)
    right_eye_rect: Rect
    eye_openness: tuple[float, float]  # (image-left, image-right), height / width
    iris_diameter_px: float
    head_pose: HeadPose | None = None
    distance_mm: float | None = None  # eye to camera, from iris size

    @property
    def is_blinking(self) -> bool:
        return min(self.eye_openness) < 0.12


@dataclass(frozen=True)
class GazeFeatures:
    t_ns: int
    vector: np.ndarray  # float32, model-specific length
    raw_xy: tuple[float, float]  # the model's own uncalibrated estimate
    model_id: str


class TrackingState(str, Enum):
    TRACKING = "tracking"
    NO_FACE = "no_face"
    BLINK = "blink"
    UNCALIBRATED = "uncalibrated"


@dataclass(frozen=True)
class GazePoint:
    t_ns: int
    state: TrackingState
    x: float = float("nan")  # screen points, top-left origin, filtered
    y: float = float("nan")
    raw_x: float = float("nan")  # before smoothing
    raw_y: float = float("nan")
    fixation_id: int | None = None
    distance_mm: float | None = None

    @property
    def valid(self) -> bool:
        return self.state is TrackingState.TRACKING and np.isfinite(self.x) and np.isfinite(self.y)


@dataclass
class Sample:
    """One inference result, kept in a short history for calibration and clicks."""

    t_ns: int
    features: GazeFeatures | None
    face: FaceObservation | None
    point: GazePoint
    extra: dict = field(default_factory=dict)
