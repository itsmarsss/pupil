"""Calibration features: the gaze model's vector, a geometric block, and a head-pose block.

Layout: [model features | geometric features (GEO_DIM) | head pose (POSE_DIM)]. The geometric block comes from
the eyeball model (geometry/eyeball.py); on replayed sessions it cut the learned calibrator's error ~8–10 %, as
the two approaches' errors are only weakly correlated.

The model's features change when the head turns even if gaze doesn't, and a fit made
with the head still extrapolates badly once it moves. Giving the calibrator explicit
head pose (and gaze × pose terms) lets it learn to compensate, provided calibration
includes some head movement.
"""
from __future__ import annotations

import numpy as np

from ..core.types import FaceObservation

FEATURE_SET = "geo-v1"  # bump when the layout changes; old profiles then recalibrate

# Pose block layout, appended after the model vector.
YAW, PITCH, ROLL, FACE_X, FACE_Y, FACE_SIZE = range(6)
POSE_NAMES = ["yaw", "pitch", "roll", "face_x", "face_y", "face_size", "gx*yaw", "gy*yaw", "gx*pitch", "gy*pitch"]
POSE_DIM = len(POSE_NAMES)


def pose_block(face: FaceObservation, raw_xy: tuple[float, float]) -> np.ndarray:
    """Head yaw/pitch/roll (deg), face centre (-0.5..0.5 of frame), face width (fraction
    of frame, a distance proxy), and the model's own estimate times yaw/pitch."""
    yaw, pitch, roll = face.head_pose.yaw_pitch_roll_deg if face.head_pose is not None else (0.0, 0.0, 0.0)
    w, h = face.frame_size
    fx, fy, fw, fh = face.face_rect
    gx, gy = raw_xy
    return np.array(
        [yaw, pitch, roll, (fx + fw / 2) / w - 0.5, (fy + fh / 2) / h - 0.5, fw / w,
         gx * yaw, gy * yaw, gx * pitch, gy * pitch],
        dtype=np.float32,
    )


def calibration_vector(model_vector: np.ndarray, face: FaceObservation, geo: np.ndarray) -> np.ndarray:
    """geo: the eyeball model's block (geometry.eyeball.geo_block)."""
    return np.concatenate([model_vector.astype(np.float32), geo.astype(np.float32),
                           pose_block(face, (float(model_vector[0]), float(model_vector[1])))])


def pose_of(X: np.ndarray) -> np.ndarray:
    """The pose block of calibration vectors (n x POSE_DIM)."""
    return np.atleast_2d(X)[:, -POSE_DIM:]


class PoseRange:
    """Range of head yaw/pitch seen in calibration; confidence falls off outside it."""

    MARGIN_DEG = 3.0  # free slack beyond the calibrated range
    FALLOFF_DEG = 10.0  # confidence reaches 0 this far beyond the slack

    def __init__(self, X: np.ndarray):
        pose = pose_of(X)
        self.lo = np.percentile(pose[:, [YAW, PITCH]], 2, axis=0)
        self.hi = np.percentile(pose[:, [YAW, PITCH]], 98, axis=0)

    def outside_deg(self, vector: np.ndarray) -> float:
        yp = pose_of(vector)[0, [YAW, PITCH]]
        beyond = np.maximum(self.lo - yp, 0) + np.maximum(yp - self.hi, 0)
        return float(np.max(beyond))

    def confidence(self, vector: np.ndarray) -> float:
        excess = self.outside_deg(vector) - self.MARGIN_DEG
        return float(np.clip(1 - excess / self.FALLOFF_DEG, 0.0, 1.0))
