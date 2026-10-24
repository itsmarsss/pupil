"""Image normalisation before the gaze model.

`frontal`: warp the frame as seen by a virtual camera that looks straight at the face
centre from a fixed distance, with the head's roll removed (Zhang, Sugano & Bulling,
"Revisiting Data Normalization for Appearance-Based Gaze Estimation", ETRA 2018). Face
position, distance and roll then stop changing the model's input.

`roll`: only rotate the image so the eyes are level (cheaper, smaller change).

Both transform the landmarks with the image and recompute the crops, so the model
adapter works unchanged. Head pose in the observation is kept from the original frame;
the calibration's pose features still describe the real head.
"""
from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np

from ..core.types import FaceObservation, Frame, GazeFeatures
from .tracker import L_INNER, L_OUTER, R_INNER, R_OUTER, _PNP_MODEL, eye_rects, face_rect

NORMALISED_DISTANCE_MM = 500.0
# Generic-model points averaged for the face centre: both outer eye corners, both mouth corners.
_CENTRE_ROWS = [2, 3, 4, 5]


def camera_matrix(w: int, h: int, hfov_deg: float) -> np.ndarray:
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def frontal_warp(face: FaceObservation, K: np.ndarray, distance_mm: float = NORMALISED_DISTANCE_MM) -> np.ndarray | None:
    """Homography from the camera image to the normalised virtual-camera image."""
    if face.head_pose is None:
        return None
    R, _ = cv2.Rodrigues(np.asarray(face.head_pose.rvec, np.float64))
    t = np.asarray(face.head_pose.tvec, np.float64).reshape(3, 1)
    centre = (R @ _PNP_MODEL[_CENTRE_ROWS].T + t).mean(axis=1)
    d = float(np.linalg.norm(centre))
    if d < 1e-6:
        return None
    z = centre / d  # virtual camera looks at the face centre
    head_x = R[:, 0]
    y = np.cross(z, head_x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)  # x follows the head's x axis, which removes roll
    Rn = np.stack([x, y, z])
    S = np.diag([1.0, 1.0, distance_mm / d])
    return K @ S @ Rn @ np.linalg.inv(K)  # same intrinsics for the virtual camera


def roll_warp(face: FaceObservation) -> np.ndarray:
    """3x3 rotation about the eye midpoint that levels the eye line."""
    lm = face.landmarks
    left = (lm[L_OUTER, :2] + lm[L_INNER, :2]) / 2
    right = (lm[R_INNER, :2] + lm[R_OUTER, :2]) / 2
    angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    mid = (left + right) / 2
    M = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), angle, 1.0)
    return np.vstack([M, [0, 0, 1]])


def apply_warp(frame: Frame, face: FaceObservation, H: np.ndarray) -> tuple[Frame, FaceObservation] | None:
    w, h = frame.width, frame.height
    image = cv2.warpPerspective(frame.image, H, (w, h), flags=cv2.INTER_LINEAR)
    pts = cv2.perspectiveTransform(face.landmarks[None, :, :2].astype(np.float64), H)[0]
    lm = face.landmarks.copy()
    lm[:, :2] = np.round(pts)
    l_rect, r_rect = eye_rects(lm)
    for x, y, rw, rh in (l_rect, r_rect):
        if x <= 0 or y <= 0 or x + rw >= w or y + rh >= h:
            return None
    warped = replace(face, landmarks=lm, face_rect=face_rect(lm, w, h), left_eye_rect=l_rect, right_eye_rect=r_rect)
    return Frame(frame.t_ns, image, frame.source_id), warped


class NormalisedModel:
    """Wraps a GazeModel: normalises each frame, then runs the model on the result."""

    def __init__(self, model, method: str = "frontal", hfov_deg: float = 54.0):
        if method not in ("frontal", "roll"):
            raise ValueError(method)
        self.model, self.method, self.hfov_deg = model, method, hfov_deg
        self.model_id = f"{model.model_id}+{method}"
        self._K: np.ndarray | None = None

    def infer(self, frame: Frame, face: FaceObservation) -> GazeFeatures | None:
        if self.method == "roll":
            H = roll_warp(face)
        else:
            if self._K is None or self._K[0, 2] != frame.width / 2:
                self._K = camera_matrix(frame.width, frame.height, self.hfov_deg)
            H = frontal_warp(face, self._K)
        out = apply_warp(frame, face, H) if H is not None else None
        if out is None:
            return None
        feats = self.model.infer(*out)
        return replace(feats, model_id=self.model_id) if feats is not None else None

    def close(self) -> None:
        self.model.close()
