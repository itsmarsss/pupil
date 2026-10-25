"""Appearance features with no pretrained weights: small normalised eye patches.

Each eye crop is converted to grayscale, shrunk to a small patch, histogram-equalised and standardised, then the
two patches are flattened. The per-person calibration (ridge regression) learns the mapping to the screen, as in
WebGazer. The first two values are a crude gaze estimate from where each iris sits between its eye corners,
standing in for the "model's own estimate" that the pose block multiplies with head angles.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.types import FaceObservation, Frame, GazeFeatures
from ..face.tracker import L_INNER, L_OUTER, R_INNER, R_OUTER, crop

L_IRIS, R_IRIS = 468, 473  # image-left eye's iris centre (landmark 468 sits in the 33/133 eye), image-right eye's


def _iris_offset(lm: np.ndarray, outer: int, inner: int, iris: int) -> np.ndarray:
    a, b = lm[outer, :2], lm[inner, :2]
    width = np.linalg.norm(b - a) + 1e-6
    rel = lm[iris, :2] - (a + b) / 2
    return rel / width  # x across the eye, y below the corner line, in eye widths


class EyePatchModel:
    def __init__(self, size: tuple[int, int] = (16, 8)):  # 16x8 scored best of 16/24/32 on 138 replayed dots
        self.size = size
        self.model_id = f"eyepatch-{size[0]}x{size[1]}"

    def _patch(self, image: np.ndarray, rect) -> np.ndarray | None:
        p = crop(image, rect)
        if p is None or p.size == 0:
            return None
        g = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, self.size, interpolation=cv2.INTER_AREA)
        g = cv2.equalizeHist(g).astype(np.float32)
        return (g - g.mean()) / (g.std() + 1e-6)

    def infer(self, frame: Frame, face: FaceObservation) -> GazeFeatures | None:
        left = self._patch(frame.image, face.left_eye_rect)
        right = self._patch(frame.image, face.right_eye_rect)
        if left is None or right is None:
            return None
        lm = face.landmarks
        raw = (_iris_offset(lm, L_OUTER, L_INNER, L_IRIS) + _iris_offset(lm, R_INNER, R_OUTER, R_IRIS)) / 2
        vec = np.concatenate([raw, left.ravel(), right[:, ::-1].ravel()]).astype(np.float32)  # mirror the right eye
        return GazeFeatures(face.t_ns, vec, (float(raw[0]), float(raw[1])), self.model_id)

    def close(self) -> None:
        pass
