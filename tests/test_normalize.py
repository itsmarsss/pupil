import math
from pathlib import Path

import numpy as np
import pytest

from pupil.core.types import FaceObservation, Frame
from pupil.face.normalize import apply_warp, camera_matrix, frontal_warp, roll_warp
from pupil.face.tracker import L_INNER, L_OUTER, R_INNER, R_OUTER

FIXTURE = Path(__file__).resolve().parents[1] / "dev" / "fixtures" / "face.jpg"


def tilted_face(angle_deg=15.0):
    lm = np.zeros((478, 3))
    lm[:, :2] = (320, 260)
    a = math.radians(angle_deg)
    for idx, dx in ((L_OUTER, -60), (L_INNER, -20), (R_INNER, 20), (R_OUTER, 60)):
        lm[idx, :2] = (320 + dx * math.cos(a), 240 + dx * math.sin(a))
    return FaceObservation(0, (640, 480), lm, (200, 150, 240, 240), (0, 0, 1, 1), (0, 0, 1, 1), (0.3, 0.3), 12.0)


def test_roll_warp_levels_the_eyes():
    face = tilted_face(15)
    frame = Frame(0, np.zeros((480, 640, 3), np.uint8))
    _, warped = apply_warp(frame, face, roll_warp(face))
    lm = warped.landmarks
    assert abs(lm[L_OUTER, 1] - lm[R_OUTER, 1]) <= 1  # level after warping
    assert abs(face.landmarks[L_OUTER, 1] - face.landmarks[R_OUTER, 1]) > 25


def test_apply_warp_rejects_eyes_off_frame():
    face = tilted_face(0)
    frame = Frame(0, np.zeros((480, 640, 3), np.uint8))
    shift = np.array([[1, 0, 600], [0, 1, 0], [0, 0, 1]], float)
    assert apply_warp(frame, face, shift) is None


@pytest.mark.skipif(not FIXTURE.exists(), reason="local face fixture not present")
def test_frontal_warp_centres_face():
    import cv2

    from pupil.face.tracker import FaceTracker

    img = cv2.cvtColor(cv2.imread(str(FIXTURE)), cv2.COLOR_BGR2RGB)
    canvas = np.zeros((1080, 1440, 3), np.uint8)
    canvas[100 : 100 + img.shape[0], 150 : 150 + img.shape[1]] = img  # face well off-centre
    frame = Frame(1, canvas)
    face = FaceTracker(smooth=False).process(frame)
    H = frontal_warp(face, camera_matrix(1440, 1080, face.focal_px))
    _, warped = apply_warp(frame, face, H)
    x, y, w, h = warped.face_rect
    assert abs(x + w / 2 - 720) < 60 and abs(y + h / 2 - 540) < 60
