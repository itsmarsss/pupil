from pathlib import Path

import numpy as np
import pytest

from pupil.capture.camera import crop_to_aspect
from pupil.face.tracker import L_INNER, L_OUTER, R_INNER, R_OUTER, eye_rects, face_rect

FIXTURE = Path(__file__).resolve().parents[1] / "dev" / "fixtures" / "face.jpg"


def _landmarks_with_eyes():
    lm = np.zeros((478, 3))
    lm[:, 0], lm[:, 1] = 300, 300
    lm[L_OUTER, :2] = (200, 300)
    lm[L_INNER, :2] = (250, 300)
    lm[R_INNER, :2] = (350, 300)
    lm[R_OUTER, :2] = (400, 300)
    return lm


def test_eye_rect_geometry():
    left, right = eye_rects(_landmarks_with_eyes())
    # inner-corner distance 100 → 20 px padding each side; height 0.75 x width; 60 % above
    assert left == (180, 259, 90, 68)
    assert right == (330, 259, 90, 68)


def test_face_rect_is_squared():
    lm = np.zeros((478, 3))
    lm[:240, :2] = (100, 100)
    lm[240:, :2] = (300, 260)  # 200 wide, 160 tall
    x, y, w, h = face_rect(lm, 640, 480)
    assert (x, y, w, h) == (110, 90, 180, 180)


def test_crop_to_aspect_centre():
    img = np.zeros((1080, 1920, 3), np.uint8)
    out = crop_to_aspect(img, 4 / 3)
    assert out.shape[:2] == (1080, 1440)
    assert crop_to_aspect(out, 4 / 3) is out


@pytest.mark.skipif(not FIXTURE.exists(), reason="local face fixture not present")
def test_tracker_and_model_on_fixture():
    import cv2

    from pupil.core.paths import GAZE_MODEL_FILE, model_path
    from pupil.core.types import Frame
    from pupil.face.tracker import FaceTracker

    if not model_path(GAZE_MODEL_FILE).exists():
        pytest.skip("gaze model weights not present")
    from pupil.models.mnn_face_eye import MNNFaceEyeModel

    img = cv2.cvtColor(cv2.imread(str(FIXTURE)), cv2.COLOR_BGR2RGB)
    frame = Frame(1_000_000_000, img)
    face = FaceTracker(smooth=False).process(frame)
    assert face is not None
    assert face.left_eye_rect[0] < face.right_eye_rect[0]
    yaw, pitch, roll = face.head_pose.yaw_pitch_roll_deg
    assert abs(yaw) < 30 and abs(pitch) < 30 and abs(roll) < 30
    feats = MNNFaceEyeModel().infer(frame, face)
    assert feats is not None and feats.vector.shape == (258,)
