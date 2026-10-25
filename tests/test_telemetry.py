import json
from pathlib import Path

import numpy as np
import pytest

from gaze.core.device import Device
from gaze.core.screen import Screen
from gaze.geometry.eyeball import ScreenPlane

FIXTURE = Path(__file__).resolve().parents[1] / "dev" / "fixtures" / "face.jpg"
SCREEN = Screen(1512, 982, 302.0, 196.1)
PLANE = ScreenPlane(SCREEN)


def test_scene_info_corners_frame_the_camera():
    from gaze.telemetry import scene_info

    info = scene_info(SCREEN, PLANE, Device(), 110.0)
    c = np.array(info["screen"]["corners_cm"])
    assert c.shape == (4, 3)
    assert c[0, 1] > 0 > c[2, 1]  # camera sits just below the top edge
    assert c[0, 0] > 0 > c[1, 0]  # +x is the user's left, so the screen's left edge is at +x
    json.dumps(info)


@pytest.mark.skipif(not FIXTURE.exists(), reason="local face fixture not present")
def test_frame_state_from_a_real_frame():
    import cv2

    from gaze.calibration.profile import EXPLICIT, Profile, ProfileMeta
    from gaze.core.paths import GAZE_MODEL_FILE, model_path
    from gaze.core.types import Frame
    from gaze.face.tracker import FaceTracker
    from gaze.telemetry import calibration_info, frame_state
    from gaze.tracker import Tracker

    if not model_path(GAZE_MODEL_FILE).exists():
        pytest.skip("gaze model weights not present")
    from gaze.models.mnn_face_eye import MNNFaceEyeModel

    img = cv2.cvtColor(cv2.imread(str(FIXTURE)), cv2.COLOR_BGR2RGB)
    tracker = Tracker(source=None, face_tracker=FaceTracker(smooth=False), model=MNNFaceEyeModel(), screen=SCREEN)
    sample = tracker.process(Frame(1_000_000_000, img))
    assert sample.calib is not None and "reconstruction" in sample.extra

    uncal = frame_state(sample, None, tracker.plane, {"phase": "idle"})
    assert uncal["state"] == "uncalibrated" and uncal["phase"] == "idle"
    assert len(uncal["image"]["landmarks"]) == 478 and len(uncal["scene"]["points"]) == 478

    rng = np.random.default_rng(0)
    profile = Profile(ProfileMeta("t", "m", "c", (1512, 982)), len(sample.calib))
    profile.plane = tracker.plane
    for i, target in enumerate([(150, 100), (1360, 100), (756, 491), (150, 880), (1360, 880)]):
        X = sample.calib + rng.normal(0, 0.05, (8, len(sample.calib)))
        profile.add(X, target, EXPLICIT, measures=np.tile(sample.geo, (8, 1)))
    profile.fit()
    tracker.profile = profile
    state = frame_state(tracker.process(Frame(1_100_000_000, img)), profile, tracker.plane)
    assert state["gaze"]["ml"] is not None and "hits" in state["scene"]
    size = len(json.dumps(state))
    assert size < 60_000  # ~30 Hz over a local socket
    json.dumps(calibration_info(profile))
