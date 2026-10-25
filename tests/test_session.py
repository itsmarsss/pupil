from pathlib import Path

import numpy as np
import pytest

from pupil.calibration.targets import TargetTiming, grid
from pupil.core.screen import Screen
from pupil.core.types import Frame, GazePoint, Sample, TrackingState
from pupil.eval.session import SessionWriter, read_session

FIXTURE = Path(__file__).resolve().parents[1] / "dev" / "fixtures" / "face.jpg"
MS = 1_000_000


def test_writer_skips_unlabelled_and_rebuilds_canvas(tmp_path):
    from pupil.core.types import FaceObservation

    yy, xx = np.mgrid[0:120, 0:160]
    img = np.dstack([xx, yy, (xx + yy) // 2]).astype(np.uint8)  # smooth, so JPEG is near-lossless
    face = FaceObservation(0, (160, 120), np.zeros((478, 3)), (60, 40, 30, 30), (0, 0, 1, 1), (0, 0, 1, 1), (0.3, 0.3), 5.0)
    sample = Sample(0, None, face, GazePoint(0, TrackingState.NO_FACE))
    w = SessionWriter(tmp_path, {"screen_pt": [100, 100]})
    w.on_frame(Frame(1, img), sample)  # not labelled: dropped
    w.label("calib", (10, 20), 0)
    w.on_frame(Frame(2, img), sample)
    w.close()
    frames = list(read_session(tmp_path))
    assert len(frames) == 1 and frames[0].target == (10.0, 20.0)
    canvas = frames[0].image
    assert canvas.shape == img.shape
    x, y, rw, rh = 48, 28, 54, 54  # 30 px face x 1.8 around its centre
    diff = np.abs(canvas[y : y + rh, x : x + rw].astype(int) - img[y : y + rh, x : x + rw].astype(int))
    assert diff.mean() < 2
    assert canvas[0, 0].sum() == 0


@pytest.mark.skipif(not FIXTURE.exists(), reason="local face fixture not present")
def test_replay_runs_all_variants(tmp_path):
    import cv2

    from pupil.core.paths import PUPIL_GAZE_MODEL, model_path
    from pupil.eval.replay import replay
    from pupil.face.tracker import FaceTracker

    if not model_path(PUPIL_GAZE_MODEL).exists():
        pytest.skip("gaze model weights not present")
    from pupil.models.mnn_face_eye import MNNFaceEyeModel

    face_img = cv2.cvtColor(cv2.imread(str(FIXTURE)), cv2.COLOR_BGR2RGB)
    canvas = np.zeros((1080, 1440, 3), np.uint8)
    canvas[200 : 200 + face_img.shape[0], 400 : 400 + face_img.shape[1]] = face_img
    timing = TargetTiming(dwell_s=0.5, settle_s=0.1, tail_s=0.0)
    screen = Screen(1512, 982, 302, 196)
    w = SessionWriter(tmp_path, {"screen_pt": [1512, 982], "screen_mm": [302, 196], "timing": vars(timing)})
    tracker = FaceTracker()
    t = 1_000 * MS
    for phase, pts in (("calib", grid(5, 1512, 982)), ("valid", grid(3, 1512, 982))):
        for target in pts:
            w.label(phase, target, t)
            for _ in range(10):
                t += 40 * MS
                frame = Frame(t, canvas)
                face = tracker.process(frame)
                w.on_frame(frame, Sample(t, None, face, GazePoint(t, TrackingState.NO_FACE)))
            t += 200 * MS
    w.close()

    rows = replay(tmp_path, MNNFaceEyeModel(), screen)
    assert {r[0] for r in rows} == {"hd", "vga"}
    assert {r[1] for r in rows} == {"dots", "dots+pose", "live"}  # no head-motion phase recorded
    assert all(r[4].data_loss < 0.5 for r in rows) and {r[3] for r in rows} == {"all"}
    rows = replay(tmp_path, MNNFaceEyeModel(), screen, modes=("hd",), all_calibrators=True)
    assert len(rows) == 5 + 8 + 1  # dots: auto + raw + 3 model; dots+pose: those + 3 full; live: auto
