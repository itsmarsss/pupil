import numpy as np

from gaze.calibration.implicit import evaluate_click, window
from gaze.core.types import FaceObservation, GazeFeatures, GazePoint, Sample, TrackingState

MS = 1_000_000
PT_PER_DEG = 48.0


def face(open_=0.3):
    return FaceObservation(0, (640, 480), np.zeros((478, 3)), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (open_, open_), 10.0)


def sample(t, x, y, fixation=0, state=TrackingState.TRACKING, open_=0.3):
    feats = GazeFeatures(t, np.full(4, t, np.float32), (0, 0), "m")
    return Sample(t, feats, face(open_), GazePoint(t, state, x, y, x, y, fixation))


def run(**kw):
    return [sample(i * 33 * MS, **kw) for i in range(8)]


def test_accepts_fixation_near_click():
    d = evaluate_click(run(x=500, y=500), (520, 510), PT_PER_DEG)
    assert d.accepted and d.features.shape == (8, 4)
    assert d.error_deg < 1


def test_rejects_far_click():
    d = evaluate_click(run(x=100, y=100), (900, 700), PT_PER_DEG)
    assert not d.accepted and d.reason == "too_far"


def test_rejects_saccade():
    d = evaluate_click(run(x=500, y=500, fixation=None), (500, 500), PT_PER_DEG)
    assert not d.accepted and d.reason == "not_fixating"


def test_rejects_blink_and_uncalibrated():
    assert evaluate_click(run(x=1, y=1, open_=0.05), (1, 1), PT_PER_DEG).reason == "no_face"
    assert evaluate_click(run(x=1, y=1, state=TrackingState.UNCALIBRATED), (1, 1), PT_PER_DEG).reason == "uncalibrated"


def test_window_precedes_click():
    start, end = window(1000 * MS)
    assert start < end < 1000 * MS
