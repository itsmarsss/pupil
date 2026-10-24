import csv

import numpy as np

from gaze.core.types import GazePoint, Sample, TrackingState
from gaze.sinks.publisher import GazePublisher
from gaze.sinks.recorder import GazeRecorder
from gaze.sinks.snapper import Element, choose, probe_points

BUTTON_A = Element("AXButton", 100, 100, 80, 30)
BUTTON_B = Element("AXButton", 300, 100, 80, 30)


def test_snaps_to_element_under_gaze():
    assert choose([BUTTON_A, BUTTON_B], (120, 110), radius_pt=60) == BUTTON_A


def test_snaps_to_clearly_closest():
    assert choose([BUTTON_A, BUTTON_B], (200, 115), radius_pt=60) == BUTTON_A  # 20 vs 100 away


def test_refuses_ambiguous():
    assert choose([BUTTON_A, BUTTON_B], (240, 115), radius_pt=80) is None  # 60 vs 60


def test_ignores_far_and_huge():
    assert choose([BUTTON_B], (100, 400), radius_pt=60) is None
    huge = Element("AXCell", 0, 0, 1000, 800)
    assert choose([huge], (10, 10), radius_pt=60) is None


def test_prefers_smaller_when_nested():
    outer = Element("AXCell", 90, 90, 200, 60)
    assert choose([outer, BUTTON_A], (120, 110), radius_pt=60) == BUTTON_A


def test_probe_points_cover_radius():
    pts = np.array(probe_points(100, 100, 40))
    assert len(pts) == 17
    assert np.isclose(np.hypot(*(pts - 100).T).max(), 40)


def test_recorder_writes_samples_and_events(tmp_path):
    rec = GazeRecorder(tmp_path / "g.csv")
    rec.on_sample(Sample(1, None, None, GazePoint(1, TrackingState.TRACKING, 10.0, 20.0, 11.0, 21.0, 3, 550.0)))
    rec.on_sample(Sample(2, None, None, GazePoint(2, TrackingState.NO_FACE)))
    rec.event("click", 5, 6)
    rec.close()
    rows = list(csv.DictReader((tmp_path / "g.csv").open()))
    assert [r["state"] for r in rows] == ["tracking", "no_face", "event"]
    assert rows[0]["x"] == "10.0" and rows[0]["fixation_id"] == "3"
    assert rows[1]["x"] == "" and rows[2]["event"] == "click"


def test_publisher_sends_points_only_when_valid():
    pub = GazePublisher(busy=lambda: True, port=0)
    msg = pub.message(Sample(1, None, None, GazePoint(1, TrackingState.TRACKING, 10.0, 20.0, 11.0, 21.0, 3, 550.0, 0.8)))
    assert (msg["x"], msg["y"], msg["fixation"], msg["busy"]) == (10.0, 20.0, 3, True)
    assert "x" not in pub.message(Sample(2, None, None, GazePoint(2, TrackingState.BLINK, 10.0, 20.0)))
