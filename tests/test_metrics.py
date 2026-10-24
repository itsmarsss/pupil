import numpy as np
import pytest

from gaze.core.screen import Screen
from gaze.eval.metrics import append_result, score

SCREEN = Screen(1512, 982, 302.0, 196.1)


def test_perfect_gaze_scores_zero():
    per_target = {(100.0, 100.0): np.tile([100.0, 100.0], (30, 1))}
    r = score(per_target, SCREEN, 550, frames_total=30)
    assert r.accuracy_deg == pytest.approx(0)
    assert r.precision_rms_deg == pytest.approx(0)
    assert r.data_loss == 0


def test_offset_converts_to_degrees():
    pt_1deg = SCREEN.deg_to_pt(1.0, 550)
    per_target = {(500.0, 500.0): np.tile([500.0 + pt_1deg, 500.0], (10, 1))}
    r = score(per_target, SCREEN, 550, frames_total=20)
    assert r.accuracy_deg == pytest.approx(1.0, rel=1e-3)
    assert r.data_loss == pytest.approx(0.5)


def test_jitter_shows_in_precision_not_accuracy():
    rng = np.random.default_rng(0)
    pts = np.array([300.0, 300.0]) + rng.normal(scale=20, size=(200, 2))
    pts -= pts.mean(axis=0) - [300, 300]
    r = score({(300.0, 300.0): pts}, SCREEN, 550, frames_total=200)
    assert r.accuracy_deg < 0.01
    assert r.precision_rms_deg > 0.3


def test_append_result(tmp_path):
    r = score({(1.0, 1.0): np.ones((3, 2))}, SCREEN, 550, frames_total=3)
    path = tmp_path / "results.csv"
    append_result(path, r, mode="hd")
    append_result(path, r, mode="vga")
    lines = path.read_text().splitlines()
    assert len(lines) == 3 and lines[0].startswith("time,mode,accuracy_deg")
