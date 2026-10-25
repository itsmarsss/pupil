
import numpy as np
import pytest

from pupil.core.screen import Screen
from pupil.filter.fixation import FixationDetector
from pupil.filter.one_euro import OneEuroFilter

MS = 1_000_000


@pytest.fixture
def screen():
    return Screen(width_pt=1512, height_pt=982, width_mm=302.0, height_mm=196.1)


def test_pt_deg_roundtrip(screen):
    pt = screen.deg_to_pt(2.0, 550)
    assert screen.pt_to_deg(pt, 550) == pytest.approx(2.0)
    # 1° at 55 cm is ~9.6 mm, ~48 pt on this screen
    assert 40 < screen.deg_to_pt(1.0, 550) < 55


def test_cocoa_flip(screen):
    assert screen.to_cocoa_y(0) == 982
    assert screen.to_cocoa_y(982) == 0


def test_one_euro_smooths_noise_at_rest():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    rng = np.random.default_rng(0)
    raw = 100 + rng.normal(0, 10, 300)
    out = [float(f(v, i * 33 * MS)) for i, v in enumerate(raw)]
    assert np.std(out[50:]) < np.std(raw[50:]) / 3


def test_one_euro_follows_step_with_beta():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    for i in range(30):
        f(0.0, i * 33 * MS)
    out = [float(f(500.0, (30 + i) * 33 * MS)) for i in range(10)]
    assert out[-1] > 450


def test_one_euro_resets_after_gap():
    f = OneEuroFilter()
    f(0.0, 0)
    assert float(f(100.0, 2_000 * MS)) == 100.0


def test_fixation_detector_splits_on_saccade():
    det = FixationDetector(threshold_deg_s=60, min_duration_ms=80)
    pt_per_deg = 48.0
    ids = []
    t = 0
    for _ in range(10):  # still at A
        ids.append(det.update(100, 100, t, pt_per_deg))
        t += 33 * MS
    ids.append(det.update(900, 500, t, pt_per_deg))  # saccade
    t += 33 * MS
    for _ in range(10):  # still at B
        ids.append(det.update(900, 500, t, pt_per_deg))
        t += 33 * MS
    first = {i for i in ids[:10] if i is not None}
    second = {i for i in ids[11:] if i is not None}
    assert ids[10] is None
    assert len(first) == 1 and len(second) == 1 and first != second


def test_fixation_requires_min_duration():
    det = FixationDetector(min_duration_ms=100)
    assert det.update(0, 0, 0, 48) is None
    assert det.update(0, 0, 33 * MS, 48) is None
    assert det.update(0, 0, 66 * MS, 48) is None
    assert det.update(0, 0, 133 * MS, 48) is not None


def test_app_support_moves_from_the_old_name_once(tmp_path, monkeypatch):
    from pupil.core import paths

    old, new = tmp_path / "Gaze", tmp_path / "Pupil"
    (old / "profiles").mkdir(parents=True)
    (old / "profiles" / "default.json").write_text("{}")
    monkeypatch.setattr(paths, "_OLD_APP_SUPPORT", old)
    monkeypatch.setattr(paths, "APP_SUPPORT", new)
    paths.migrate_app_support()
    assert (new / "profiles" / "default.json").exists() and not old.exists()
    old.mkdir()  # a stray old folder later never overwrites the new one
    paths.migrate_app_support()
    assert old.exists() and (new / "profiles" / "default.json").exists()


def test_settings_read_the_new_name_then_the_old(monkeypatch):
    from pupil.core.env import env

    monkeypatch.setenv("GAZE_NOTCH", "0")
    assert env("NOTCH", "1") == "0"
    monkeypatch.setenv("PUPIL_NOTCH", "1")
    assert env("NOTCH", "1") == "1"
