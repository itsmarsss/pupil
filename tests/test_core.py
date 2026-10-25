
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

    monkeypatch.setenv("PUPIL_NOTCH", "0")
    assert env("NOTCH", "1") == "0"
    monkeypatch.setenv("PUPIL_NOTCH", "1")
    assert env("NOTCH", "1") == "1"


def test_dotenv_values_drop_inline_comments():
    from pupil.control.config import _dotenv_value

    assert _dotenv_value("30            # step limit for a normal task") == "30"
    assert _dotenv_value("anthonylu/accessible-decision-researcher") == "anthonylu/accessible-decision-researcher"
    assert _dotenv_value('"a # inside quotes"  # comment') == "a # inside quotes"
    assert _dotenv_value("sk-abc#123") == "sk-abc#123"  # a # inside a value (no space before it) stays
    assert _dotenv_value("") == ""


def test_face_pointing_follows_the_head_with_emulated_noise():
    from types import SimpleNamespace as NS

    import numpy as np

    from pupil.core.screen import Screen
    from pupil.core.types import TrackingState
    from pupil.geometry.eyeball import ScreenPlane
    from pupil.tracker import Tracker

    t = Tracker.__new__(Tracker)
    t.screen = Screen(width_pt=1512, height_pt=982, width_mm=302.4, height_mm=196.4)
    t.plane = ScreenPlane(t.screen)
    t.pointing, t.face_noise_deg, t.face_gain = "face", 0.6, 1.0
    t.face_neutral_pt = t.face_anchor_pt = None
    t._rng, t.distance_mm, t._last_point = np.random.default_rng(0), 550.0, None
    class _Raw:  # no smoothing: see the noise itself
        def __call__(self, x, y, t_ns, ppd):
            return x, y

        def reset(self):
            pass

    t._filter = _Raw()
    t._fixations = NS(update=lambda *a: None, reset=lambda: None)

    def measure(x_cm, y_cm):  # a head at 55 cm whose forward axis hits the screen plane at (x, y) cm
        nose = np.array([0.0, -5.0, 55.0])
        forward = nose - np.array([x_cm, y_cm, 0.0])  # the head frame's +z points away from the screen
        basis = np.eye(3)
        basis[:, 2] = forward / np.linalg.norm(forward)
        return NS(measure=NS(nose=nose, basis=basis))

    face = NS(distance_mm=550.0, is_blinking=False)
    centre_cm = t.plane.to_cm(756, 491)
    points = np.array([(p.raw_x, p.raw_y) for p in (t._face_point(i, face, measure(*centre_cm)) for i in range(400))])
    assert t._face_point(0, face, measure(*centre_cm)).state is TrackingState.TRACKING
    spread = points.std(axis=0)
    sigma = t.screen.deg_to_pt(0.6, 550.0)
    assert np.allclose(points.mean(axis=0), (756, 491), atol=sigma / 4)  # centred on where the head points
    assert np.allclose(spread, sigma, rtol=0.2)  # with ~0.6° of noise
    t.recentre_face(points + [40, -25], (756, 491))  # head pointed a little off while looking at the dot
    later = np.array([(p.raw_x, p.raw_y) for p in (t._face_point(i, face, measure(*centre_cm)) for i in range(400))])
    assert np.allclose(np.median(later + [40, -25], axis=0), (756, 491), atol=sigma / 3)  # now it lands on the dot
    t.face_gain = 2.5  # a head turn moves the point 2.5x as far from the neutral direction
    aside = t.plane.to_cm(756 + 40, 491)
    moved = np.mean([t._face_point(i, face, measure(*aside)).raw_x for i in range(400)])
    still = np.mean([t._face_point(i, face, measure(*centre_cm)).raw_x for i in range(400)])
    assert abs((moved - still) - 100) < sigma / 2
    assert t._face_point(0, None, None).state is TrackingState.NO_FACE
