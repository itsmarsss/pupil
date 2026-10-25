import numpy as np

from pupil.filter.smoother import PRESETS, GazeSmoother

MS = 1_000_000
PPD = 48.0  # points per degree


def stream(points, start=0):
    return [(x, y, start + i * 33 * MS) for i, (x, y) in enumerate(points)]


def test_hold_removes_fixation_jitter():
    rng = np.random.default_rng(0)
    noisy = 500 + rng.normal(scale=0.5 * PPD, size=(60, 2))  # 0.5° noise around one point
    sm = GazeSmoother(PRESETS["medium"])
    out = np.array([sm(x, y, t, PPD) for x, y, t in stream(noisy)])
    raw_jitter = np.hypot(*np.diff(noisy[20:], axis=0).T).mean()
    out_jitter = np.hypot(*np.diff(out[20:], axis=0).T).mean()
    assert out_jitter < raw_jitter / 5


def test_hold_follows_a_real_jump_quickly():
    sm = GazeSmoother(PRESETS["medium"])
    pts = [(500, 500)] * 20 + [(900, 500)] * 10  # ~8° saccade
    out = [sm(x, y, t, PPD) for x, y, t in stream(pts)]
    arrived = next(i for i, (x, _) in enumerate(out) if i >= 20 and abs(x - 900) < PPD)
    assert arrived - 20 <= 3


def test_single_outlier_does_not_move_the_ring():
    sm = GazeSmoother(PRESETS["medium"])
    pts = [(500, 500)] * 20 + [(900, 500)] + [(500, 500)] * 5
    out = [sm(x, y, t, PPD) for x, y, t in stream(pts)]
    assert all(abs(x - 500) < 5 for x, _ in out[19:])


def test_off_passes_through():
    sm = GazeSmoother(PRESETS["off"])
    assert sm(123.0, 456.0, 0, PPD) == (123.0, 456.0)
