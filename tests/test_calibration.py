import numpy as np
import pytest

from gaze.calibration.profile import CLICK, EXPLICIT, Profile, ProfileMeta
from gaze.calibration.regression import Ridge, fit_best
from gaze.calibration.targets import TargetTiming, grid, shuffled

DIM = 258
SCREEN = (1512.0, 982.0)


def synth(targets, per_target=20, signal="raw", noise=0.3, seed=0):
    """Features whose first two dims (or a hidden 20-d subspace) encode the target."""
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(2, 20))
    X, Y, g = [], [], []
    for i, (tx, ty) in enumerate(targets):
        u = np.array([tx / SCREEN[0] - 0.5, ty / SCREEN[1] - 0.5])
        for _ in range(per_target):
            f = rng.normal(size=DIM)
            if signal == "raw":
                f[:2] = 4 * u + rng.normal(scale=noise, size=2)
            else:  # spread over 20 dims, nothing in the first two
                f[:2] = rng.normal(size=2)
                f[10:30] = u @ mix * 3 + rng.normal(scale=noise, size=20)
            X.append(f)
            Y.append((tx, ty))
            g.append(i)
    return np.array(X), np.array(Y), np.ones(len(X)), np.array(g)


def test_ridge_recovers_affine_map():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    Y = X @ np.array([[100.0, 5.0], [-3.0, 80.0]]) + [700, 450]
    m = Ridge("raw", 1e-6).fit(np.hstack([X, rng.normal(size=(200, 5))]), Y)
    pred = m.predict(np.hstack([X, np.zeros((200, 5))]))
    assert np.abs(pred - Y).max() < 1e-3


def test_too_few_targets_fall_back_to_raw():
    X, Y, w, g = synth(grid(3, *SCREEN), signal="raw")
    model, report = fit_best(X, Y, w, g)
    assert report.chosen == "raw"


def test_prefers_stronger_regularisation_when_close():
    X, Y, w, g = synth(grid(9, *SCREEN), signal="raw")
    model, report = fit_best(X, Y, w, g)
    best = min(report.cv_error_pt.values())
    close = [n for n, e in report.cv_error_pt.items() if e <= best * 1.10]
    assert report.chosen == max(close, key=lambda n: float(n.split("=")[1]))


def test_rich_signal_chooses_full_model():
    rng = np.random.default_rng(3)
    targets = [tuple(p) for p in rng.uniform([0, 0], SCREEN, size=(80, 2))]
    X, Y, w, g = synth(targets, per_target=5, signal="hidden", noise=0.05)
    model, report = fit_best(X, Y, w, g)
    assert report.chosen.startswith("model")
    assert report.cv_error_pt[report.chosen] < 100


def test_profile_roundtrip(tmp_path):
    meta = ProfileMeta("me", "model@1234", "FaceTime HD Camera", SCREEN)
    p = Profile(meta, DIM)
    X, Y, _, g = synth(grid(9, *SCREEN))
    for gid in np.unique(g):
        p.add(X[g == gid], tuple(Y[g == gid][0]), EXPLICIT)
    assert p.fit() is not None and p.calibrated
    p.save(tmp_path)

    q = Profile.load("me", tmp_path)
    assert q is not None and q.calibrated
    assert q.meta.matches("model@1234", "FaceTime HD Camera", SCREEN)
    assert not q.meta.matches("other@0000", "FaceTime HD Camera", SCREEN)
    assert np.allclose(p.predict(X[0]), q.predict(X[0]))
    assert q.new_group() == 9


def test_profile_needs_three_explicit_targets():
    p = Profile(ProfileMeta("x", "m", "c", SCREEN), DIM)
    X, Y, _, g = synth(grid(5, *SCREEN))
    for gid in range(2):
        p.add(X[g == gid], tuple(Y[g == gid][0]), EXPLICIT)
    assert p.fit() is None and not p.calibrated


def test_click_eviction_keeps_explicit():
    p = Profile(ProfileMeta("x", "m", "c", SCREEN), DIM)
    p.MAX_CLICK_GROUPS = 3
    p.add(np.zeros((4, DIM)), (1, 1), EXPLICIT)
    for i in range(5):
        p.add(np.full((2, DIM), i), (i, i), CLICK)
    assert p.count(EXPLICIT) == 1
    assert p.count(CLICK) == 3
    assert set(p.Y[p.sources == CLICK][:, 0]) == {2, 3, 4}


def test_grid_and_timing():
    pts = grid(9, *SCREEN)
    assert len(pts) == 9 and pts[0] == pytest.approx((756, 491))  # centre first, to orient
    g25 = grid(25, *SCREEN)
    assert len(g25) == 25 and g25[0] == pytest.approx((756, 491))
    g16 = grid(16, *SCREEN)
    assert len(g16) == 16 and len(set(g16)) == 16
    assert min(x for x, _ in g16) == pytest.approx(151.2) and max(y for _, y in g16) == pytest.approx(883.8)
    s = shuffled(grid(5, *SCREEN), seed=0)
    assert s[0] == grid(5, *SCREEN)[0] and sorted(s) == sorted(grid(5, *SCREEN))
    t = TargetTiming(dwell_s=1.0, settle_s=0.4, tail_s=0.1)
    assert not t.accepts(0, 300_000_000)
    assert t.accepts(0, 500_000_000)
    assert not t.accepts(0, 950_000_000)


def test_recentre_shifts_predictions_onto_the_dot():
    p = Profile(ProfileMeta("x", "m", "c", SCREEN), DIM)
    X, Y, _, g = synth(grid(9, *SCREEN))
    for gid in np.unique(g):
        p.add(X[g == gid], tuple(Y[g == gid][0]), EXPLICIT)
    p.fit()
    before = np.array(p.predict(X[0]))
    drifted = np.tile(before + [40.0, -25.0], (10, 1))  # everything 40 pt right, 25 up
    p.recentre(drifted, tuple(before))
    assert np.allclose(p.meta.offset_pt, (-40.0, 25.0))
    assert np.allclose(p.predict(X[0]), before - [40, -25])
    with pytest.raises(ValueError):
        p.recentre(np.tile(before + [900.0, 0.0], (5, 1)), tuple(before))  # too far: recalibrate instead


def test_each_camera_gets_its_own_profile(tmp_path, monkeypatch):
    import json

    from gaze.calibration import profile as profile_mod
    from gaze.ui.flows import Flows

    monkeypatch.setattr(profile_mod, "profiles_dir", lambda: tmp_path)
    flows = Flows.__new__(Flows)  # only the naming logic, no UI
    flows.base_profile = "default"
    assert flows._profile_name_for("FaceTime HD Camera") == "default"  # nothing saved yet: base name
    (tmp_path / "default").mkdir()
    (tmp_path / "default" / "meta.json").write_text(json.dumps({"camera": "FaceTime HD Camera"}))
    assert flows._profile_name_for("FaceTime HD Camera") == "default"
    assert flows._profile_name_for("OBS Virtual Camera") == "default-obs-virtual-camera"


def test_profile_save_is_atomic_and_damaged_files_load_as_none(tmp_path, monkeypatch):
    import numpy as np
    import pytest

    from gaze.calibration import profile as profile_mod
    from gaze.calibration.profile import Profile, ProfileMeta

    p = Profile(ProfileMeta("p", "m", "cam", (100.0, 100.0)), 4)
    p.add(np.ones((3, 4), np.float32), (10.0, 20.0), profile_mod.EXPLICIT)
    p.save(tmp_path)
    good = (tmp_path / "p" / "samples.npz").read_bytes()

    def disk_full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(profile_mod.np, "savez_compressed", disk_full)
    with pytest.raises(OSError):
        p.save(tmp_path)
    assert (tmp_path / "p" / "samples.npz").read_bytes() == good  # the old profile survives
    assert not list((tmp_path / "p").glob("*.tmp"))
    monkeypatch.undo()

    (tmp_path / "p" / "samples.npz").write_bytes(b"PK\x05\x06" + b"\0" * 18)  # an empty zip, as the full disk left
    assert Profile.load("p", tmp_path) is None
