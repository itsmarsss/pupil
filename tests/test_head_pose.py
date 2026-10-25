import json

import numpy as np
import pytest

from pupil.calibration.features import POSE_DIM, YAW, PoseRange, calibration_vector, pose_of
from pupil.calibration.profile import EXPLICIT, MOTION, Profile, ProfileMeta
from pupil.calibration.regression import fit_best
from pupil.calibration.targets import grid
from pupil.core.types import FaceObservation, HeadPose
from pupil.geometry.eyeball import GEO_DIM

SCREEN = (1512.0, 982.0)
MODEL_DIM = 30


def face_with(yaw_deg=0.0):
    rvec = np.array([0.0, np.radians(yaw_deg), 0.0])
    return FaceObservation(0, (1440, 1080), np.zeros((478, 3)), (600, 400, 240, 240), (0, 0, 1, 1), (0, 0, 1, 1),
                           (0.3, 0.3), 30.0, HeadPose(rvec, np.zeros(3)))


def test_calibration_vector_layout():
    v = calibration_vector(np.arange(MODEL_DIM, dtype=np.float32), face_with(10), np.full(GEO_DIM, 7.0))
    assert v.shape == (MODEL_DIM + GEO_DIM + POSE_DIM,)
    assert np.all(v[MODEL_DIM : MODEL_DIM + GEO_DIM] == 7.0)  # geometric block sits between model and pose
    pose = pose_of(v)[0]
    assert pose[YAW] == pytest.approx(10, abs=0.5)
    assert pose[3] == pytest.approx((600 + 120) / 1440 - 0.5)  # face centre x


def test_pose_range_confidence():
    X = np.zeros((100, MODEL_DIM + POSE_DIM))
    X[:, -POSE_DIM + YAW] = np.linspace(-5, 5, 100)
    r = PoseRange(X)
    inside, edge, far = (np.zeros((1, MODEL_DIM + POSE_DIM)) for _ in range(3))
    edge[0, -POSE_DIM + YAW] = 5 + PoseRange.MARGIN_DEG
    far[0, -POSE_DIM + YAW] = 30
    assert r.confidence(inside) == 1.0
    assert r.confidence(edge) == pytest.approx(1.0, abs=0.05)
    assert r.confidence(far) == 0.0


def synth(targets, yaws, rng, per=20):
    """Model features shift with yaw as well as gaze, like the real ones do."""
    X, Y, g = [], [], []
    for i, ((tx, ty), yaw) in enumerate(zip(targets, yaws)):
        for _ in range(per):
            yj = yaw + rng.normal(scale=0.5)
            u = np.array([tx / SCREEN[0] - 0.5, ty / SCREEN[1] - 0.5])
            model = rng.normal(scale=0.05, size=MODEL_DIM)
            model[:2] = 4 * u + 0.15 * yj  # head turn pushes the raw estimate sideways
            pose = np.zeros(POSE_DIM)
            pose[YAW] = yj
            pose[6:] = [model[0] * yj, model[1] * yj, 0, 0]
            X.append(np.concatenate([model, pose]))
            Y.append((tx, ty))
            g.append(i)
    return np.array(X), np.array(Y, float), np.array(g)


def test_motion_sweep_fixes_head_turn_extrapolation():
    rng = np.random.default_rng(0)
    dots = grid(9, *SCREEN)
    Xd, Yd, gd = synth(dots, [0.0] * 9, rng)  # dots with the head still
    centre = [(SCREEN[0] / 2, SCREEN[1] / 2)] * 12
    Xm, Ym, _ = synth(centre, np.linspace(-15, 15, 12), rng, per=10)  # look at centre, turn head
    Xt, Yt, _ = synth(dots, [12.0] * 9, rng, per=10)  # test: head turned 12°

    def err(model):
        return float(np.median(np.hypot(*(model.predict(Xt) - Yt).T)))

    no_pose, _ = fit_best(Xd, Yd, np.ones(len(Xd)), gd, POSE_DIM)  # pose block present but unused
    X = np.vstack([Xd, Xm])
    Y = np.vstack([Yd, Ym])
    g = np.concatenate([gd, np.full(len(Xm), 99)])
    subs = np.concatenate([gd, 99_000 + np.arange(len(Xm)) // 10])
    w = np.concatenate([np.ones(len(Xd)), np.full(len(Xm), 0.5)])
    with_pose, report = fit_best(X, Y, w, g, POSE_DIM, score_groups=subs, use_pose=True)
    assert err(no_pose) > 150  # dots-only fit is far off with the head turned
    assert err(with_pose) < err(no_pose) / 3
    assert report.chosen.split("/")[0] in ("rawpose", "full")


def test_profile_motion_chunks_and_old_profiles(tmp_path):
    rng = np.random.default_rng(1)
    meta = ProfileMeta("p", "m", "cam", SCREEN)
    p = Profile(meta, MODEL_DIM + POSE_DIM)
    X, Y, g = synth(grid(5, *SCREEN), [0.0] * 5, rng)
    for i in range(5):
        p.add(X[g == i], tuple(Y[g == i][0]), EXPLICIT)
    Xm, _, _ = synth([(756, 491)], [0.0], rng, per=45)
    p.add(Xm, (756, 491), MOTION, chunk=15)
    assert len(np.unique(p.subs[p.sources == MOTION])) == 3
    assert p.fit() is not None and p.pose_range is not None
    p.clear(EXPLICIT, MOTION)
    assert len(p.X) == 0

    # a profile saved before head-pose features: loads, but doesn't match, so the app recalibrates
    d = tmp_path / "old"
    d.mkdir()
    old = {"name": "old", "model_id": "m", "camera": "cam", "screen_pt": list(SCREEN)}
    (d / "meta.json").write_text(json.dumps(old))
    np.savez(d / "samples.npz", X=np.zeros((3, 258), np.float32), Y=np.zeros((3, 2)), groups=np.arange(3),
             sources=np.array(["explicit"] * 3), t=np.zeros(3))
    q = Profile.load("old", tmp_path)
    assert q is not None and not q.meta.matches("m", "cam", SCREEN) and not q.calibrated
