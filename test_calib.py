"""Synthetic checks for calibration. Faces include iris ring, eyelid and rigid landmarks. Data mimic the first
real run (vertical rotation shows up at ~0.3x, horizontal ~0.8x, plus kappa) and, optionally, eyelid occlusion
that drags MediaPipe's iris-centre landmark vertically while the iris's left/right edge points stay put."""
import numpy as np

import calib
import features as F
import geometry as g
from test_geometry import project, rot_y

TRUE = {"right": g.EyeParams(kappa_yaw_deg=3.5, kappa_pitch_deg=-4.0, gain_yaw=1.25, gain_pitch=3.3),
        "left": g.EyeParams(kappa_yaw_deg=2.0, kappa_pitch_deg=-3.0, gain_yaw=1.2, gain_pitch=3.5)}
SCREEN = g.Screen(30.12, 18.93, 1512, 950, cam_to_top_cm=0.32)
INTR = g.Intrinsics.from_hfov(1280, 720, 64.0)
CORNERS = {"right": (g.R_EYE_CORNERS, np.array([[-4.5, 3.0, 2.0], [-1.5, 3.0, 2.0]])),
           "left": (g.L_EYE_CORNERS, np.array([[1.5, 3.0, 2.0], [4.5, 3.0, 2.0]]))}
RIGID_HEAD = np.random.default_rng(42).uniform([-6, -6, -1], [6, 5, 4], size=(len(F.RIGID), 3))


def make(head_t, R, target, rng=None, lid_px=0.0, eyes=TRUE, intr=INTR):
    """Landmarks + pose for a head whose visual axes hit `target`. lid_px: eyelid drag of the iris-centre landmark."""
    lms = np.zeros((478, 3))
    depth = abs(head_t[2])
    B = g.head_basis(R)
    P = lambda p: project(p, depth, intr)
    for name, ((a, b), corners) in CORNERS.items():
        cam = (R @ corners.T).T + head_t
        lms[a], lms[b] = P(cam[0]), P(cam[1])
        ep = eyes[name]
        c = cam.mean(axis=0) - B[:, 2] * ep.center_depth_cm
        d_opt = g.invert_eye_model(B, g.normalize(target - c), ep)
        iris = c + ep.eyeball_radius_cm * d_opt
        ring = F.R_IRIS_RING if name == "right" else F.L_IRIS_RING
        for i, off in zip(ring, ([0.55, 0, 0], [0, 0.55, 0], [-0.55, 0, 0], [0, -0.55, 0])):
            lms[i] = P(iris + B @ np.array(off))
        center_idx = g.R_IRIS_CENTER if name == "right" else g.L_IRIS_CENTER
        lms[center_idx] = P(iris)
        lms[center_idx, 1] += lid_px / intr.height
        # eyelids for eye openness
        ear_idx = F.R_EAR if name == "right" else F.L_EAR
        mid = cam.mean(axis=0)
        for i, off in zip(ear_idx[1:3], ([-0.5, 0.45, 0], [0.5, 0.45, 0])):
            lms[i] = P(mid + B @ np.array(off))
        for i, off in zip(ear_idx[4:6], ([0.5, -0.45, 0], [-0.5, -0.45, 0])):
            lms[i] = P(mid + B @ np.array(off))
    for i, p in zip(F.RIGID, RIGID_HEAD):
        lms[i] = P(R @ p + head_t)
    if rng is not None:
        lms[:, :2] += rng.normal(0, 0.0004, size=(478, 2))  # ~0.5 px landmark noise
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [head_t[0], head_t[1], head_t[2] / g.mediapipe_depth_scale(intr)]
    return lms, pose


CALIB = [(fx, fy) for fy in (0.04, 0.5, 0.96) for fx in (0.04, 0.5, 0.96)] + [(0.27, 0.3), (0.73, 0.3), (0.27, 0.7), (0.73, 0.7)]
TEST = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92), (0.3, 0.06), (0.94, 0.3), (0.7, 0.94), (0.06, 0.7),
        (0.4, 0.4), (0.62, 0.4), (0.38, 0.62), (0.6, 0.6)]


def data(rng, lid_sd=0.0, blinks=0.0, test_frames=30, clean_test=False):
    fix = []
    for gi, (fx, fy) in enumerate(CALIB):
        t = SCREEN.from_px(np.array([fx * 1512, fy * 950]))
        for _ in range(10):
            lms, pose = make(np.array([1.0, -9.0, -50.0]) + rng.normal(0, 0.3, 3), rot_y(5), t, rng, rng.normal(0, lid_sd))
            if rng.random() < blinks:  # blink: lids nearly shut, iris landmarks unreliable
                for idx in (F.R_EAR, F.L_EAR):
                    lms[[idx[1], idx[2], idx[4], idx[5]], 1] = lms[idx[0], 1]
                lms[[g.R_IRIS_CENTER, g.L_IRIS_CENTER], 1] += 0.02
            fix.append(calib.Fixation(lms, pose, t, gi))
    test = []
    for fx, fy in TEST:
        t = SCREEN.from_px(np.array([fx * 1512, fy * 950]))
        test.append((t, [make(np.array([3.0, -8.0, -52.0]), rot_y(-4), t, None if clean_test else rng, rng.normal(0, lid_sd))
                         for _ in range(1 if clean_test else test_frames)]))
    return fix, test


def score(uc, test, intr):
    errs = []
    for t, frames in test:
        hits = [gf.hit for lms, pose in frames if (gf := calib.predict(lms, pose, intr, SCREEN, uc)) is not None and gf.hit is not None]
        errs.append(np.linalg.norm(np.mean(hits, axis=0)[:2] - t[:2]))
    return float(np.mean(errs))


def main():
    # 1) clean data, physical model only: parameters recovered, held-out error small
    fix, test = data(np.random.default_rng(0), clean_test=True)
    uc = calib.fit(fix, INTR, SCREEN, calib.GazeConfig("center", True, False, "none", False))
    for e in ("right", "left"):
        got, tr = uc.eyes[e], TRUE[e]
        print(f"{e:5s} kappa {got['kappa_yaw_deg']:+.2f}/{got['kappa_pitch_deg']:+.2f} (true {tr.kappa_yaw_deg:+.2f}/{tr.kappa_pitch_deg:+.2f})  "
              f"gain {got['gain_yaw']:.2f}/{got['gain_pitch']:.2f} (true {tr.gain_yaw:.2f}/{tr.gain_pitch:.2f})")
    err = score(uc, test, INTR)
    print(f"per-eye physical fit (noisy calibration, noise-free test frames, new head pose): held-out {err:.2f} cm")
    assert err < 0.4

    # 2) eyelid drag (sd 1 px on the centre landmark) + 5% blinks: ablation, each row adds one fix
    fix, test = data(np.random.default_rng(1), lid_sd=1.0, blinks=0.05)
    print("eyelid drag + blinks, held-out error by config:")
    res = {}
    for cfg in calib.ABLATION:
        res[cfg] = score(calib.fit(fix, INTR, SCREEN, cfg), test, INTR)
        print(f"  {cfg.label():40s} {res[cfg]:.2f} cm")
    assert res[calib.ABLATION[2]] < res[calib.ABLATION[1]]  # equator beats the dragged centre landmark
    full = calib.GazeConfig("equator", True, True, "quad", True, False)
    assert res[full] < res[calib.ABLATION[0]]

    # 3) drift correction recovers a head shift with one dot
    uc = calib.fit(fix, INTR, SCREEN, calib.ABLATION[-1])
    t = SCREEN.from_px(np.array([756.0, 475.0]))
    moved = [calib.Fixation(*make(np.array([6.0, -4.0, -46.0]), rot_y(8), t, np.random.default_rng(3)), t) for _ in range(10)]
    before = np.linalg.norm(np.mean([calib.predict(f.lms, f.pose, INTR, SCREEN, uc).hit for f in moved], axis=0)[:2] - t[:2])
    uc2 = calib.drift_correct(uc, moved, INTR, SCREEN)
    after = np.linalg.norm(np.mean([calib.predict(f.lms, f.pose, INTR, SCREEN, uc2).hit for f in moved], axis=0)[:2] - t[:2])
    print(f"drift correction at the centre dot: {before:.2f} -> {after:.2f} cm")
    assert after <= before + 1e-6


if __name__ == "__main__":
    main()


def rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def test_head_comp():
    """True eyeball centres sit deeper than the model assumes -> errors that grow with head movement, invisible
    to a still-head dot calibration. Head-sweep frames should let head compensation remove most of it."""
    from dataclasses import replace as rep
    deep = {e: rep(p, center_depth_cm=1.2) for e, p in TRUE.items()}
    rng = np.random.default_rng(5)
    home = np.array([1.0, -9.0, -50.0])
    fix = []
    for gi, (fx, fy) in enumerate(CALIB):
        t = SCREEN.from_px(np.array([fx * 1512, fy * 950]))
        for _ in range(10):
            fix.append(calib.Fixation(*make(home + rng.normal(0, 0.2, 3), rot_y(3), t, rng, eyes=deep), t, gi))
    for si, (fx, fy) in enumerate([(0.5, 0.5), (0.25, 0.5), (0.75, 0.5)]):  # head sweep: eyes on the dot, head moving
        t = SCREEN.from_px(np.array([fx * 1512, fy * 950]))
        for k in range(60):
            a = 2 * np.pi * k / 60
            R = rot_y(3 + 8 * np.cos(a)) @ rot_x(5 * np.sin(a))
            pos = home + [3 * np.sin(a), 2 * np.cos(2 * a), 0]
            fix.append(calib.Fixation(*make(pos, R, t, rng, eyes=deep), t, calib.SWEEP_GROUP0 + si))
    test = []
    for fx, fy in TEST:
        t = SCREEN.from_px(np.array([fx * 1512, fy * 950]))
        test.append((t, [make(home + [3.5, -1.5, 0], rot_y(-5) @ rot_x(4), t, rng, eyes=deep) for _ in range(10)]))
    off = score(calib.fit(fix, INTR, SCREEN, calib.GazeConfig(head_comp=False)), test, INTR)
    uc = calib.fit(fix, INTR, SCREEN, calib.GazeConfig(head_comp=True))
    on = score(uc, test, INTR)
    still = [(t, [make(home, rot_y(3), t, rng, eyes=deep) for _ in range(10)]) for t, _ in test]
    print(f"head moved after calibration: without head comp {off:.2f} cm -> with {on:.2f} cm "
          f"(head still: {score(uc, still, INTR):.2f} cm)")
    assert uc.head_comp is not None and on < off


if __name__ == "__main__":
    test_head_comp()
